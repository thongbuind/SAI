"""Công cụ so sánh của phòng Đánh giá.

``compare_attention_kernels``: so một attention kernel với kernel tham chiếu trên cùng input ngẫu nhiên
— sai khác output, sai khác gradient theo q/k/v, thời gian và bộ nhớ đỉnh của forward+backward. Kernel
không có tham số nên kết quả khớp nghĩa là có thể thay kernel mà dùng lại nguyên trọng số đã train.
``attention_shape`` lấy shape attention (H, Hkv, D) của một model để so trên đúng cỡ model đó.

Tác vụ ``compare-kernel`` (trang Kiến trúc của web lab, hợp đồng trong ``foundation_lab.core.tasks``) so một
kernel ứng viên với kernel mốc của từng model, trên shape attention của model đó. Kernel mốc luôn là
``attention_kernel`` trong config hiện tại của model, không gõ tay. Spec::

    candidate: eager
    models: [SAI_modular_35M, SAI_modular_375M]   # mặc định ARCH_CHECK_MODELS
    seq_len: 128

Kết quả ghi vào ``runs/compare-kernel/<run_id>/``: mỗi model một bản ghi và một nhóm số đo.

``compare_objectives``: so một objective (hàm loss trong ``OBJECTIVES``) với objective tham chiếu trên cùng hidden,
head weight và targets ngẫu nhiên — sai khác tổng loss, sai khác gradient theo hidden và head weight, thời gian
và bộ nhớ đỉnh của forward+backward.

Tác vụ ``compare-objective`` (trang Objective của web lab) so một objective ứng viên với objective mốc của một
stage trong từng experiment. Mốc luôn lấy từ config hiện tại của experiment: ``objective``, ``loss_chunk_size``
và ``max_microbatch_tokens`` trong ``params`` của stage (đã gộp ``stage_defaults`` và mặc định của recipe),
cùng ``d_model`` và ``vocab_size`` của model. Spec::

    candidate: lm_ce
    candidate_chunk_size: 1024                     # bỏ trống: như loss_chunk_size của mốc
    experiments: [SAI_35M, SAI_375M]               # mặc định OBJECTIVE_CHECK_EXPERIMENTS
    stage: pretrain

Kết quả ghi vào ``runs/compare-objective/<run_id>/``: mỗi experiment một bản ghi và một nhóm số đo.
"""

import math
import statistics
import time
from dataclasses import dataclass
from typing import Callable, Literal
import torch
from SAI.modeling import SAIConfig
from foundation_lab.architecture import resolve_model_config
from foundation_lab.architecture.registry import ATTENTION_KERNELS
from foundation_lab.core.config import ConfigError, apply_overrides, from_dict
from foundation_lab.core.experiment import StageSpec, resolve_config_ref, resolve_experiment_dict
from foundation_lab.core.tasks import TASKS, Task
from foundation_lab.engine.runtime import select_device
from foundation_lab.objectives import OBJECTIVES
from foundation_lab.paths import EXPERIMENT_CONFIG_DIR, MODEL_CONFIG_DIRS
from foundation_lab.state.events import EventLog
from foundation_lab.state.results import Metric, RunResult, write_result

MIB = 1024 ** 2
MS_PER_SECOND = 1000
KERNEL_COMMAND = "compare-kernel"
OBJECTIVE_COMMAND = "compare-objective"
# Mặc định đo trên shape attention của cả ba cỡ model; chỉ bản lắp ghép SAI_modular chọn kernel qua config.
ARCH_CHECK_MODELS = ("SAI_modular_35M", "SAI_modular_100M", "SAI_modular_375M")
# Khoá số đo trong report của compare_attention_kernels → (đơn vị, chiều tốt hơn). Số đo None (bộ nhớ đỉnh
# ngoài CUDA) không thành Metric.
KERNEL_METRICS = {
    "max_abs_diff_out": ("", "lower"),
    "max_abs_diff_grad_q": ("", "lower"),
    "max_abs_diff_grad_k": ("", "lower"),
    "max_abs_diff_grad_v": ("", "lower"),
    "matches_reference": ("", "higher"),
    "reference_ms": ("ms", "lower"),
    "candidate_ms": ("ms", "lower"),
    "speedup": ("×", "higher"),
    "reference_peak_mib": ("MiB", "lower"),
    "candidate_peak_mib": ("MiB", "lower"),
}
# Mặc định so trên cả ba cỡ model của pipeline chuẩn (core/configs/).
OBJECTIVE_CHECK_EXPERIMENTS = ("SAI_35M", "SAI_100M", "SAI_375M")
# Như KERNEL_METRICS, cho report của compare_objectives.
OBJECTIVE_METRICS = {
    "loss_rel_diff": ("", "lower"),
    "max_abs_diff_grad_hidden": ("", "lower"),
    "max_abs_diff_grad_weight": ("", "lower"),
    "matches_reference": ("", "higher"),
    "reference_ms": ("ms", "lower"),
    "candidate_ms": ("ms", "lower"),
    "speedup": ("×", "higher"),
    "reference_peak_mib": ("MiB", "lower"),
    "candidate_peak_mib": ("MiB", "lower"),
}

def _sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()

def _peak_memory_mib(device: torch.device, run) -> float | None:
    """Bộ nhớ đỉnh cấp phát thêm khi chạy ``run`` (MiB). Chỉ CUDA có bộ đếm đỉnh; CPU/MPS trả None."""
    if device.type != "cuda":
        return None
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)
    baseline = torch.cuda.memory_allocated(device)
    run()
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated(device) - baseline) / MIB

def _interleaved_median_ms(device: torch.device, run_reference, run_candidate, repeats: int) -> tuple[float, float]:
    """Trung vị thời gian (ms) của mốc và của ứng viên qua ``repeats`` lượt, mỗi lượt chạy mốc rồi ứng viên.

    Đo xen kẽ để tải máy đổi giữa chừng (tiến trình khác, nhiệt độ CPU) rơi vào cả hai bên thay vì dồn về một bên.
    Trước khi đo, mỗi bên chạy một lần khởi động không tính giờ. Mỗi lần đo đồng bộ CUDA trước và sau.
    """
    def timed(run) -> float:
        _sync(device)
        start = time.perf_counter()
        run()
        _sync(device)
        return (time.perf_counter() - start) * MS_PER_SECOND

    run_reference()
    run_candidate()
    reference_times, candidate_times = [], []
    for _ in range(repeats):
        reference_times.append(timed(run_reference))
        candidate_times.append(timed(run_candidate))
    return statistics.median(reference_times), statistics.median(candidate_times)

def _matches(reference: torch.Tensor, candidate: torch.Tensor) -> bool:
    """Khớp theo dung sai mặc định của ``torch.testing.assert_close`` cho dtype của tensor."""
    try:
        torch.testing.assert_close(candidate, reference)
    except AssertionError:
        return False
    return True

def attention_shape(config: SAIConfig) -> dict:
    """Shape attention của model: D = d_model // num_heads, như SAI/modeling.py."""
    if config.d_model % config.num_heads:
        raise ValueError(f"d_model={config.d_model} không chia hết cho num_heads={config.num_heads}")
    return {
        "num_heads": config.num_heads,
        "num_kv_heads": config.num_kv_heads,
        "head_dim": config.d_model // config.num_heads,
    }

def compare_attention_kernels(
    reference: str, candidate: str, *,
    batch_size: int = 2, num_heads: int = 8, num_kv_heads: int = 2, seq_len: int = 128,
    head_dim: int = 64, dtype: torch.dtype = torch.float32, device: str = "cpu",
    causal: bool = True, repeats: int = 5, seed: int = 0,
) -> dict:
    device = torch.device(device)
    generator = torch.Generator(device="cpu").manual_seed(seed)

    def randn(*shape):
        return torch.randn(*shape, generator=generator).to(device=device, dtype=dtype)

    q = randn(batch_size, num_heads, seq_len, head_dim)
    k = randn(batch_size, num_kv_heads, seq_len, head_dim)
    v = randn(batch_size, num_kv_heads, seq_len, head_dim)
    grad_out = randn(batch_size, num_heads, seq_len, head_dim)

    def run(name):
        kernel = ATTENTION_KERNELS.get(name)
        inputs = [t.clone().requires_grad_(True) for t in (q, k, v)]
        out = kernel(*inputs, is_causal=causal)
        out.backward(grad_out)
        return out.detach(), [t.grad for t in inputs]

    ref_out, ref_grads = run(reference)
    cand_out, cand_grads = run(candidate)
    report = {
        "reference": reference,
        "candidate": candidate,
        "dtype": str(dtype).replace("torch.", ""),
        "shape": {"B": batch_size, "H": num_heads, "Hkv": num_kv_heads, "T": seq_len, "D": head_dim},
        "max_abs_diff_out": (ref_out - cand_out).abs().max().item(),
    }
    for name, ref_grad, cand_grad in zip(("q", "k", "v"), ref_grads, cand_grads):
        report[f"max_abs_diff_grad_{name}"] = (ref_grad - cand_grad).abs().max().item()
    report["matches_reference"] = _matches(ref_out, cand_out) and all(
        _matches(ref_grad, cand_grad) for ref_grad, cand_grad in zip(ref_grads, cand_grads)
    )
    report["reference_ms"], report["candidate_ms"] = _interleaved_median_ms(
        device, lambda: run(reference), lambda: run(candidate), repeats,
    )
    report["speedup"] = report["reference_ms"] / report["candidate_ms"]
    report["reference_peak_mib"] = _peak_memory_mib(device, lambda: run(reference))
    report["candidate_peak_mib"] = _peak_memory_mib(device, lambda: run(candidate))
    return report

def compare_objectives(
    reference: str, candidate: str, *,
    reference_chunk_size: int, candidate_chunk_size: int, num_tokens: int, d_model: int, vocab_size: int,
    dtype: torch.dtype = torch.float32, device: str = "cpu", repeats: int = 3, seed: int = 0,
) -> dict:
    """So objective ``candidate`` với ``reference`` theo hợp đồng trong ``objectives/registry.py``.

    Mọi token chịu loss (token_weight = 1): phần mask đã có test riêng của từng objective. Objective trả về tổng
    trên N token nên sai khác loss tính tương đối theo mốc; gradient so theo hidden và head weight.
    """
    device = torch.device(device)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    hidden = torch.randn(num_tokens, d_model, generator=generator).to(device=device, dtype=dtype)   # (N, d_model)
    # Chia sqrt(d_model) để logits có phương sai cỡ 1; không chia thì logits cỡ sqrt(d_model), softmax bão hoà.
    weight = (torch.randn(vocab_size, d_model, generator=generator) / math.sqrt(d_model)).to(device=device, dtype=dtype)
    targets = torch.randint(0, vocab_size, (num_tokens,), generator=generator).to(device)        # (N,) long
    token_weight = torch.ones(num_tokens, device=device)                                          # (N,) float32

    def run(name, chunk_size):
        objective = OBJECTIVES.get(name)
        inputs = [t.clone().requires_grad_(True) for t in (hidden, weight)]
        loss = objective(*inputs, targets, token_weight, chunk_size=chunk_size)
        loss.backward()
        return loss.detach(), [t.grad for t in inputs]

    ref_loss, ref_grads = run(reference, reference_chunk_size)
    cand_loss, cand_grads = run(candidate, candidate_chunk_size)
    report = {
        "reference": reference,
        "candidate": candidate,
        "reference_chunk_size": reference_chunk_size,
        "candidate_chunk_size": candidate_chunk_size,
        "dtype": str(dtype).replace("torch.", ""),
        "shape": {"N": num_tokens, "d_model": d_model, "V": vocab_size},
        "reference_loss": ref_loss.item(),
        "candidate_loss": cand_loss.item(),
        "loss_rel_diff": abs(cand_loss.item() - ref_loss.item()) / abs(ref_loss.item()),
    }
    for name, ref_grad, cand_grad in zip(("hidden", "weight"), ref_grads, cand_grads):
        report[f"max_abs_diff_grad_{name}"] = (ref_grad - cand_grad).abs().max().item()
    report["matches_reference"] = _matches(ref_loss, cand_loss) and all(
        _matches(ref_grad, cand_grad) for ref_grad, cand_grad in zip(ref_grads, cand_grads)
    )
    report["reference_ms"], report["candidate_ms"] = _interleaved_median_ms(
        device, lambda: run(reference, reference_chunk_size), lambda: run(candidate, candidate_chunk_size), repeats,
    )
    report["speedup"] = report["reference_ms"] / report["candidate_ms"]
    report["reference_peak_mib"] = _peak_memory_mib(device, lambda: run(reference, reference_chunk_size))
    report["candidate_peak_mib"] = _peak_memory_mib(device, lambda: run(candidate, candidate_chunk_size))
    return report

def _run_comparisons(
    command: str, items: dict, measure: Callable[[str, object], dict], label_key: str, metric_specs: dict,
    run_dir, events: EventLog,
) -> RunResult:
    """Đo lần lượt từng mục (model, experiment) bằng ``measure(nhãn, mục) -> report``; mục nào xong thì ghi ngay
    một bản ghi ``{label_key: nhãn, **report}`` vào events.jsonl. Mỗi mục một nhóm số đo theo ``metric_specs``
    (khoá report → (đơn vị, chiều tốt hơn)); số đo None (bộ nhớ đỉnh ngoài CUDA) không thành Metric.
    """
    started = time.perf_counter()
    events.emit("started", total=len(items))
    records, metrics = [], []
    for label, item in items.items():
        report = measure(label, item)
        record = {label_key: label, **report}
        records.append(record)
        metrics += [
            Metric(label, key, float(report[key]), unit, better)
            for key, (unit, better) in metric_specs.items() if report[key] is not None
        ]
        events.emit("record", record=record)
        events.emit("progress", done=len(records), total=len(items))

    result = RunResult(
        command=command,
        metrics=metrics,
        records=records,
        timings={"total_ms": (time.perf_counter() - started) * MS_PER_SECOND},
    )
    write_result(run_dir.result_path, result)
    events.emit("finished")
    return result

@dataclass
class KernelCompareSpec:
    candidate: str                  # kernel ứng viên, tên trong ATTENTION_KERNELS
    models: dict[str, dict]         # nhãn → config model đã resolve; kernel mốc là attention_kernel của config
    batch_size: int = 2
    seq_len: int = 128
    dtype: Literal["float32", "float16", "bfloat16"] = "float32"
    device: Literal["auto", "cuda", "mps", "cpu"] = "cpu"
    causal: bool = True
    repeats: int = 5

    def __post_init__(self):
        if self.candidate not in ATTENTION_KERNELS:
            raise ConfigError(f"candidate: kernel '{self.candidate}' không tồn tại; các lựa chọn: {ATTENTION_KERNELS.names()}")
        if not self.models:
            raise ConfigError("models: cần ít nhất một model")
        for name in ("batch_size", "seq_len", "repeats"):
            if getattr(self, name) < 1:
                raise ConfigError(f"{name}: cần số nguyên dương, nhận {getattr(self, name)}")

def resolve_kernel_compare(data: dict, overrides=None) -> dict:
    """Áp ``--set`` rồi thay ``models`` (danh sách tên, mặc định ARCH_CHECK_MODELS) bằng config của từng model.

    ``models`` cũng nhận mapping nhãn → tên hoặc config viết sẵn (vd. một shape tự chọn, hoặc spec đã resolve).
    Kết quả tự đủ: chạy lại ``resolved_config.yaml`` dùng đúng shape và kernel mốc của lúc resolve.
    """
    resolved = apply_overrides(data, overrides)
    models = resolved.get("models", list(ARCH_CHECK_MODELS))
    if isinstance(models, list):
        if not all(isinstance(name, str) for name in models) or len(set(models)) != len(models):
            raise ConfigError(f"models: cần danh sách tên model không trùng, nhận {models!r}")
        # Tên vừa là nhãn trong kết quả vừa là tham chiếu tới file config.
        models = dict(zip(models, models))
    if not isinstance(models, dict):
        raise ConfigError(f"models: cần danh sách tên model hoặc mapping nhãn → tên/config, nhận {models!r}")
    resolved["models"] = {
        label: resolve_config_ref(ref, MODEL_CONFIG_DIRS, f"models.{label}") for label, ref in models.items()
    }
    return resolved

def _kernel_model_config(label: str, model: dict):
    """Config đã kiểm tra của một model cần so; họ mô hình phải chọn kernel qua config."""
    _, config = resolve_model_config(model)
    if not hasattr(config, "attention_kernel"):
        raise ConfigError(f"models.{label}: họ '{model['family']}' không chọn attention kernel qua config; dùng bản SAI_modular")
    if config.attention_kernel not in ATTENTION_KERNELS:
        raise ConfigError(
            f"models.{label}.attention_kernel: kernel '{config.attention_kernel}' không tồn tại; "
            f"các lựa chọn: {ATTENTION_KERNELS.names()}"
        )
    return config

def validate_kernel_compare(resolved: dict) -> KernelCompareSpec:
    """Kiểm tra schema, kernel ứng viên và từng model (họ, kernel mốc, shape) trước khi tạo tensor nào."""
    spec = from_dict(KernelCompareSpec, resolved, KERNEL_COMMAND)
    for label, model in spec.models.items():
        try:
            attention_shape(_kernel_model_config(label, model))
        except ValueError as error:
            raise ConfigError(f"models.{label}: {error}") from error
    return spec

def run_kernel_compare(spec: KernelCompareSpec, run_dir, events: EventLog) -> RunResult:
    """So lần lượt từng model trên shape attention của model đó, kernel mốc là attention_kernel trong config."""
    device = str(select_device(spec.device))

    def measure(label: str, model: dict) -> dict:
        config = _kernel_model_config(label, model)
        return compare_attention_kernels(
            config.attention_kernel, spec.candidate,
            batch_size=spec.batch_size, seq_len=spec.seq_len, **attention_shape(config),
            dtype=getattr(torch, spec.dtype), device=device, causal=spec.causal, repeats=spec.repeats,
        )

    return _run_comparisons(KERNEL_COMMAND, spec.models, measure, "model", KERNEL_METRICS, run_dir, events)

def kernel_compare_inputs(_spec: KernelCompareSpec) -> dict:
    """Không đọc file nào: input là tensor ngẫu nhiên, config model đã nhúng trong spec."""
    return {}

TASKS.register(KERNEL_COMMAND)(Task(
    department="arch",
    spec_cls=KernelCompareSpec,
    resolve=resolve_kernel_compare,
    validate=validate_kernel_compare,
    run=run_kernel_compare,
    inputs=kernel_compare_inputs,
    help="so một attention kernel với kernel mặc định của từng model: output, gradient, tốc độ, bộ nhớ",
))

@dataclass
class ObjectiveSetup:
    """Mốc của một experiment, lấy lúc resolve từ params của stage và config model."""
    objective: str                  # params.objective của stage
    loss_chunk_size: int            # params.loss_chunk_size
    max_microbatch_tokens: int      # params.max_microbatch_tokens: số token N mặc định của phép đo
    d_model: int
    vocab_size: int

@dataclass
class ObjectiveCompareSpec:
    candidate: str                              # objective ứng viên, tên trong OBJECTIVES
    experiments: dict[str, ObjectiveSetup]      # nhãn → mốc của experiment đó
    candidate_chunk_size: int | None = None     # None: như loss_chunk_size của mốc
    stage: str = "pretrain"                     # stage lấy mốc trong từng experiment
    tokens: int | None = None                   # N; None: max_microbatch_tokens của stage
    dtype: Literal["float32", "float16", "bfloat16"] = "float32"
    device: Literal["auto", "cuda", "mps", "cpu"] = "cpu"
    repeats: int = 3

    def __post_init__(self):
        if self.candidate not in OBJECTIVES:
            raise ConfigError(f"candidate: objective '{self.candidate}' không tồn tại; các lựa chọn: {OBJECTIVES.names()}")
        if not self.experiments:
            raise ConfigError("experiments: cần ít nhất một experiment")
        for label, setup in self.experiments.items():
            if setup.objective not in OBJECTIVES:
                raise ConfigError(
                    f"experiments.{label}.objective: objective '{setup.objective}' không tồn tại; "
                    f"các lựa chọn: {OBJECTIVES.names()}"
                )
            for name in ("loss_chunk_size", "max_microbatch_tokens", "d_model", "vocab_size"):
                if getattr(setup, name) < 1:
                    raise ConfigError(f"experiments.{label}.{name}: cần số nguyên dương, nhận {getattr(setup, name)}")
        for name in ("candidate_chunk_size", "tokens", "repeats"):
            value = getattr(self, name)
            if value is not None and value < 1:
                raise ConfigError(f"{name}: cần số nguyên dương, nhận {value}")

def _objective_setup(label: str, name: str, stage: str) -> dict:
    """Mốc lấy từ experiment ``name`` trong core/configs/: params của ``stage`` đã gộp mặc định, cỡ head của model."""
    # Import muộn: recipes → eval.lm_loss → eval/__init__ → module này, import ở đầu file sẽ thành vòng.
    from foundation_lab.recipes.base import RECIPES, LMTrainParams

    where = f"experiments.{label}"
    experiment = resolve_experiment_dict(resolve_config_ref(name, (EXPERIMENT_CONFIG_DIR,), where))
    stages = experiment["pipeline"]
    if stage not in stages:
        raise ConfigError(f"{where}: experiment '{name}' không có stage '{stage}'; các stage: {list(stages)}")
    stage_spec = from_dict(StageSpec, stages[stage], f"{where}.pipeline.{stage}")
    if stage_spec.recipe not in RECIPES:
        raise ConfigError(f"{where}.pipeline.{stage}.recipe: recipe '{stage_spec.recipe}' không tồn tại")
    # Dựng params bằng dataclass của recipe để khoá stage không ghi lấy đúng mặc định trong code.
    params = from_dict(RECIPES.get(stage_spec.recipe).config_cls, stage_spec.params, f"{where}.pipeline.{stage}.params")
    if not isinstance(params, LMTrainParams):
        raise ConfigError(f"{where}: recipe '{stage_spec.recipe}' của stage '{stage}' không chọn objective qua params")
    _, model_config = resolve_model_config(experiment["model"])
    return {
        "objective": params.objective,
        "loss_chunk_size": params.loss_chunk_size,
        "max_microbatch_tokens": params.max_microbatch_tokens,
        "d_model": model_config.d_model,
        "vocab_size": model_config.vocab_size,
    }

def resolve_objective_compare(data: dict, overrides=None) -> dict:
    """Áp ``--set`` rồi thay ``experiments`` (danh sách tên, mặc định OBJECTIVE_CHECK_EXPERIMENTS) bằng mốc của từng
    experiment tại ``stage``.

    ``experiments`` cũng nhận mapping nhãn → tên hoặc mốc viết sẵn (vd. spec đã resolve). Kết quả tự đủ: chạy lại
    ``resolved_config.yaml`` dùng đúng mốc của lúc resolve, kể cả khi config experiment đã đổi sau đó.
    """
    resolved = apply_overrides(data, overrides)
    experiments = resolved.get("experiments", list(OBJECTIVE_CHECK_EXPERIMENTS))
    if isinstance(experiments, list):
        if not all(isinstance(name, str) for name in experiments) or len(set(experiments)) != len(experiments):
            raise ConfigError(f"experiments: cần danh sách tên experiment không trùng, nhận {experiments!r}")
        # Tên vừa là nhãn trong kết quả vừa là tham chiếu tới file experiment.
        experiments = dict(zip(experiments, experiments))
    if not isinstance(experiments, dict):
        raise ConfigError(f"experiments: cần danh sách tên experiment hoặc mapping nhãn → tên/mốc, nhận {experiments!r}")
    stage = resolved.get("stage", ObjectiveCompareSpec.stage)
    resolved["experiments"] = {
        label: _objective_setup(label, ref, stage) if isinstance(ref, str) else ref for label, ref in experiments.items()
    }
    return resolved

def validate_objective_compare(resolved: dict) -> ObjectiveCompareSpec:
    """Kiểm tra schema, objective ứng viên và mốc của từng experiment trước khi tạo tensor nào."""
    return from_dict(ObjectiveCompareSpec, resolved, OBJECTIVE_COMMAND)

def run_objective_compare(spec: ObjectiveCompareSpec, run_dir, events: EventLog) -> RunResult:
    """So lần lượt từng experiment trên cỡ head của model đó, objective mốc là objective của stage."""
    device = str(select_device(spec.device))

    def measure(_label: str, setup: ObjectiveSetup) -> dict:
        report = compare_objectives(
            setup.objective, spec.candidate,
            reference_chunk_size=setup.loss_chunk_size,
            candidate_chunk_size=setup.loss_chunk_size if spec.candidate_chunk_size is None else spec.candidate_chunk_size,
            num_tokens=setup.max_microbatch_tokens if spec.tokens is None else spec.tokens,
            d_model=setup.d_model, vocab_size=setup.vocab_size,
            dtype=getattr(torch, spec.dtype), device=device, repeats=spec.repeats,
        )
        return {"stage": spec.stage, **report}

    return _run_comparisons(OBJECTIVE_COMMAND, spec.experiments, measure, "experiment", OBJECTIVE_METRICS, run_dir, events)

def objective_compare_inputs(_spec: ObjectiveCompareSpec) -> dict:
    """Không đọc file nào: input là tensor ngẫu nhiên, mốc của từng experiment đã nhúng trong spec."""
    return {}

TASKS.register(OBJECTIVE_COMMAND)(Task(
    department="objectives",
    spec_cls=ObjectiveCompareSpec,
    resolve=resolve_objective_compare,
    validate=validate_objective_compare,
    run=run_objective_compare,
    inputs=objective_compare_inputs,
    help="so một objective với objective của stage trong từng experiment: loss, gradient, tốc độ, bộ nhớ",
))
