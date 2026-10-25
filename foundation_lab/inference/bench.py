"""So sánh các biến thể giải mã trên cùng một checkpoint và cùng bộ câu hỏi.

Spec (YAML/JSON)::

    model: SAI_35M                      # tên trong config/models/, architecture/configs/ hoặc mapping có khoá family
    checkpoint: model/sft1_35M.pt       # tương đối theo thư mục gốc dự án
    device: auto
    prompts: ["xin chào", "thủ đô của việt nam là gì?"]
    variants:
      - name: default                   # bắt buộc: decoder "sai" + GenerationConfig mặc định
      - name: sample
        generation: {strategy: sample, temperature: 0.8, seed: 0}

Spec phải có một biến thể dùng đúng cách sinh của bản phát hành (decoder ``sai``, ``generation`` để mặc định)
để mọi biến thể thử nghiệm luôn được so với hành vi người dùng đang nhận.

Mỗi câu hỏi chạy lần lượt trên mọi biến thể rồi mới sang câu tiếp theo, nên kết quả từng câu so sánh được
ngay khi đang chạy. Kết quả ghi vào thư mục run: ``events.jsonl`` (dần dần) và ``result.json`` (khi xong).
``infer`` là một tác vụ theo hợp đồng chung trong ``foundation_lab.core.tasks``.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
from SAI.generate import GenerationConfig, validate_generation_config
from foundation_lab.architecture import resolve_model_config
from foundation_lab.core.config import ConfigError, apply_overrides, from_dict
from foundation_lab.core.experiment import resolve_config_ref
from foundation_lab.core.tasks import TASKS, Task
from foundation_lab.engine.runtime import select_device
from foundation_lab.eval.text import distinct_n
from foundation_lab.inference.loading import load_for_inference
from foundation_lab.inference.registry import DECODERS
from foundation_lab.paths import MODEL_CONFIG_DIRS, resolve_project_path
from foundation_lab.state.events import EventLog
from foundation_lab.state.results import Metric, RunResult, write_result

COMMAND = "infer"
# Decoder gọi prefill/decode_step có KV cache, nên họ mô hình phải khai báo các capability này.
REQUIRED_CAPABILITIES = frozenset({"causal_lm", "kv_cache"})
# Câu trả lời dừng vì hết max_new_tokens (bị cắt), không phải vì model tự kết thúc.
TRUNCATED_REASON = "length"
DISTINCT_N = 2
MS_PER_SECOND = 1000
# Decoder chính thức (model.generate_response); cùng GenerationConfig() mặc định là cách sinh của bản phát hành.
RELEASE_DECODER = "sai"

@dataclass
class VariantSpec:
    name: str
    decoder: str = RELEASE_DECODER
    generation: GenerationConfig = field(default_factory=GenerationConfig)

    def __post_init__(self):
        if self.decoder not in DECODERS:
            raise ConfigError(f"decoder '{self.decoder}' không tồn tại; các lựa chọn: {DECODERS.names()}")

def is_release_variant(variant: VariantSpec) -> bool:
    """Biến thể sinh đúng như bản phát hành: decoder chính thức với GenerationConfig mặc định."""
    return variant.decoder == RELEASE_DECODER and variant.generation == GenerationConfig()

@dataclass
class InferenceSpec:
    model: dict
    checkpoint: str
    prompts: list[str]
    variants: list[VariantSpec]
    device: Literal["auto", "cuda", "mps", "cpu"] = "auto"

    def __post_init__(self):
        # Báo lỗi lúc đọc spec, trước khi tốn RAM và thời gian nạp model.
        if not self.prompts or any(not prompt.strip() for prompt in self.prompts):
            raise ConfigError("prompts: cần ít nhất một câu hỏi và không có câu rỗng")
        if not self.variants:
            raise ConfigError("variants: cần ít nhất một biến thể")
        names = [variant.name for variant in self.variants]
        if len(set(names)) != len(names):
            raise ConfigError(f"variants: tên biến thể bị trùng {names}")
        if not any(is_release_variant(variant) for variant in self.variants):
            raise ConfigError(
                f"variants: cần một biến thể cấu hình mặc định (decoder '{RELEASE_DECODER}', không ghi generation), "
                "vd. {name: default}"
            )

def resolve_inference(data: dict, overrides=None) -> dict:
    """Thay ``model: <tên>`` bằng nội dung config rồi áp override ``--set``; kết quả tự đủ, ghi được vào run."""
    resolved = dict(data)
    resolved["model"] = resolve_config_ref(data.get("model"), MODEL_CONFIG_DIRS, "model")
    return apply_overrides(resolved, overrides)

def validate_inference(resolved: dict) -> InferenceSpec:
    """Kiểm tra toàn bộ spec: schema, họ mô hình, checkpoint tồn tại, giá trị của từng GenerationConfig."""
    spec = from_dict(InferenceSpec, resolved, "inference")
    family, _ = resolve_model_config(spec.model)
    missing = REQUIRED_CAPABILITIES - family.capabilities
    if missing:
        raise ConfigError(f"model: họ '{spec.model['family']}' thiếu capability {sorted(missing)} để giải mã")
    if not resolve_project_path(spec.checkpoint).is_file():
        raise ConfigError(f"checkpoint: không tìm thấy file {spec.checkpoint}")
    for index, variant in enumerate(spec.variants):
        try:
            validate_generation_config(variant.generation)
        except ValueError as error:
            raise ConfigError(f"variants[{index}].generation: {error}") from error
    return spec

def variant_metrics(variants: list[VariantSpec], records: list[dict]) -> list[Metric]:
    """Số đo của từng biến thể trên mọi câu hỏi."""
    metrics = []
    for variant in variants:
        rows = [record for record in records if record["variant"] == variant.name]
        seconds = sum(row["latency_ms"] for row in rows) / MS_PER_SECOND
        completion_tokens = sum(row["completion_tokens"] for row in rows)
        metrics += [
            Metric(variant.name, "latency_ms_mean", sum(row["latency_ms"] for row in rows) / len(rows), "ms", "lower"),
            Metric(variant.name, "tokens_per_second", completion_tokens / seconds, "token/s", "higher"),
            Metric(variant.name, "completion_tokens_mean", completion_tokens / len(rows), "token"),
            Metric(variant.name, "truncated_rate", sum(row["finish_reason"] == TRUNCATED_REASON for row in rows) / len(rows), "", "lower"),
            Metric(variant.name, f"distinct_{DISTINCT_N}", distinct_n([row["text"] for row in rows], DISTINCT_N), "", "higher"),
        ]
    return metrics

def run_inference(spec: InferenceSpec, run_dir, events: EventLog) -> RunResult:
    total = len(spec.prompts) * len(spec.variants)
    events.emit("started", total=total)
    started = time.perf_counter()
    model, tokenizer = load_for_inference(spec.model, resolve_project_path(spec.checkpoint), select_device(spec.device))
    load_ms = (time.perf_counter() - started) * MS_PER_SECOND
    events.emit("progress", done=0, total=total, message="đã nạp model")

    records = []
    for prompt_index, prompt in enumerate(spec.prompts):
        for variant in spec.variants:
            decode = DECODERS.get(variant.decoder)
            decode_started = time.perf_counter()
            output = decode(model, tokenizer, prompt, variant.generation)
            record = {
                "prompt_index": prompt_index,
                "variant": variant.name,
                "text": output.text,
                "finish_reason": output.finish_reason,
                "prompt_tokens": output.prompt_tokens,
                "completion_tokens": output.completion_tokens,
                "decode_steps": output.decode_steps,
                "latency_ms": (time.perf_counter() - decode_started) * MS_PER_SECOND,
            }
            records.append(record)
            events.emit("record", record=record)
            events.emit("progress", done=len(records), total=total)

    result = RunResult(
        command=COMMAND,
        metrics=variant_metrics(spec.variants, records),
        records=records,
        timings={"load_ms": load_ms, "total_ms": (time.perf_counter() - started) * MS_PER_SECOND},
        baseline=next(variant.name for variant in spec.variants if is_release_variant(variant)),
    )
    write_result(run_dir.result_path, result)
    events.emit("finished")
    return result

def inference_inputs(spec: InferenceSpec) -> dict[str, list[Path]]:
    return {"checkpoint": [resolve_project_path(spec.checkpoint)]}

TASKS.register(COMMAND)(Task(
    department="inference",
    spec_cls=InferenceSpec,
    resolve=resolve_inference,
    validate=validate_inference,
    run=run_inference,
    inputs=inference_inputs,
    help="so sánh các biến thể giải mã trên một checkpoint, ghi vào runs/infer/",
))
