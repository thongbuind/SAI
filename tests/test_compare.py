import json
import pytest
import torch
import yaml
from helpers import TINY_MODEL
from foundation_lab.architecture.models.modular import SAIModularConfig
from foundation_lab.catalog import RECIPES, resolve_model_config
from foundation_lab.cli import main
from foundation_lab.core.config import from_dict
from foundation_lab.core.experiment import resolve_config_ref, resolve_experiment
from foundation_lab.eval.compare import (
    ARCH_CHECK_MODELS, OBJECTIVE_CHECK_EXPERIMENTS, _interleaved_median_ms, compare_attention_kernels, compare_objectives,
    resolve_kernel_compare, resolve_objective_compare,
)
from foundation_lab.paths import EXPERIMENT_CONFIG_DIR, MODEL_CONFIG_DIRS
from foundation_lab.state.events import EventLog
from foundation_lab.state.results import read_result

def run_task(tmp_path, command, *settings, spec=None, run_id="r0"):
    """Chạy tác vụ ``command`` với ``--set`` (và spec nếu có); trả về thư mục run."""
    args = [command] + ([str(spec)] if spec else [])
    for setting in settings:
        args += ["--set", setting]
    main([*args, "--runs-root", str(tmp_path / "runs"), "--run-id", run_id])
    return tmp_path / "runs" / command / run_id

def run_compare(tmp_path, *settings, **kwargs):
    """``compare-kernel`` trên shape nhỏ."""
    return run_task(tmp_path, "compare-kernel", "seq_len=8", "repeats=1", *settings, **kwargs)

def run_objective(tmp_path, *settings, **kwargs):
    """``compare-objective`` trên ít token."""
    return run_task(tmp_path, "compare-objective", "tokens=8", "repeats=1", *settings, **kwargs)

def test_timing_warms_up_then_interleaves_reference_and_candidate():
    calls = []
    _interleaved_median_ms(torch.device("cpu"), lambda: calls.append("mốc"), lambda: calls.append("ứng viên"), repeats=3)
    # Một lần khởi động mỗi bên, rồi mỗi lượt đo mốc trước, ứng viên sau.
    assert calls == ["mốc", "ứng viên"] * (1 + 3)

def test_compare_sdpa_with_eager_reference():
    report = compare_attention_kernels("sdpa", "eager", seq_len=16, head_dim=8, repeats=1)
    for key in ("out", "grad_q", "grad_k", "grad_v"):
        assert report[f"max_abs_diff_{key}"] < 1e-5, (key, report)
    assert report["matches_reference"]
    assert report["reference_ms"] > 0 and report["candidate_ms"] > 0
    assert report["speedup"] == report["reference_ms"] / report["candidate_ms"]
    assert report["reference_peak_mib"] is None and report["candidate_peak_mib"] is None  # CPU: không có bộ đếm đỉnh

def test_compare_kernel_task_uses_model_shape_and_default_kernel(tmp_path):
    run_path = run_compare(tmp_path, "candidate=eager", "models=[SAI_modular_35M]")
    result = read_result(run_path / "result.json")
    [record] = result["records"]
    # Mốc là attention_kernel trong config hiện tại của model, không phải tên gõ tay.
    default_kernel = resolve_config_ref("SAI_modular_35M", MODEL_CONFIG_DIRS, "test")["attention_kernel"]
    assert (record["model"], record["reference"], record["candidate"]) == ("SAI_modular_35M", default_kernel, "eager")
    assert record["shape"] == {"B": 2, "H": 8, "Hkv": 8, "T": 8, "D": 64}
    assert record["matches_reference"]
    assert {metric["group"] for metric in result["metrics"]} == {"SAI_modular_35M"}
    # Mốc nằm trong từng bản ghi (kernel của từng model), không phải một group.
    assert result["command"] == "compare-kernel" and result["baseline"] is None

    events = EventLog.read(run_path / "events.jsonl")
    assert [event["kind"] for event in events] == ["started", "record", "progress", "finished"]
    assert events[1]["record"] == record

def test_compare_kernel_defaults_to_every_model_size():
    resolved = resolve_kernel_compare({"candidate": "eager"})
    assert list(resolved["models"]) == list(ARCH_CHECK_MODELS)
    assert all(model["family"] == "SAI_modular" for model in resolved["models"].values())

def test_describe_lists_models_by_size(capsys):
    main(["describe"])
    description = json.loads(capsys.readouterr().out)
    modular = [name for name in description["models"] if description["model_families"][name] == "SAI_modular"]
    # Web hiện model theo thứ tự của describe: spec mặc định của web phải trùng mặc định của CLI.
    assert modular == list(ARCH_CHECK_MODELS)

def test_resolved_config_reruns_with_the_embedded_model(tmp_path):
    """resolved_config.yaml tự đủ: chạy lại nó dùng đúng config model đã nhúng, kể cả model viết thẳng trong spec."""
    spec = tmp_path / "spec.yaml"
    mqa = {**TINY_MODEL, "family": "SAI_modular", "num_kv_heads": 1}   # không ghi attention_kernel: dùng mặc định
    spec.write_text(yaml.safe_dump({"candidate": "eager", "models": {"mqa": mqa}}), encoding="utf-8")
    first = run_compare(tmp_path, spec=spec, run_id="a")
    again = run_compare(tmp_path, spec=first / "resolved_config.yaml", run_id="b")

    [first_record] = read_result(first / "result.json")["records"]
    [again_record] = read_result(again / "result.json")["records"]
    assert first_record["reference"] == again_record["reference"] == SAIModularConfig.attention_kernel
    assert first_record["shape"] == again_record["shape"] == {"B": 2, "H": 4, "Hkv": 1, "T": 8, "D": 8}

@pytest.mark.parametrize("settings, message", [
    (["candidate=khong_co"], "candidate"),
    (["candidate=eager", "models=[SAI_35M]"], "không chọn attention kernel"),
    (["candidate=eager", "models=[SAI_modular_35M, SAI_modular_35M]"], "không trùng"),
    (["candidate=eager", "models=[]"], "models"),
    (["candidate=eager", "repeats=0"], "repeats"),
    ([], "candidate"),
])
def test_invalid_spec_exits_2_without_creating_run(tmp_path, capsys, settings, message):
    with pytest.raises(SystemExit) as error:
        run_compare(tmp_path, *settings)
    assert error.value.code == 2
    assert message in capsys.readouterr().err
    assert not (tmp_path / "runs").exists()

def test_compare_objective_chunk_size_keeps_loss_and_gradients():
    report = compare_objectives(
        "lm_ce", "lm_ce", reference_chunk_size=8, candidate_chunk_size=3,
        num_tokens=20, d_model=8, vocab_size=16, repeats=1,
    )
    assert report["loss_rel_diff"] < 1e-6, report
    assert report["max_abs_diff_grad_hidden"] < 1e-6 and report["max_abs_diff_grad_weight"] < 1e-6, report
    assert report["matches_reference"]
    assert report["shape"] == {"N": 20, "d_model": 8, "V": 16}
    assert report["speedup"] == report["reference_ms"] / report["candidate_ms"]
    assert report["reference_peak_mib"] is None and report["candidate_peak_mib"] is None  # CPU: không có bộ đếm đỉnh

@pytest.mark.parametrize("stage", ["pretrain", "sft1"])
def test_objective_baseline_comes_from_current_stage_params(stage):
    resolved = resolve_objective_compare({"candidate": "lm_ce", "stage": stage})
    assert list(resolved["experiments"]) == list(OBJECTIVE_CHECK_EXPERIMENTS)
    for name, setup in resolved["experiments"].items():
        # Mốc là params của stage dựng bằng dataclass của recipe (khoá stage không ghi lấy mặc định trong code).
        experiment = resolve_experiment(EXPERIMENT_CONFIG_DIR / f"{name}.yaml")
        stage_config = experiment["pipeline"][stage]
        params = from_dict(RECIPES.get(stage_config["recipe"]).config_cls, stage_config["params"])
        _, model_config = resolve_model_config(experiment["model"])
        assert setup == {
            "objective": params.objective,
            "loss_chunk_size": params.loss_chunk_size,
            "max_microbatch_tokens": params.max_microbatch_tokens,
            "d_model": model_config.d_model,
            "vocab_size": model_config.vocab_size,
        }

def test_describe_lists_experiments_by_size(capsys):
    main(["describe"])
    experiments = json.loads(capsys.readouterr().out)["experiments"]
    # Web hiện experiment theo thứ tự này: spec mặc định của web phải trùng mặc định của CLI.
    assert list(experiments) == list(OBJECTIVE_CHECK_EXPERIMENTS)
    assert all("pretrain" in stages and "continued" not in stages for stages in experiments.values())  # continued đang tắt

def test_compare_objective_task_records_baseline_and_candidate(tmp_path):
    run_path = run_objective(tmp_path, "candidate=lm_ce", "candidate_chunk_size=3", "experiments=[SAI_35M]")
    result = read_result(run_path / "result.json")
    [record] = result["records"]
    baseline = resolve_objective_compare({"experiments": ["SAI_35M"]})["experiments"]["SAI_35M"]
    assert (record["experiment"], record["stage"]) == ("SAI_35M", "pretrain")
    assert (record["reference"], record["reference_chunk_size"]) == (baseline["objective"], baseline["loss_chunk_size"])
    assert (record["candidate"], record["candidate_chunk_size"]) == ("lm_ce", 3)
    assert record["shape"] == {"N": 8, "d_model": baseline["d_model"], "V": baseline["vocab_size"]}
    assert record["matches_reference"]
    assert {metric["group"] for metric in result["metrics"]} == {"SAI_35M"}
    assert result["command"] == "compare-objective" and result["baseline"] is None

    events = EventLog.read(run_path / "events.jsonl")
    assert [event["kind"] for event in events] == ["started", "record", "progress", "finished"]
    assert events[1]["record"] == record

def test_resolved_objective_config_reruns_with_the_embedded_baseline(tmp_path):
    """resolved_config.yaml tự đủ: chạy lại dùng đúng mốc đã nhúng, kể cả mốc viết thẳng trong spec."""
    spec = tmp_path / "spec.yaml"
    setup = {"objective": "lm_ce", "loss_chunk_size": 5, "max_microbatch_tokens": 12, "d_model": 8, "vocab_size": 16}
    spec.write_text(yaml.safe_dump({"candidate": "lm_ce", "experiments": {"tiny": setup}}), encoding="utf-8")
    first = run_task(tmp_path, "compare-objective", "repeats=1", spec=spec, run_id="a")
    again = run_task(tmp_path, "compare-objective", spec=first / "resolved_config.yaml", run_id="b")

    [first_record] = read_result(first / "result.json")["records"]
    [again_record] = read_result(again / "result.json")["records"]
    # tokens bỏ trống: N là max_microbatch_tokens của mốc; chunk ứng viên bỏ trống: như chunk của mốc.
    assert first_record["shape"] == again_record["shape"] == {"N": 12, "d_model": 8, "V": 16}
    assert first_record["candidate_chunk_size"] == again_record["candidate_chunk_size"] == 5

@pytest.mark.parametrize("settings, message", [
    (["candidate=khong_co"], "candidate"),
    (["candidate=lm_ce", "experiments=[khong_co]"], "không tìm thấy"),
    (["candidate=lm_ce", "experiments=[SAI_35M, SAI_35M]"], "không trùng"),
    (["candidate=lm_ce", "experiments=[]"], "experiments"),
    (["candidate=lm_ce", "stage=khong_co"], "không có stage"),
    (["candidate=lm_ce", "tokens=0"], "tokens"),
    (["candidate=lm_ce", "candidate_chunk_size=0"], "candidate_chunk_size"),
    ([], "candidate"),
])
def test_invalid_objective_spec_exits_2_without_creating_run(tmp_path, capsys, settings, message):
    with pytest.raises(SystemExit) as error:
        run_objective(tmp_path, *settings)
    assert error.value.code == 2
    assert message in capsys.readouterr().err
    assert not (tmp_path / "runs").exists()

def test_list_command_shows_every_department(capsys):
    main(["list"])
    out = capsys.readouterr().out
    for department in ("[arch]", "[data]", "[objectives]", "[engine]", "[recipes]"):
        assert department in out
    assert "sdpa" in out and "causal_lm" in out and "sft" in out
