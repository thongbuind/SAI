"""Phòng Suy luận: spec, lệnh ``infer`` (thư mục run, events, result) và ``describe``.

Model tí hon nhưng vocab bằng tokenizer chính thức để giải mã thật bằng ``SAI.generate`` trên CPU.
"""

import dataclasses
import io
import json
import re
import pytest
import torch
import yaml
from helpers import TINY_MODEL
from SAI.generate import GenerationConfig
from foundation_lab.catalog import build_model
from foundation_lab.cli import main
from foundation_lab.core.config import ConfigError, from_dict
from foundation_lab.eval.text import distinct_n
from foundation_lab.inference import InferenceSpec, resolve_inference, validate_inference
from foundation_lab.inference.bench import variant_metrics
from foundation_lab.core.config import load_with_extends
from foundation_lab.paths import INFERENCE_BENCHMARK, LAB_DIR, PROJECT_ROOT
from foundation_lab.state.checkpoint import weights_by_model
from foundation_lab.state.events import EventLog
from foundation_lab.state.results import read_result

OFFICIAL_VOCAB = 10000
MODEL = {**TINY_MODEL, "vocab_size": OFFICIAL_VOCAB}

@pytest.fixture
def checkpoint(tmp_path):
    torch.manual_seed(0)
    path = tmp_path / "tiny.pt"
    torch.save(build_model(MODEL).state_dict(), path)
    return path

def make_spec(weights, **changes) -> dict:
    spec = {
        "model": MODEL,
        "checkpoint": str(weights),
        "device": "cpu",
        "prompts": ["xin chào", "thủ đô của việt nam là gì?"],
        "variants": [
            {"name": "greedy", "generation": {"strategy": "greedy", "max_new_tokens": 4}},
            {"name": "beam2", "generation": {"strategy": "beam", "beam_size": 2, "max_new_tokens": 4}},
        ],
    }
    return {**spec, **changes}

def run_cli(tmp_path, spec: dict, run_id="r0"):
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(yaml.safe_dump(spec, allow_unicode=True), encoding="utf-8")
    main(["infer", str(spec_path), "--runs-root", str(tmp_path / "runs"), "--run-id", run_id])
    return tmp_path / "runs" / "infer" / run_id

@pytest.mark.parametrize("changes, message", [
    ({"prompts": []}, "prompts"),
    ({"prompts": ["  "]}, "prompts"),
    ({"variants": []}, "variants"),
    ({"variants": [{"name": "a"}, {"name": "a"}]}, "trùng"),
    ({"variants": [{"name": "a", "decoder": "khong_co"}]}, "decoder"),
    ({"variants": [{"name": "a", "generation": {"top_pp": 0.9}}]}, "variants[0].generation"),
    ({"variants": [{"name": "a", "generation": {"top_p": 2.0}}]}, "variants[0].generation: top_p"),
    ({"checkpoint": "model/khong_co.pt"}, "checkpoint"),
    ({"device": "tpu"}, "device"),
])
def test_invalid_spec_is_rejected_before_loading(checkpoint, changes, message):
    with pytest.raises(ConfigError, match=re.escape(message)):
        validate_inference(resolve_inference(make_spec(checkpoint, **changes)))

def test_model_name_resolves_to_config_file(checkpoint):
    resolved = resolve_inference(make_spec(checkpoint, model="SAI_35M"))
    assert resolved["model"]["family"] == "SAI" and resolved["model"]["d_model"] > 0

def test_infer_writes_run_dir_with_events_and_result(tmp_path, checkpoint):
    run_path = run_cli(tmp_path, make_spec(checkpoint))

    result = read_result(run_path / "result.json")
    assert result["command"] == "infer" and result["format_version"] == 1
    # Mỗi câu hỏi chạy trên mọi biến thể rồi mới sang câu sau.
    assert [(r["prompt_index"], r["variant"]) for r in result["records"]] == [
        (0, "greedy"), (0, "beam2"), (1, "greedy"), (1, "beam2"),
    ]
    assert all(0 < r["completion_tokens"] <= 4 and r["latency_ms"] > 0 for r in result["records"])
    assert {m["group"] for m in result["metrics"]} == {"greedy", "beam2"}
    assert set(result["timings"]) == {"load_ms", "total_ms"}

    events = EventLog.read(run_path / "events.jsonl")
    assert [e["seq"] for e in events] == list(range(1, len(events) + 1))
    assert events[0] == {**events[0], "kind": "started", "total": 4}
    assert events[-1]["kind"] == "finished"
    assert [e["record"] for e in events if e["kind"] == "record"] == result["records"]
    assert [e["done"] for e in events if e["kind"] == "progress"] == [0, 1, 2, 3, 4]

    assert yaml.safe_load((run_path / "resolved_config.yaml").read_text(encoding="utf-8"))["prompts"] == make_spec(checkpoint)["prompts"]
    assert json.loads((run_path / "lineage.json").read_text(encoding="utf-8"))["data"]["checkpoint"][0]["exists"]

def test_greedy_is_deterministic_across_runs(tmp_path, checkpoint):
    first = read_result(run_cli(tmp_path, make_spec(checkpoint), "a") / "result.json")
    second = read_result(run_cli(tmp_path, make_spec(checkpoint), "b") / "result.json")
    assert [r["text"] for r in first["records"]] == [r["text"] for r in second["records"]]

def test_infer_reads_spec_from_stdin(tmp_path, checkpoint, monkeypatch):
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(make_spec(checkpoint), ensure_ascii=False)))
    main(["infer", "-", "--runs-root", str(tmp_path / "runs"), "--run-id", "stdin"])
    assert (tmp_path / "runs" / "infer" / "stdin" / "result.json").exists()

def test_config_error_exits_2_without_creating_run(tmp_path, checkpoint):
    with pytest.raises(SystemExit) as exit_info:
        run_cli(tmp_path, make_spec(checkpoint, prompts=[]))
    assert exit_info.value.code == 2
    assert not (tmp_path / "runs").exists()

def test_runtime_failure_is_recorded_in_events(tmp_path):
    broken = tmp_path / "broken.pt"
    broken.write_bytes(b"khong phai checkpoint")
    with pytest.raises(Exception):
        run_cli(tmp_path, make_spec(broken))
    events = EventLog.read(tmp_path / "runs" / "infer" / "r0" / "events.jsonl")
    assert [e["kind"] for e in events] == ["started", "failed"]
    assert events[-1]["error"]
    assert not (tmp_path / "runs" / "infer" / "r0" / "result.json").exists()

def test_variant_metrics():
    variants = from_dict(InferenceSpec, {
        "model": MODEL, "checkpoint": "x.pt", "prompts": ["a"], "variants": [{"name": "v"}],
    }).variants
    records = [
        {"variant": "v", "text": "a b a b", "finish_reason": "length", "completion_tokens": 4, "latency_ms": 1000},
        {"variant": "v", "text": "c d", "finish_reason": "stop", "completion_tokens": 2, "latency_ms": 500},
    ]
    metrics = {m.name: m for m in variant_metrics(variants, records)}
    assert metrics["latency_ms_mean"].value == 750 and metrics["latency_ms_mean"].better == "lower"
    assert metrics["tokens_per_second"].value == pytest.approx(4.0)
    assert metrics["completion_tokens_mean"].value == 3
    assert metrics["truncated_rate"].value == 0.5
    # bigram: (a b) (b a) (a b) (c d) → 3 khác nhau / 4
    assert metrics["distinct_2"].value == pytest.approx(0.75)

@pytest.mark.parametrize("texts, n, expected", [
    (["a b c"], 1, 1.0),
    (["a a a a"], 1, 0.25),
    (["a"], 2, 0.0),
])
def test_distinct_n(texts, n, expected):
    assert distinct_n(texts, n) == pytest.approx(expected)

def test_describe_lists_infer_schema(capsys):
    main(["describe"])
    description = json.loads(capsys.readouterr().out)
    assert description["format_version"] == 1
    assert {"kind": "decoder", "names": ["sai"]} in description["departments"]["inference"]
    spec = description["commands"]["infer"]["spec"]
    assert spec["name"] == "InferenceSpec"
    assert {name for name, field in spec["fields"].items() if field.get("required")} == {"model", "checkpoint", "prompts", "variants"}
    assert spec["fields"]["device"] == {"type": "choice", "values": ["auto", "cuda", "mps", "cpu"], "default": "auto"}

    variant = spec["fields"]["variants"]["items"]
    generation = variant["fields"]["generation"]["fields"]
    assert {name: field["default"] for name, field in generation.items()} == dataclasses.asdict(GenerationConfig())
    assert generation["stable_steps"]["nullable"] and generation["stable_steps"]["type"] == "int"
    assert "SAI_35M" in description["models"]
    assert set(description["checkpoints"]) <= set(description["models"])
    assert PROJECT_ROOT / description["benchmark_path"] == INFERENCE_BENCHMARK

def test_benchmark_is_shared_by_example_spec():
    """benchmark.yaml là spec một phần (chỉ có prompts) mà spec khác "extends"; web ghi đè nó bằng YAML cùng dạng."""
    benchmark = yaml.safe_load(INFERENCE_BENCHMARK.read_text(encoding="utf-8"))
    assert list(benchmark) == ["prompts"] and benchmark["prompts"] and all(isinstance(p, str) for p in benchmark["prompts"])
    example = resolve_inference(load_with_extends(LAB_DIR / "inference" / "configs" / "compare_35M.yaml"))
    assert example["prompts"] == benchmark["prompts"]

def test_weights_are_grouped_by_the_model_that_loads_them(tmp_path):
    small = {"family": "SAI", "d_model": 8}
    modular = {"family": "SAI_modular", "d_model": 8}
    for name in ("pretrained_35M", "sft1_35M", "sft1_999M", "ghi_chu"):
        (tmp_path / "model").mkdir(exist_ok=True)
        (tmp_path / "model" / f"{name}.pt").write_bytes(b"")

    def stage_weights(run: str, model: dict):
        run_dir = tmp_path / "runs" / "exp" / run
        (run_dir / "stages" / "sft1").mkdir(parents=True)
        (run_dir / "resolved_config.yaml").write_text(yaml.safe_dump({"model": model}), encoding="utf-8")
        (run_dir / "stages" / "sft1" / "best.pt").write_bytes(b"")

    stage_weights("r1", modular)
    stage_weights("r2", {**small, "d_model": 16})  # run đã override model: không khớp config nào
    assert weights_by_model(tmp_path, {"SAI_35M": small, "SAI_modular_35M": modular}) == {
        "SAI_35M": ["model/pretrained_35M.pt", "model/sft1_35M.pt"],
        "SAI_modular_35M": ["runs/exp/r1/stages/sft1/best.pt"],
    }

def test_describe_defaults_build_a_valid_spec(capsys, checkpoint):
    """Spec dựng từ mặc định trong schema (cộng các khoá bắt buộc) phải qua được from_dict: schema khớp loader."""
    main(["describe"])
    spec_schema = json.loads(capsys.readouterr().out)["commands"]["infer"]["spec"]

    def defaults(schema):
        return {name: defaults(field) if field["type"] == "object" else field["default"]
                for name, field in schema["fields"].items() if "default" in field or field["type"] == "object"}

    variant = {"name": "v", **defaults(spec_schema["fields"]["variants"]["items"])}
    spec = from_dict(InferenceSpec, {**defaults(spec_schema), "model": MODEL, "checkpoint": str(checkpoint), "prompts": ["a"], "variants": [variant]})
    assert spec.variants[0].generation == GenerationConfig()
