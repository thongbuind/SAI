import math
import pytest
import torch
from helpers import TINY_MODEL, tiny_experiment
import foundation_lab
from foundation_lab.catalog import build_model
from foundation_lab.core.experiment import resolve_experiment_dict
from foundation_lab.data.loaders import BucketPadCollator
from foundation_lab.engine import loop
from foundation_lab.engine.optim import build_adamw
from foundation_lab.engine.runtime import Runtime
from foundation_lab.core.experiment import RuntimeSpec
from foundation_lab.engine.step import accumulate_lm_gradients, optimizer_step
from foundation_lab.objectives import chunked_lm_loss
from foundation_lab.paths import PROJECT_ROOT, TOKENIZER_MODEL, project_relative_path
from foundation_lab.pipeline import run_experiment
from foundation_lab.state.lineage import collect_lineage
from foundation_lab.state.metrics import MetricsLogger
from foundation_lab.state.run import RunDir

def _run(tmp_path, data, **kwargs):
    resolved = resolve_experiment_dict(data)
    run_dir = RunDir.create("tiny", root=tmp_path / "runs", run_id="r0")
    run_dir.write_config(resolved)
    return run_dir, run_experiment(resolved, run_dir, **kwargs)

def test_pipeline_end_to_end_writes_run_artifacts(tmp_path, tiny_data):
    run_dir, results = _run(tmp_path, tiny_experiment(tiny_data, dropout=0.05))
    assert list(results) == ["pretrain", "continued", "sft1", "sft2"]
    for stage in results:
        assert run_dir.is_done(stage)
        assert (run_dir.path / "stages" / stage / "best.pt").exists()
        assert (run_dir.path / "stages" / stage / "last.ckpt.pt").exists()
        assert results[stage]["test_loss"] > 0
    records = MetricsLogger.read(run_dir.metrics_path)
    assert {r["split"] for r in records} == {"train", "val", "test"}
    assert {r["stage"] for r in records} == set(results)

def test_frozen_blocks_do_not_change_during_sft(tmp_path, tiny_data):
    run_dir, _ = _run(tmp_path, tiny_experiment(tiny_data, continued=False))
    before = torch.load(run_dir.best_weights("pretrain"))
    after = torch.load(run_dir.best_weights("sft1"))
    for key in before:
        if key.startswith("blocks.0."):
            assert torch.equal(before[key], after[key]), key
    assert not torch.equal(before["blocks.1.gqa.wqkv.weight"], after["blocks.1.gqa.wqkv.weight"])

def test_resume_skips_finished_stages(tmp_path, tiny_data):
    data = tiny_experiment(tiny_data, continued=False)
    run_dir, first = _run(tmp_path, data, stages=["pretrain"])
    resolved = run_dir.read_config()
    results = run_experiment(resolved, run_dir, resume=True)
    assert results["pretrain"] == first["pretrain"]
    assert run_dir.is_done("sft1") and run_dir.is_done("sft2")

class _Interrupted(Exception):
    """Giả lập tiến trình train bị dừng giữa chừng."""

@pytest.mark.parametrize("crash_at_step,checkpoint_batches_done", [
    (4, 4),         # giữa epoch 1: checkpoint cuối là step 2 (sau batch 4), step 3 đã chạy nhưng mất
    (7, None),      # step đầu epoch 2: checkpoint cuối là checkpoint cuối epoch 1
])
def test_resume_matches_uninterrupted_run(tmp_path, tiny_data, monkeypatch, crash_at_step, checkpoint_batches_done):
    # 48 mẫu train / batch 4 = 12 batch mỗi epoch; tích luỹ 2 → 6 optimizer step mỗi epoch.
    # Bật dropout để đối chiếu cả việc khôi phục RNG đúng chỗ.
    data = tiny_experiment(tiny_data, dropout=0.05, continued=False)
    data["stage_defaults"]["checkpoint_every_steps"] = 2
    full, _ = _run(tmp_path / "full", data, stages=["pretrain"])

    real_optimizer_step = loop.optimizer_step
    completed_steps = []

    def optimizer_step_then_crash(*args, **kwargs):
        if len(completed_steps) == crash_at_step - 1:
            raise _Interrupted
        completed_steps.append(True)
        return real_optimizer_step(*args, **kwargs)

    monkeypatch.setattr(loop, "optimizer_step", optimizer_step_then_crash)
    with pytest.raises(_Interrupted):
        _run(tmp_path / "resumed", data, stages=["pretrain"])
    monkeypatch.undo()

    run_dir = RunDir(tmp_path / "resumed" / "runs" / "tiny" / "r0")
    assert torch.load(run_dir.last_checkpoint("pretrain"))["batches_done"] == checkpoint_batches_done
    run_experiment(run_dir.read_config(), run_dir, stages=["pretrain"], resume=True)
    _assert_same_final_weights(full, run_dir, "pretrain")

@pytest.mark.parametrize("crash_epoch,crash_mid_epoch", [
    (0, False),     # ngay sau checkpoint cuối epoch 1: epoch 2 dựng lại loader khi resume
    (1, True),      # ngay sau checkpoint đầu tiên của epoch 2, loader đã được dựng lại
])
def test_resume_with_resampled_loader_matches_uninterrupted_run(
    tmp_path, tiny_data, monkeypatch, crash_epoch, crash_mid_epoch,
):
    # sft2 dựng lại train loader mỗi epoch (resample_per_epoch); khởi tạo mới để chạy riêng stage này.
    data = tiny_experiment(tiny_data, dropout=0.05, continued=False)
    data["pipeline"]["sft2"]["init"] = None
    data["stage_defaults"]["checkpoint_every_steps"] = 1
    full, _ = _run(tmp_path / "full", data, stages=["sft2"])

    real_save_checkpoint = loop.save_checkpoint

    def save_checkpoint_then_crash(path, epoch, *args, batches_done=None, **kwargs):
        real_save_checkpoint(path, epoch, *args, batches_done=batches_done, **kwargs)
        if epoch == crash_epoch and (batches_done is not None) == crash_mid_epoch:
            raise _Interrupted

    monkeypatch.setattr(loop, "save_checkpoint", save_checkpoint_then_crash)
    with pytest.raises(_Interrupted):
        _run(tmp_path / "resumed", data, stages=["sft2"])
    monkeypatch.undo()

    run_dir = RunDir(tmp_path / "resumed" / "runs" / "tiny" / "r0")
    run_experiment(run_dir.read_config(), run_dir, stages=["sft2"], resume=True)
    _assert_same_final_weights(full, run_dir, "sft2")

def _assert_same_final_weights(expected_run: RunDir, resumed_run: RunDir, stage: str):
    expected, resumed = torch.load(expected_run.best_weights(stage)), torch.load(resumed_run.best_weights(stage))
    assert all(torch.equal(expected[k], resumed[k]) for k in expected)
    expected_last = torch.load(expected_run.last_checkpoint(stage))["model_state_dict"]
    resumed_last = torch.load(resumed_run.last_checkpoint(stage))["model_state_dict"]
    assert all(torch.equal(expected_last[k], resumed_last[k]) for k in expected_last)

def test_missing_init_reference_fails_with_clear_message(tmp_path, tiny_data):
    with pytest.raises(FileNotFoundError, match="pretrain.best"):
        _run(tmp_path, tiny_experiment(tiny_data), stages=["sft1"])

def test_seeded_runs_are_reproducible(tmp_path, tiny_data):
    data = tiny_experiment(tiny_data, dropout=0.05, continued=False, epochs=1)
    a, _ = _run(tmp_path / "a", data, stages=["pretrain"])
    b, _ = _run(tmp_path / "b", data, stages=["pretrain"])
    wa, wb = torch.load(a.best_weights("pretrain")), torch.load(b.best_weights("pretrain"))
    assert all(torch.equal(wa[k], wb[k]) for k in wa)

def test_tiny_data_fit():
    """Lưới an toàn: vòng tối ưu phải học thuộc được vài mẫu (loss → gần 0)."""
    torch.manual_seed(0)
    model = build_model(TINY_MODEL)
    runtime = Runtime(RuntimeSpec(device="cpu", compile=False))
    rng = torch.Generator().manual_seed(0)
    items = []
    for _ in range(4):
        tokens = torch.randint(3, 64, (12,), generator=rng).tolist()
        items.append(([1] + tokens, tokens + [2], 13, None))
    batch = BucketPadCollator(max_seq_len=256, with_mask=False)(items)
    optimizer = build_adamw(model.parameters(), lr=1e-2, weight_decay=0.0, fused=False)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    denominator = batch.num_loss_tokens.item()
    for _ in range(200):
        loss_sum, _, _ = accumulate_lm_gradients(
            model.forward_features, model.lm_head.weight, batch, objective=chunked_lm_loss, runtime=runtime,
            denominator=denominator, accumulation_steps=1, max_microbatch_tokens=256, chunk_size=64,
        )
        optimizer_step(model, optimizer, scheduler, runtime.scaler, max_grad_norm=1.0)
    # Token đầu sau BOS không đoán được (4 mẫu cùng bắt đầu bằng BOS): loss tổng tối thiểu = 4·ln 4.
    floor = 4 * math.log(4)
    assert (loss_sum.item() - floor) / denominator < 0.01

def test_lineage_records_lab_version():
    assert collect_lineage("hash", {})["lab_version"] == foundation_lab.__version__

def test_project_relative_path_drops_machine_prefix_only_inside_project(tmp_path):
    assert project_relative_path(PROJECT_ROOT / "model" / "sft1_35M.pt") == "model/sft1_35M.pt"
    assert project_relative_path(str(PROJECT_ROOT / "runs")) == "runs"
    # Ngoài dự án, vốn tương đối hoặc không phải đường dẫn (tham số CLI): giữ nguyên.
    assert project_relative_path(tmp_path / "x.pt") == str(tmp_path / "x.pt")
    assert project_relative_path("compare-kernel") == "compare-kernel"
    assert project_relative_path("-") == "-"

def test_lineage_paths_are_project_relative(tmp_path):
    outside = tmp_path / "shard.bin"
    outside.write_bytes(b"x")
    lineage = collect_lineage("hash", {"pretrain": [PROJECT_ROOT / "model" / "missing.pt", outside]})
    assert lineage["tokenizer"]["path"] == TOKENIZER_MODEL.relative_to(PROJECT_ROOT).as_posix()
    assert [item["path"] for item in lineage["data"]["pretrain"]] == ["model/missing.pt", str(outside)]
