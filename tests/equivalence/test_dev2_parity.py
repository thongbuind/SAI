"""Đối chiếu lab với pipeline huấn luyện cũ (trước khi có lab): cùng dữ liệu, cùng seed → cùng trọng
số và cùng test loss.

Code cũ được lấy từ lịch sử git của chính repo này (thư mục ``src/`` tại commit ``DEV2_REF``) bằng
``git archive`` vào thư mục tạm; working tree không bị đụng tới. Code cũ chạy thật (qua
``dev2_runner.py``) trên dữ liệu và config tí hon, mỗi phase một tiến trình như cách nó vẫn chạy.
Lab chạy cùng pipeline trong tiến trình pytest. Mọi thứ trên CPU, FP32, model vài chục nghìn tham số.

Đổi commit đối chiếu bằng biến môi trường ``SAI_DEV2_REF`` (commit, tag hoặc branch).
"""

import io
import json
import os
import re
import subprocess
import sys
import tarfile
from pathlib import Path
import pytest
import torch
from helpers import TINY_MODEL, tiny_experiment, write_tiny_dataset
from foundation_lab.catalog import build_model
from foundation_lab.core.experiment import resolve_experiment_dict
from foundation_lab.core.seed import seed_everything
from foundation_lab.paths import PROJECT_ROOT
from foundation_lab.pipeline import run_experiment
from foundation_lab.state.run import RunDir

DEV2_REF = os.environ.get("SAI_DEV2_REF", "56e2ff1")   # commit cuối của pipeline cũ, trước khi có lab
RUNNER = Path(__file__).with_name("dev2_runner.py")
SEED = 1234
DROPOUT = 0.05      # bật dropout để đối chiếu cả thứ tự tiêu thụ RNG

def _ref_has_old_pipeline() -> bool:
    try:
        subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "cat-file", "-e", f"{DEV2_REF}:src/training/pretrain.py"],
            check=True, capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return False
    return True

pytestmark = [
    pytest.mark.dev2,
    pytest.mark.skipif(not _ref_has_old_pipeline(),
                       reason=f"git ref {DEV2_REF} không có src/training/pretrain.py"),
]

def extract_dev2_src(dest: Path):
    """Giải nén thư mục ``src/`` tại ``DEV2_REF`` vào ``dest``."""
    archive = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "archive", "--format=tar", DEV2_REF, "src"],
        check=True, capture_output=True,
    ).stdout
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(dest, filter="data")

# Config dev_2 tương đương tiny_experiment(): cùng kiến trúc, cùng siêu tham số.
DEV2_BASE = {
    "vocab_size": TINY_MODEL["vocab_size"], "max_seq_len": TINY_MODEL["max_seq_len"], "dropout": DROPOUT,
    "pretrain_epochs": 2, "continued_pretrain_epochs": 2, "sft1_epochs": 2, "sft2_epochs": 2,
    "train_ratio": 0.8, "val_ratio": 0.1,
    "pretrain_learning_rate": 1e-3, "continued_pretrain_learning_rate": 5e-4,
    "sft1_learning_rate": 5e-4, "sft2_learning_rate": 3e-4,
    "accumulation_steps": 2, "max_microbatch_tokens": 256,
    "pretrain_weight_decay": 0.2, "continued_pretrain_weight_decay": 0.1,
    "sft1_learning_weight_decay": 0.05, "sft2_learning_weight_decay": 0.05,
}
DEV2_TINY = {
    "d_model": TINY_MODEL["d_model"], "num_heads": TINY_MODEL["num_heads"],
    "num_kv_heads": TINY_MODEL["num_kv_heads"], "num_layers": TINY_MODEL["num_layers"],
    "ff_dim": TINY_MODEL["ff_dim"], "batch_size": 4, "max_microbatch_tokens": 256,
    "freeze": [0], "num_workers": 0,
}

# (stage trong lab, chỉ số stage → seed, lệnh dev_2, file trọng số dev_2 sinh ra)
PHASES = [
    ("pretrain", 0, ["src/training/pretrain.py", "--model", "tiny", "--phase", "pretrain"], "pretrained_tiny.pt"),
    ("continued", 1, ["src/training/pretrain.py", "--model", "tiny", "--phase", "continued_pretrain"], "continued_pretrained_tiny.pt"),
    ("sft1", 2, ["src/training/finetune.py", "--model", "tiny", "--phase", "sft1"], "sft1_tiny.pt"),
    ("sft2", 3, ["src/training/finetune.py", "--model", "tiny", "--phase", "sft2"], "sft2_tiny.pt"),
]
TEST_LOSS = re.compile(r"Test Loss(?: \(CE only\))?: ([0-9.]+)")

def _run_dev2(root: Path, seed: int, args) -> str:
    proc = subprocess.run(
        [sys.executable, str(RUNNER), str(root), str(seed), *args],
        cwd=root, capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == 0, f"dev_2 {' '.join(args)} lỗi:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
    return proc.stdout

@pytest.fixture(scope="module")
def dev2_workspace(tmp_path_factory):
    """Code cũ (src/ tại DEV2_REF) + config tí hon + dữ liệu tí hon; chạy đủ 4 phase."""
    root = tmp_path_factory.mktemp("dev2")
    extract_dev2_src(root)
    (root / "config").mkdir()
    (root / "config" / "base.json").write_text(json.dumps(DEV2_BASE))
    (root / "config" / "tiny.json").write_text(json.dumps(DEV2_TINY))
    paths = write_tiny_dataset(root / "data" / "processed")

    test_losses = {}
    for stage, index, args, _ in PHASES:
        stdout = _run_dev2(root, SEED + index, args)
        test_losses[stage] = float(TEST_LOSS.findall(stdout)[-1])
    return {"root": root, "paths": paths, "test_losses": test_losses}

@pytest.fixture(scope="module")
def lab_run(dev2_workspace, tmp_path_factory):
    """Lab chạy cùng pipeline; mỗi stage khởi tạo từ đúng file dev_2 đã dùng để các stage độc lập nhau."""
    model_dir = dev2_workspace["root"] / "model"
    data = tiny_experiment(dev2_workspace["paths"], seed=SEED, dropout=DROPOUT, freeze_blocks=(0,))
    resolved = resolve_experiment_dict(data)
    run_dir = RunDir.create("dev2_parity", root=tmp_path_factory.mktemp("runs"), run_id="r0")
    results = run_experiment(
        resolved, run_dir,
        init_overrides={
            "continued": str(model_dir / "pretrained_tiny.pt"),
            "sft1": str(model_dir / "pretrained_tiny.pt"),
            "sft2": str(model_dir / "sft1_tiny.pt"),
        },
    )
    return run_dir, results

def test_model_init_matches_dev2(tmp_path):
    """Cùng seed → TransformerModel cũ và ``SAI.modeling.SAIModel`` có state_dict và logits giống hệt."""
    script = tmp_path / "dump_init.py"
    script.write_text(
        "import sys, torch\n"
        "from src.model.TransformerModel import TransformerModel\n"
        f"m = TransformerModel({TINY_MODEL['vocab_size']}, {TINY_MODEL['d_model']}, {TINY_MODEL['num_heads']}, "
        f"{TINY_MODEL['num_kv_heads']}, {TINY_MODEL['num_layers']}, {TINY_MODEL['ff_dim']}, "
        f"{TINY_MODEL['max_seq_len']}, 0.0).eval()\n"
        "ids = torch.arange(3, 23).view(2, 10)\n"
        "torch.save({'state': m.state_dict(), 'logits': m(ids).detach()}, sys.argv[1])\n"
    )
    extract_dev2_src(tmp_path)
    out = tmp_path / "init.pt"
    _run_dev2(tmp_path, SEED, ["dump_init.py", str(out)])
    reference = torch.load(out)

    seed_everything(SEED)
    model = build_model(TINY_MODEL).eval()
    state = model.state_dict()
    assert state.keys() == reference["state"].keys()
    for key in state:
        assert torch.equal(state[key], reference["state"][key]), key
    with torch.no_grad():
        logits = model(torch.arange(3, 23).view(2, 10))
    assert torch.equal(logits, reference["logits"])

@pytest.mark.parametrize("stage,dev2_file", [(p[0], p[3]) for p in PHASES])
def test_stage_weights_match_dev2(dev2_workspace, lab_run, stage, dev2_file):
    run_dir, _ = lab_run
    reference = torch.load(dev2_workspace["root"] / "model" / dev2_file)
    weights = torch.load(run_dir.best_weights(stage))
    assert weights.keys() == reference.keys()
    worst = max((weights[k] - reference[k]).abs().max().item() for k in weights)
    assert worst <= 1e-6, f"{stage}: sai khác lớn nhất {worst:.3e}"

@pytest.mark.parametrize("stage", [p[0] for p in PHASES])
def test_stage_test_loss_matches_dev2(dev2_workspace, lab_run, stage):
    _, results = lab_run
    assert abs(results[stage]["test_loss"] - dev2_workspace["test_losses"][stage]) <= 1e-4
