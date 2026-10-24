"""Checkpoint huấn luyện (để resume) và trọng số (để dùng/khởi tạo).

- Trọng số ``best.pt``: ``state_dict`` thuần, cùng định dạng các checkpoint đã phát hành (vd. ``pretrained_35M.pt``).
- Checkpoint resume ``last.ckpt.pt``: ``state_dict`` của model/optimizer/scheduler, vị trí huấn luyện,
  ``scaler_state_dict`` và ``rng_state``. Checkpoint thiếu ``scaler_state_dict``/``rng_state`` vẫn nạp được.
"""

import re
from dataclasses import dataclass
from pathlib import Path
import torch
from foundation_lab.core.log import log_progress
from foundation_lab.state.run import RunDir

CHECKPOINT_FORMAT = "foundation_lab.train_checkpoint.v1"

@dataclass
class ResumeState:
    start_epoch: int = 0
    global_step: int = 0
    best_val_loss: float = float("inf")
    resume_batch_idx: int = 0
    sampler_state: dict | None = None
    # Trạng thái RNG lúc lưu checkpoint; vòng lặp tự khôi phục đúng chỗ tương ứng (xem engine/loop.py).
    rng_state: dict | None = None

def capture_rng_state() -> dict:
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "mps": torch.mps.get_rng_state() if torch.backends.mps.is_available() else None,
    }

def restore_rng_state(state: dict | None):
    # torch.load(map_location=device) chuyển cả tensor trạng thái RNG sang device, nhưng các hàm
    # set_rng_state chỉ nhận ByteTensor trên CPU, nên phải đưa về CPU trước.
    if not state:
        return
    torch.set_rng_state(state["torch"].cpu())
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([cuda_state.cpu() for cuda_state in state["cuda"]])
    if state.get("mps") is not None and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state["mps"].cpu())

def _atomic_save(obj, path: Path):
    """Ghi file tạm rồi đổi tên, để file đích không bao giờ bị ghi dở khi tiến trình chết giữa chừng."""
    tmp_path = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp_path)
    tmp_path.replace(path)

def save_weights(model, path: Path):
    """Lưu ``state_dict`` thuần (định dạng ``best.pt``), ghi nguyên tử."""
    _atomic_save(model.state_dict(), path)

def save_checkpoint(path: Path, epoch: int, global_step: int, model, optimizer, scheduler, best_val_loss: float, batches_done=None, sampler_state=None, scaler=None):
    """Lưu toàn bộ trạng thái huấn luyện, ghi nguyên tử (file tạm rồi đổi tên).
    batches_done=None nghĩa là epoch đã xong; ngược lại là số batch đã train trong epoch."""
    _atomic_save({
        "format": CHECKPOINT_FORMAT,
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_loss": best_val_loss,
        "batches_done": batches_done,
        "sampler_state": sampler_state,
        "scaler_state_dict": scaler.state_dict() if scaler is not None and scaler.is_enabled() else None,
        "rng_state": capture_rng_state(),
    }, path)

def load_checkpoint(path: Path, model, optimizer, scheduler, device, scaler=None) -> ResumeState:
    """Nạp trạng thái huấn luyện vào model/optimizer/scheduler (và scaler nếu có).

    RNG KHÔNG được khôi phục ở đây mà trả về trong ``ResumeState.rng_state``: vòng lặp phải khôi phục
    nó đúng tại điểm tương ứng với lúc lưu, vì giữa lúc nạp và lúc đó còn bước rút số từ RNG toàn cục
    (vd. tạo iterator của DataLoader).
    """
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if scaler is not None and ckpt.get("scaler_state_dict") is not None:
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    batches_done = ckpt.get("batches_done")
    if batches_done is None:
        start_epoch = ckpt["epoch"] + 1
        resume_batch_idx = 0
    else:
        start_epoch = ckpt["epoch"]
        resume_batch_idx = batches_done
    state = ResumeState(
        start_epoch=start_epoch,
        global_step=ckpt["global_step"],
        best_val_loss=ckpt["best_val_loss"],
        resume_batch_idx=resume_batch_idx,
        sampler_state=ckpt.get("sampler_state"),
        rng_state=ckpt.get("rng_state"),
    )
    log_progress(f"Resumed from checkpoint: epoch {ckpt['epoch']+1}, step {state.global_step}, best_val_loss {state.best_val_loss:.5f}")
    if resume_batch_idx > 0:
        log_progress(f"Resume giữa epoch: bỏ qua {resume_batch_idx} batch đã train")
    return state

def load_model_weights(model, path, device, strict: bool = True):
    """Nạp trọng số từ ``state_dict`` thuần hoặc từ checkpoint resume.

    ``strict=False`` dùng cho cấy ghép module: trả về (khoá thiếu, khoá thừa) để báo cáo.
    """
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    result = model.load_state_dict(state, strict=strict)
    if not strict:
        if result.missing_keys:
            log_progress(f"Khoá thiếu (khởi tạo mới): {len(result.missing_keys)}")
        if result.unexpected_keys:
            log_progress(f"Khoá thừa (bỏ qua): {len(result.unexpected_keys)}")
    return result

# Trọng số dùng được để suy luận: checkpoint đặt tay trong model/ và best.pt của mọi stage trong runs/.
MODEL_DIR_WEIGHTS = ("model", "*.pt")
STAGE_WEIGHTS = ("runs", "*/*/stages/*/best.pt")
WEIGHT_PATTERNS = (MODEL_DIR_WEIGHTS, STAGE_WEIGHTS)
# Checkpoint đặt tay đặt tên <stage>_<cỡ>.pt (vd. sft1_375M.pt) và thuộc model chính thức SAI_<cỡ>.
MODEL_DIR_WEIGHT_NAME = re.compile(r"^[A-Za-z0-9]+_(?P<size>\d+M)$")
OFFICIAL_MODEL_PREFIX = "SAI_"
# runs/<experiment>/<run_id>/stages/<stage>/best.pt: thư mục run là cha thứ 3 của file.
STAGE_WEIGHT_RUN_PARENT = 2

def _glob_weights(project_root: Path, pattern: tuple[str, str]) -> list[Path]:
    directory, glob = pattern
    return sorted((project_root / directory).glob(glob))

def find_weights(project_root: Path) -> list[str]:
    """Đường dẫn (tương đối theo ``project_root``, dạng posix) của các file trọng số đang có trên máy."""
    return [path.relative_to(project_root).as_posix() for pattern in WEIGHT_PATTERNS for path in _glob_weights(project_root, pattern)]

def weights_by_model(project_root: Path, models: dict[str, dict]) -> dict[str, list[str]]:
    """Gom trọng số theo model nạp được chúng; ``models``: tên model → config đã resolve.

    - ``model/<stage>_<cỡ>.pt`` thuộc model chính thức ``SAI_<cỡ>``.
    - ``best.pt`` của một stage thuộc model có config trùng khoá ``model`` trong ``resolved_config.yaml`` của run.

    Trọng số không khớp quy tắc nào (tên lạ, run đã override config model) bị bỏ qua; model không có trọng số
    nào không có trong kết quả.
    """
    grouped: dict[str, list[str]] = {}

    def add(model_name: str | None, path: Path):
        if model_name in models:
            grouped.setdefault(model_name, []).append(path.relative_to(project_root).as_posix())

    for path in _glob_weights(project_root, MODEL_DIR_WEIGHTS):
        match = MODEL_DIR_WEIGHT_NAME.match(path.stem)
        add(f"{OFFICIAL_MODEL_PREFIX}{match['size']}" if match else None, path)
    for path in _glob_weights(project_root, STAGE_WEIGHTS):
        run_model = RunDir(path.parents[STAGE_WEIGHT_RUN_PARENT]).read_config().get("model")
        add(next((name for name, config in models.items() if config == run_model), None), path)
    return grouped
