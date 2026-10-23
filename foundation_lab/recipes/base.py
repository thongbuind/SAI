from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal
import torch
from foundation_lab.core.config import ConfigError
from foundation_lab.core.log import banner, log_progress
from foundation_lab.core.registry import Registry
from foundation_lab.engine.optim import LR_SCHEDULES
from foundation_lab.eval.lm_loss import evaluate_lm_loss
from foundation_lab.objectives import OBJECTIVES
from foundation_lab.state.checkpoint import ResumeState, load_checkpoint

RECIPES = Registry("recipe")

@dataclass
class StageContext:
    """Mọi thứ pipeline chuẩn bị cho một stage; recipe chỉ đọc từ đây."""
    name: str
    index: int
    model: torch.nn.Module
    features: Callable               # model.forward_features (đã compile nếu bật)
    runtime: Any                     # foundation_lab.engine.runtime.Runtime
    data: Any                        # nguồn dữ liệu từ DATA_SOURCES
    stage_dir: Path
    metrics: Any                     # foundation_lab.state.metrics.MetricsLogger
    num_workers: int = 1
    resume_checkpoint: Path | None = None

    @property
    def best_weights_path(self) -> Path:
        return self.stage_dir / "best.pt"

    @property
    def checkpoint_path(self) -> Path:
        return self.stage_dir / "last.ckpt.pt"

@dataclass
class LMTrainParams:
    """Tham số chung của các recipe causal LM. Mặc định = hành vi dev_2."""
    epochs: int
    learning_rate: float
    weight_decay: float
    batch_size: int = 64
    accumulation_steps: int = 1
    max_microbatch_tokens: int = 57_600
    warmup_ratio: float = 0.1
    lr_schedule: str = "dev2_step"
    adamw_fused: bool | None = True       # None: chỉ bật fused khi chạy CUDA
    max_grad_norm: float = 1.0
    loss_chunk_size: int = 4096
    checkpoint_every_steps: int = 5000
    log_every: int = 10
    eval_log_every: int = 100
    final_eval: Literal["last", "best"] = "last"   # test trên trọng số cuối vòng lặp hay trên best theo val
    objective: str = "lm_ce"                       # loss của bước train, tên trong OBJECTIVES

    def __post_init__(self):
        # Kiểm tra ngay khi validate experiment, không đợi tới lúc stage chạy (có thể sau stage khác hàng giờ).
        if self.objective not in OBJECTIVES:
            raise ConfigError(f"objective '{self.objective}' không tồn tại; các lựa chọn: {OBJECTIVES.names()}")
        if self.lr_schedule not in LR_SCHEDULES:
            raise ConfigError(f"lr_schedule '{self.lr_schedule}' không tồn tại; các lựa chọn: {LR_SCHEDULES.names()}")

class Recipe:
    """Một phương pháp huấn luyện, viết sao cho đọc như thuật toán.

    Lớp con khai báo:
    - ``config_cls``: dataclass tham số (kiểm tra nghiêm ngặt từ ``pipeline.<stage>.params``).
    - ``requires``: capability model phải có (vd. ``causal_lm``).
    - ``data_kinds``: loại nguồn dữ liệu nhận được (vd. ``lm``, ``sft``).
    và cài ``run() -> dict`` trả về kết quả stage (được ghi vào ``done.json``).
    """

    config_cls = LMTrainParams
    requires: frozenset = frozenset()
    data_kinds: frozenset = frozenset()

    def __init__(self, cfg, ctx: StageContext):
        self.cfg = cfg
        self.ctx = ctx

    def run(self) -> dict:
        raise NotImplementedError

    # Tiện ích dùng chung cho các recipe causal LM

    def adamw_fused(self) -> bool:
        if self.cfg.adamw_fused is None:
            return self.ctx.runtime.amp_enabled
        return self.cfg.adamw_fused

    def maybe_resume(self, optimizer, scheduler) -> ResumeState:
        ctx = self.ctx
        if ctx.resume_checkpoint is None:
            return ResumeState()
        return load_checkpoint(
            ctx.resume_checkpoint, ctx.model, optimizer, scheduler, ctx.runtime.device,
            scaler=ctx.runtime.scaler,
        )

    def final_test(self, test_loader) -> float:
        ctx, cfg = self.ctx, self.cfg
        banner(f"ĐÁNH GIÁ {ctx.name.upper()} TRÊN TEST SET")
        if cfg.final_eval == "best":
            ctx.model.load_state_dict(torch.load(ctx.best_weights_path, map_location=ctx.runtime.device))
        test_loss = evaluate_lm_loss(
            ctx.model, ctx.features, test_loader,
            runtime=ctx.runtime,
            max_microbatch_tokens=cfg.max_microbatch_tokens,
            chunk_size=cfg.loss_chunk_size,
            label=f"{ctx.name} Test",
            log_every=cfg.eval_log_every,
        )
        log_progress(f"{ctx.name} Test Loss ({cfg.final_eval} weights): {test_loss:.4f}")
        ctx.metrics.log({"stage": ctx.name, "split": "test", "weights": cfg.final_eval, "loss": test_loss})
        return test_loss
