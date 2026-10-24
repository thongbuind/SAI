"""Recipe ``sft``: SFT1/SFT2. SFT1 và SFT2 là cùng một recipe, chỉ khác dữ liệu và tham số.

Thuật toán:
1. Đóng băng các block trong ``freeze_blocks``; optimizer chỉ chứa tham số còn trainable.
2. Dữ liệu: nguồn ``sft`` (có loss mask) → split; train dùng batch cân bằng theo số token chịu loss.
3. Objective: ``params.objective`` (mặc định ``lm_ce``: CE next-token) chỉ trên token có loss mask = 1
   (câu trả lời của model + EOS).
4. Chuẩn hoá: mẫu số cố định = ``target_loss_tokens`` của sampler (trung bình token chịu loss × batch_size).
5. Lịch LR ``warmup_stable_decay`` với warmup 15%. Mỗi epoch: val, lưu best. Cuối cùng: test trên best.

``resample_per_epoch=True`` (bật cho SFT2 trong ``core/configs/base.yaml``): mỗi epoch trộn lại nguồn phụ
với seed = epoch rồi split lại; val/test cố định được dựng từ lần trộn với ``mix_seed`` mặc định.
LỖI ĐÃ BIẾT: tập trộn khác nhau giữa các lần nên phần lớn mẫu val/test lọt vào train của mỗi epoch
(val/test loss lạc quan, việc chọn best theo val bị lệch). Sửa ở mốc M2.
"""

import gc
from dataclasses import dataclass, field
from typing import Literal
import torch
from foundation_lab.core.log import banner, log_progress
from foundation_lab.data.loaders import build_dataloader
from foundation_lab.engine.freeze import freeze_blocks, log_trainable_summary
from foundation_lab.engine.loop import LoopSettings, total_optimizer_steps, train_lm_epochs
from foundation_lab.engine.optim import LR_SCHEDULES, build_adamw
from foundation_lab.objectives import OBJECTIVES
from foundation_lab.recipes.base import RECIPES, LMTrainParams, Recipe

@dataclass
class SFTParams(LMTrainParams):
    warmup_ratio: float = 0.15
    adamw_fused: bool | None = None
    final_eval: Literal["last", "best"] = "best"
    freeze_blocks: list[int] = field(default_factory=list)
    resample_per_epoch: bool = False

@RECIPES.register("sft")
class SFTRecipe(Recipe):
    config_cls = SFTParams
    requires = frozenset({"causal_lm"})
    data_kinds = frozenset({"sft"})

    def _loader(self, split, shuffle: bool):
        cfg, ctx = self.cfg, self.ctx
        return build_dataloader(
            split.X, split.Y, split.lengths, cfg.batch_size, ctx.model.max_seq_len, ctx.num_workers,
            shuffle=shuffle, mode="finetune", loss_masks=split.loss_masks,
        )

    def _train_loader_for_epoch(self, epoch: int):
        """Dựng random train loader cho một epoch khi trộn lại dữ liệu theo epoch."""
        splits = self.ctx.data.load_splits(mix_seed=epoch)
        train_loader = self._loader(splits["train"], shuffle=True)
        del splits
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return train_loader

    def run(self) -> dict:
        cfg, ctx = self.cfg, self.ctx
        model = ctx.model

        freeze_blocks(model, cfg.freeze_blocks)
        log_trainable_summary(model)
        optimizer = build_adamw(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.learning_rate, weight_decay=cfg.weight_decay,
            fused=self.adamw_fused(),
        )

        banner(f"BẮT ĐẦU LOAD {ctx.name.upper()} DATA")
        rebuild_train_loader = None
        if not cfg.resample_per_epoch:
            splits = ctx.data.load_splits()
            log_progress(
                f"[{ctx.name}] Train: {len(splits['train'])}, Val: {len(splits['val'])}, "
                f"Test: {len(splits['test'])}"
            )
            train_loader = self._loader(splits["train"], shuffle=True)
            val_loader = self._loader(splits["val"], shuffle=False)
            test_loader = self._loader(splits["test"], shuffle=False)
            del splits
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            splits = ctx.data.load_splits()
            val_loader = self._loader(splits["val"], shuffle=False)
            test_loader = self._loader(splits["test"], shuffle=False)
            log_progress(
                f"[{ctx.name}] Train: ~{len(splits['train'])}, Val: {len(splits['val'])}, "
                f"Test: {len(splits['test'])}"
            )
            del splits
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            train_loader = self._train_loader_for_epoch(0)
            rebuild_train_loader = self._train_loader_for_epoch

        total_steps = total_optimizer_steps(len(train_loader), cfg.accumulation_steps, cfg.epochs)
        log_progress(
            f"SFT batching: token-balanced sortish, fixed loss denominator="
            f"{train_loader.batch_sampler.target_loss_tokens:,} token"
        )
        log_progress(
            f"Adaptive microbatch: max {cfg.max_microbatch_tokens:,} padded token-slot, "
            f"max logical samples={train_loader.batch_sampler.max_samples}"
        )

        warmup_steps = int(total_steps * cfg.warmup_ratio)
        scheduler = LR_SCHEDULES.get(cfg.lr_schedule)(optimizer, warmup_steps, total_steps)
        log_progress(f"Step-based LR: warmup={warmup_steps} steps, total={total_steps} steps")

        resume = self.maybe_resume(optimizer, scheduler)
        settings = LoopSettings(
            num_epochs=cfg.epochs,
            batch_size=cfg.batch_size,
            accumulation_steps=cfg.accumulation_steps,
            max_microbatch_tokens=cfg.max_microbatch_tokens,
            total_steps=total_steps,
            max_grad_norm=cfg.max_grad_norm,
            chunk_size=cfg.loss_chunk_size,
            checkpoint_every_steps=cfg.checkpoint_every_steps,
            log_every=cfg.log_every,
            eval_log_every=cfg.eval_log_every,
        )
        final = train_lm_epochs(
            ctx, settings,
            objective=OBJECTIVES.get(cfg.objective),
            optimizer=optimizer, scheduler=scheduler,
            train_loader=train_loader, val_loader=val_loader,
            denominator=lambda loader: loader.batch_sampler.target_loss_tokens,
            rebuild_train_loader=rebuild_train_loader,
            resume=resume,
        )
        test_loss = self.final_test(test_loader)
        return {
            "best_val_loss": final.best_val_loss,
            "test_loss": test_loss,
            "test_weights": cfg.final_eval,
            "optimizer_steps": final.global_step,
        }
