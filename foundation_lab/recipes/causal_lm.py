"""Recipe ``causal_lm``: pretrain và continued pretrain.

Thuật toán:
1. Dữ liệu: nguồn ``lm`` → split train/val/test; train dùng sortish batch theo số mẫu.
2. Objective: ``params.objective`` (mặc định ``lm_ce``: CE next-token) trên mọi token thật (trọng số = labels != PAD).
3. Chuẩn hoá: mẫu số cố định = (số token trung bình mỗi mẫu train) × batch_size.
4. Tối ưu: AdamW trên toàn bộ tham số; lịch LR ``warmup_stable_decay`` với warmup 10%.
5. Mỗi epoch: đánh giá val, lưu best theo val. Cuối cùng: test (mặc định trên trọng số cuối, ``final_eval=last``).
"""

import gc
from dataclasses import dataclass
from foundation_lab.core.log import banner, log_progress
from foundation_lab.data.loaders import build_dataloader
from foundation_lab.engine.loop import LoopSettings, total_optimizer_steps, train_lm_epochs
from foundation_lab.engine.optim import LR_SCHEDULES, build_adamw
from foundation_lab.objectives import OBJECTIVES
from foundation_lab.recipes.base import RECIPES, LMTrainParams, Recipe

@dataclass
class CausalLMParams(LMTrainParams):
    pass

@RECIPES.register("causal_lm")
class CausalLMRecipe(Recipe):
    config_cls = CausalLMParams
    requires = frozenset({"causal_lm"})
    data_kinds = frozenset({"lm"})

    def run(self) -> dict:
        cfg, ctx = self.cfg, self.ctx
        model = ctx.model
        max_seq_len = model.max_seq_len

        banner("BAT ĐAU LOAD DATA")
        splits = ctx.data.load_splits()
        train, val, test = splits["train"], splits["val"], splits["test"]
        log_progress(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

        mean_train_tokens = float(train.lengths.sum()) / max(1, len(train.lengths))
        target_tokens_per_logical_batch = mean_train_tokens * cfg.batch_size

        train_loader = build_dataloader(
            train.X, train.Y, train.lengths, cfg.batch_size, max_seq_len, ctx.num_workers,
            shuffle=True, mode="pretrain",
        )
        val_loader = build_dataloader(
            val.X, val.Y, val.lengths, cfg.batch_size, max_seq_len, ctx.num_workers,
            shuffle=False, mode="pretrain",
        )
        test_loader = build_dataloader(
            test.X, test.Y, test.lengths, cfg.batch_size, max_seq_len, ctx.num_workers,
            shuffle=False, mode="pretrain",
        )
        del splits, train, val, test
        gc.collect()
        log_progress(
            f"Adaptive microbatch: max {cfg.max_microbatch_tokens:,} padded token-slot, "
            f"logical batch_size={cfg.batch_size}"
        )
        log_progress(
            f"Token-normalized gradient: target={target_tokens_per_logical_batch:,.1f} "
            f"valid token/logical batch"
        )

        optimizer = build_adamw(
            model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay,
            fused=self.adamw_fused(),
        )
        total_steps = total_optimizer_steps(len(train_loader), cfg.accumulation_steps, cfg.epochs)
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
            denominator=lambda _loader: target_tokens_per_logical_batch,
            resume=resume,
        )
        test_loss = self.final_test(test_loader)
        return {
            "best_val_loss": final.best_val_loss,
            "test_loss": test_loss,
            "test_weights": cfg.final_eval,
            "optimizer_steps": final.global_step,
        }
