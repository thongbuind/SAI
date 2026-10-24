"""Vòng lặp epoch dùng chung cho các recipe causal LM.

Recipe quyết định THUẬT TOÁN (dữ liệu, mẫu số chuẩn hoá loss, optimizer, lịch LR, có dựng lại
train loader mỗi epoch không). Vòng lặp này chỉ lo CƠ CHẾ: microbatch, tích luỹ gradient, bước
optimizer, log, đánh giá val mỗi epoch, lưu best theo val và checkpoint resume.
"""

import math
from dataclasses import dataclass
import torch
from foundation_lab.core.log import log_progress
from foundation_lab.engine.profiling import (
    VRAM_PROFILE_STEPS, KernelLogger, TflopsBenchmarker, cuda_memory_gib,
    estimate_training_flops, print_training_flops_summary, print_vram_profile,
)
from foundation_lab.engine.step import accumulate_lm_gradients, optimizer_step
from foundation_lab.eval.lm_loss import evaluate_lm_loss
from foundation_lab.state.checkpoint import ResumeState, restore_rng_state, save_checkpoint, save_weights

@dataclass
class LoopSettings:
    num_epochs: int
    batch_size: int
    accumulation_steps: int
    max_microbatch_tokens: int
    total_steps: int
    max_grad_norm: float = 1.0
    chunk_size: int = 4096
    checkpoint_every_steps: int = 5000
    log_every: int = 10
    eval_log_every: int = 100

def total_optimizer_steps(batches_per_epoch: int, accumulation_steps: int, epochs: int) -> int:
    return math.ceil(batches_per_epoch / accumulation_steps) * epochs

def train_lm_epochs(
    ctx, settings: LoopSettings, *, objective, optimizer, scheduler, train_loader, val_loader,
    denominator, rebuild_train_loader=None, resume: ResumeState | None = None,
) -> ResumeState:
    """Chạy các epoch còn lại; trả về trạng thái cuối (best_val_loss, global_step...).

    - ``objective``: hàm loss của bước train (từ ``OBJECTIVES``); val luôn đo bằng CE (``lm_ce``).
    - ``denominator(train_loader)``: mẫu số chuẩn hoá loss của một logical batch (gọi lại sau mỗi lần dựng loader).
    - ``rebuild_train_loader(epoch)``: nếu có, được gọi ở đầu mỗi epoch > 0 để dựng train loader mới.
    """
    model, runtime, device = ctx.model, ctx.runtime, ctx.runtime.device
    features = ctx.features
    head_weight = model.lm_head.weight
    resume = resume or ResumeState()
    best_path = ctx.best_weights_path
    checkpoint_path = ctx.checkpoint_path
    stage = ctx.name

    flops_info = estimate_training_flops(
        model=model,
        num_layers=model.num_layers,
        max_seq_len=model.max_seq_len,
        d_model=model.d_model,
        epochs=settings.num_epochs,
        batches_per_epoch=len(train_loader),
        batch_size=settings.batch_size,
    )
    print_training_flops_summary(flops_info, settings.num_epochs)
    bench = TflopsBenchmarker(
        flops_per_token=flops_info["flops_per_token"],
        total_flops_needed=flops_info["total_flops"],
        total_batches_per_epoch=flops_info["total_batches_per_epoch"],
        epochs=settings.num_epochs,
        device=device,
        warmup_steps=500,
        target_steps=1000,
    )
    kernel_logger = KernelLogger(enabled=runtime.profile_kernels, log_step=1000, out_dir=ctx.stage_dir)
    if resume.resume_batch_idx > 0:
        bench.reported = True
    reported_peak_vram = 0.0

    global_step = resume.global_step
    best_val_loss = resume.best_val_loss
    start_epoch = resume.start_epoch

    # Khôi phục RNG đúng tại điểm tương ứng với lúc lưu checkpoint, để run resume rút cùng chuỗi số
    # ngẫu nhiên (dropout) như run chạy liền:
    # - checkpoint cuối epoch được lưu trước khi epoch sau dựng loader/iterator → khôi phục ngay đây;
    # - checkpoint giữa epoch được lưu sau khi iterator của epoch đã được tạo (tạo iterator rút một số
    #   từ RNG toàn cục) → khôi phục ngay sau khi tạo iterator, bên dưới.
    if resume.resume_batch_idx == 0:
        restore_rng_state(resume.rng_state)

    for epoch in range(start_epoch, settings.num_epochs):
        loader_rebuilt = rebuild_train_loader is not None and epoch > 0
        if loader_rebuilt:
            log_progress(f"[{stage}] Epoch {epoch+1}: Re-sampling train data...")
            train_loader = rebuild_train_loader(epoch)
        target_tokens_per_logical_batch = denominator(train_loader)

        sampler = train_loader.batch_sampler
        skip_batches = resume.resume_batch_idx if epoch == start_epoch else 0
        # Epoch đầu sau khi resume dùng lại sampler (seed, epoch) đã lưu để có đúng thứ tự batch như run
        # chạy liền. Ngoại lệ: resume ở ranh giới epoch mà loader vừa được dựng lại — sampler mới rút seed
        # từ RNG đã khôi phục nên đã khớp sẵn, còn state đã lưu thuộc sampler của epoch trước.
        if epoch == start_epoch and resume.sampler_state is not None and (skip_batches > 0 or not loader_rebuilt):
            sampler.load_state_dict(resume.sampler_state)
        if skip_batches > 0:
            # Resume giữa epoch: bỏ các batch đã train ngay ở sampler; batch_idx vẫn đếm từ đầu epoch để
            # mốc tích luỹ gradient và log không đổi.
            sampler.skip_next_batches(skip_batches)
        sampler_epoch = sampler.epoch
        model.train()
        train_loss_sum = torch.zeros((), device=device)
        train_valid_tokens = torch.zeros((), device=device)
        total_batches = len(train_loader)
        optimizer.zero_grad(set_to_none=True)
        batches = iter(train_loader)
        if skip_batches > 0:
            restore_rng_state(resume.rng_state)
        for batch_idx, batch in enumerate(batches, start=skip_batches):
            profile_vram = (
                runtime.profile_vram
                and device.type == "cuda"
                and epoch == start_epoch
                and batch_idx < VRAM_PROFILE_STEPS
            )
            vram_marks = {}
            if profile_vram:
                torch.cuda.synchronize()
                gib = 1024 ** 3
                reported_peak_vram = max(
                    reported_peak_vram,
                    torch.cuda.max_memory_reserved() / gib,
                )
                torch.cuda.reset_peak_memory_stats()
                vram_marks["start"] = cuda_memory_gib()

            batch = batch.to(device, non_blocking=True)
            batch_shape = batch.shape
            has_padding = batch.has_padding
            if profile_vram:
                vram_marks["h2d"] = cuda_memory_gib()

            _is_bench_step = bench.is_bench_step(epoch, start_epoch, batch_idx)
            if _is_bench_step:
                bench.on_step_begin(batch_idx, batch_shape[0] * batch_shape[1])

            _is_kernel_log = kernel_logger.should_log(epoch, start_epoch, batch_idx)
            if _is_kernel_log:
                kernel_logger.start()

            num_valid_tokens = batch.num_loss_tokens
            logical_loss_sum, microbatch_count, microbatch_size = accumulate_lm_gradients(
                features, head_weight, batch,
                objective=objective,
                runtime=runtime,
                denominator=target_tokens_per_logical_batch,
                accumulation_steps=settings.accumulation_steps,
                max_microbatch_tokens=settings.max_microbatch_tokens,
                chunk_size=settings.chunk_size,
                vram_marks=vram_marks if profile_vram else None,
            )

            if profile_vram:
                vram_marks["backward"] = cuda_memory_gib()

            if _is_kernel_log:
                kernel_logger.stop_and_report(batch_idx, epoch, device)

            optimizer_updated = (
                (batch_idx + 1) % settings.accumulation_steps == 0
                or (batch_idx + 1) == total_batches
            )
            if optimizer_updated:
                optimizer_step(model, optimizer, scheduler, runtime.scaler, settings.max_grad_norm)
                global_step += 1
                if global_step % settings.checkpoint_every_steps == 0:
                    save_checkpoint(
                        checkpoint_path, epoch, global_step,
                        model, optimizer, scheduler, best_val_loss,
                        batches_done=batch_idx + 1,
                        sampler_state={"seed": sampler.seed, "epoch": sampler_epoch},
                        scaler=runtime.scaler,
                    )
                    print()
                    log_progress(f"Step {global_step}: checkpoint saved → {checkpoint_path}")
            if profile_vram:
                vram_marks["update"] = cuda_memory_gib()

            train_loss_sum += logical_loss_sum
            train_valid_tokens += num_valid_tokens.detach()
            current_lr = optimizer.param_groups[0]["lr"]

            if (batch_idx + 1) % settings.log_every == 0 or (batch_idx + 1) == total_batches:
                avg_loss = (
                    train_loss_sum / (train_valid_tokens + 1e-8)
                ).item()
                vram_text = ""
                if device.type == "cuda":
                    gib = 1024 ** 3
                    current_vram = torch.cuda.memory_reserved() / gib
                    reported_peak_vram = max(
                        reported_peak_vram,
                        torch.cuda.max_memory_reserved() / gib,
                    )
                    vram_text = (
                        f" - VRAM: current={current_vram:.2f}GiB "
                        f"peak={reported_peak_vram:.2f}GiB"
                    )
                print(
                    f"\r{stage} | Epoch {epoch+1}/{settings.num_epochs} "
                    f"- Step {global_step}/{settings.total_steps} "
                    f"- loss: {avg_loss:.4f} "
                    f"- lr: {current_lr:.2e} "
                    f"- micro={microbatch_count}x<={microbatch_size}"
                    f"{vram_text}",
                    end=""
                )
                ctx.metrics.log({
                    "stage": stage, "split": "train", "epoch": epoch + 1, "batch": batch_idx + 1,
                    "optimizer_step": global_step, "loss": avg_loss, "lr": current_lr,
                })

            if _is_bench_step:
                bench.on_step_end(batch_idx)

            del batch, num_valid_tokens, logical_loss_sum
            if profile_vram:
                vram_marks["released"] = cuda_memory_gib()
                print_vram_profile(
                    batch_idx, batch_shape, has_padding, optimizer_updated,
                    microbatch_count, microbatch_size, vram_marks,
                )

        print()
        train_loss = (
            train_loss_sum / (train_valid_tokens + 1e-8)
        ).item()

        val_loss = evaluate_lm_loss(
            model, features, val_loader,
            runtime=runtime,
            max_microbatch_tokens=settings.max_microbatch_tokens,
            chunk_size=settings.chunk_size,
            label=f"{stage} Val",
            log_every=settings.eval_log_every,
        )
        log_progress(f"{stage} Epoch {epoch+1}/{settings.num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        ctx.metrics.log({
            "stage": stage, "split": "val", "epoch": epoch + 1,
            "optimizer_step": global_step, "train_loss": train_loss, "loss": val_loss,
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_weights(model, best_path)
            log_progress(f"Epoch {epoch+1}: val_loss cải thiện → lưu {best_path.name}")

        # Sampler đã duyệt xong epoch này nên state_dict() là state cho epoch kế tiếp.
        save_checkpoint(
            checkpoint_path, epoch, global_step,
            model, optimizer, scheduler, best_val_loss,
            sampler_state=sampler.state_dict(),
            scaler=runtime.scaler,
        )

    return ResumeState(
        start_epoch=settings.num_epochs,
        global_step=global_step,
        best_val_loss=best_val_loss,
    )
