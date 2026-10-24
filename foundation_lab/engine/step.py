"""Một bước tối ưu cho causal LM: chia microbatch, tích luỹ gradient, cập nhật optimizer."""

import torch
from foundation_lab.engine.profiling import cuda_memory_gib

DEFAULT_MAX_MICROBATCH_TOKENS = 57_600

def get_adaptive_microbatch_shape(
    logical_batch_size: int,
    padded_length: int,
    max_microbatch_tokens: int = DEFAULT_MAX_MICROBATCH_TOKENS,
):
    """Trả về (số microbatch, kích thước microbatch cân bằng)."""
    microbatch_count = max(
        1,
        (logical_batch_size * padded_length + max_microbatch_tokens - 1)
        // max_microbatch_tokens,
    )
    microbatch_count = min(logical_batch_size, microbatch_count)
    microbatch_size = (
        logical_batch_size + microbatch_count - 1
    ) // microbatch_count
    microbatch_count = (
        logical_batch_size + microbatch_size - 1
    ) // microbatch_size
    return microbatch_count, microbatch_size

def _mark_peak(vram_marks, name):
    mark = cuda_memory_gib()
    if name not in vram_marks or mark["allocated"] > vram_marks[name]["allocated"]:
        vram_marks[name] = mark

def accumulate_lm_gradients(
    features, head_weight, batch, *, objective, runtime, denominator, accumulation_steps,
    max_microbatch_tokens, chunk_size, vram_marks=None,
):
    """Forward + backward một logical batch theo các microbatch, cộng dồn gradient.

    ``objective``: hàm loss lấy từ ``OBJECTIVES`` (xem hợp đồng trong ``objectives/registry.py``).
    Chuẩn hoá theo token với mẫu số cố định ("fixed_denominator"): mỗi microbatch đóng góp
    ``loss_sum / denominator / accumulation_steps``, với ``denominator`` là số token mục tiêu
    của một logical batch do recipe quyết định (không phải số token thật của batch này).

    Trả về (tổng loss của logical batch đã detach, số microbatch, kích thước microbatch).
    """
    logical_batch_size, padded_length = batch.shape
    microbatch_count, microbatch_size = get_adaptive_microbatch_shape(
        logical_batch_size, padded_length, max_microbatch_tokens
    )
    logical_loss_sum = torch.zeros((), device=batch.input_ids.device, dtype=torch.float32)

    for micro_start in range(0, logical_batch_size, microbatch_size):
        micro_end = min(micro_start + microbatch_size, logical_batch_size)
        X_micro = batch.input_ids[micro_start:micro_end]
        Y_micro = batch.labels[micro_start:micro_end]
        weight_micro = batch.loss_weight[micro_start:micro_end]
        attention_micro = batch.attention_mask[micro_start:micro_end]

        with runtime.autocast():
            hidden = features(
                X_micro,
                attention_mask=attention_micro,
                has_padding=batch.has_padding,
            )
            if vram_marks is not None:
                _mark_peak(vram_marks, "forward")

            hidden_flat = hidden.reshape(-1, hidden.size(-1))
            targets_flat = Y_micro.reshape(-1)
            weight_flat = weight_micro.reshape(-1)
            loss_sum = objective(
                hidden_flat,
                head_weight,
                targets_flat,
                weight_flat,
                chunk_size=chunk_size,
            )
            if vram_marks is not None:
                _mark_peak(vram_marks, "loss")

        logical_loss_sum += loss_sum.detach()
        scaled_loss = (
            loss_sum / denominator / accumulation_steps
        )
        runtime.scaler.scale(scaled_loss).backward()

        del X_micro, Y_micro, weight_micro, attention_micro
        del hidden, hidden_flat, targets_flat, weight_flat
        del loss_sum, scaled_loss

    return logical_loss_sum, microbatch_count, microbatch_size

def optimizer_step(model, optimizer, scheduler, scaler, max_grad_norm: float):
    """Thứ tự: unscale → clip → step → update scaler → zero_grad → scheduler.

    Lưu ý: scheduler vẫn bước kể cả khi GradScaler bỏ qua step vì gradient inf/nan (chỉ xảy ra với FP16).
    """
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    scheduler.step()
