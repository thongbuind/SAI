import torch.optim as optim
from foundation_lab.core.registry import Registry

LR_SCHEDULES = Registry("lr_schedule")

def build_adamw(params, *, lr: float, weight_decay: float, fused: bool):
    """AdamW: weight decay áp lên MỌI tham số được truyền vào (kể cả norm, embedding)."""
    return optim.AdamW(params, lr=lr, weight_decay=weight_decay, fused=fused)

def warmup_stable_decay_lambda(warmup_steps, total_steps):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        elif current_step < total_steps * 0.4:
            return 1.0
        else:
            progress = (current_step - total_steps * 0.4) / (total_steps * 0.3)
            return max(0.1, 1.0 - 0.9 * progress)
    return lr_lambda

@LR_SCHEDULES.register("warmup_stable_decay")
def warmup_stable_decay_schedule(optimizer, warmup_steps: int, total_steps: int):
    """Warmup tuyến tính → giữ nguyên tới 40% → giảm tuyến tính tới 0.1 ở 70% → giữ 0.1."""
    return optim.lr_scheduler.LambdaLR(optimizer, warmup_stable_decay_lambda(warmup_steps, total_steps))
