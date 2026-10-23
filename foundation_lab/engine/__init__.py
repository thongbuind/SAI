"""Phòng Vận hành: device/AMP/compile, microbatch + tích luỹ gradient, optimizer, lịch LR,
đóng băng tham số, vòng lặp epoch dùng chung, profiling."""

from foundation_lab.engine.optim import LR_SCHEDULES, build_adamw
from foundation_lab.engine.runtime import Runtime

__all__ = ["LR_SCHEDULES", "Runtime", "build_adamw"]
