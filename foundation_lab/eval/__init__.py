"""Phòng Đánh giá: loss val/test và công cụ so sánh (component, sau này checkpoint và benchmark sinh)."""

from foundation_lab.eval.compare import compare_attention_kernels
from foundation_lab.eval.lm_loss import evaluate_lm_loss

__all__ = ["compare_attention_kernels", "evaluate_lm_loss"]
