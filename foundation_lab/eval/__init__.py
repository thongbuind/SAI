"""Phòng Đánh giá: loss val/test, công cụ so sánh (component, sau này checkpoint) và thước đo văn bản sinh."""

from foundation_lab.eval.compare import compare_attention_kernels
from foundation_lab.eval.lm_loss import evaluate_lm_loss
from foundation_lab.eval.text import distinct_n

__all__ = ["compare_attention_kernels", "distinct_n", "evaluate_lm_loss"]
