"""Phòng Đánh giá: loss val/test, công cụ so sánh (component, sau này checkpoint) và thước đo văn bản sinh.

Tác vụ ``compare-kernel`` (trang Kiến trúc của web lab) và ``compare-objective`` (trang Objective) nằm trong
``compare.py``, cạnh công cụ so kernel và so objective mà chúng dùng.
"""

from foundation_lab.eval.compare import attention_shape, compare_attention_kernels
from foundation_lab.eval.lm_loss import evaluate_lm_loss
from foundation_lab.eval.text import distinct_n

__all__ = ["attention_shape", "compare_attention_kernels", "distinct_n", "evaluate_lm_loss"]
