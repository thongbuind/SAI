"""Phòng Objective: mỗi loss là một hàm thuần, có test, trả về tổng để engine chuẩn hoá."""

from foundation_lab.objectives.lm_ce import chunked_lm_loss
from foundation_lab.objectives.registry import OBJECTIVES

__all__ = ["OBJECTIVES", "chunked_lm_loss"]
