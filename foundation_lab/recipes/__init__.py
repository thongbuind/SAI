"""Phòng Phương pháp huấn luyện: mỗi recipe là một phương pháp (pretrain, SFT; sau này distill, DPO, RLHF)."""

from foundation_lab.recipes import causal_lm, sft  # noqa: F401  (đăng ký recipe)
from foundation_lab.recipes.base import RECIPES, Recipe, StageContext

__all__ = ["RECIPES", "Recipe", "StageContext"]
