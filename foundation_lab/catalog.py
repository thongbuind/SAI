"""Danh mục toàn lab: import mọi phòng để các implementation tự đăng ký, gom registry về một chỗ."""

from foundation_lab.architecture import ATTENTION_KERNELS, ATTENTIONS, FFNS, MODELS, NORMS, build_model, resolve_model_config
from foundation_lab.data import DATA_SOURCES, build_data_source, resolve_data_config
from foundation_lab.engine.optim import LR_SCHEDULES
from foundation_lab.inference import DECODERS
from foundation_lab.objectives import OBJECTIVES
from foundation_lab.recipes import RECIPES
from foundation_lab.tokenizer import ENCODERS

# Phòng ban → các registry mà phòng đó sở hữu.
DEPARTMENTS = {
    "arch": [ATTENTION_KERNELS, ATTENTIONS, FFNS, NORMS, MODELS],
    "data": [DATA_SOURCES],
    "objectives": [OBJECTIVES],
    "engine": [LR_SCHEDULES],
    "recipes": [RECIPES],
    "tokenizer": [ENCODERS],
    "inference": [DECODERS],
}

__all__ = [
    "ATTENTION_KERNELS", "ATTENTIONS", "DATA_SOURCES", "DECODERS", "DEPARTMENTS", "ENCODERS", "FFNS", "LR_SCHEDULES",
    "MODELS", "NORMS", "OBJECTIVES", "RECIPES",
    "build_data_source", "build_model", "resolve_data_config", "resolve_model_config",
]
