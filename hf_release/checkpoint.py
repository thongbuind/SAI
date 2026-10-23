"""Chuyển trọng số giữa định dạng của ``SAI`` (state_dict) và định dạng Hugging Face
(``config.json`` + ``model.safetensors``).

safetensors không lưu được hai tensor dùng chung bộ nhớ, nên ``lm_head.weight`` (weight tying với
``embed.weight``) bị bỏ khi ghi và được nối lại khi đọc.
"""

import dataclasses
import json
from pathlib import Path
from safetensors.torch import load_file, save_file
from SAI.modeling import SAIConfig, SAIModel

CORE_FIELDS = tuple(field.name for field in dataclasses.fields(SAIConfig))
TIED_KEY = "lm_head.weight"
TIED_SOURCE_KEY = "embed.weight"

def save_safetensors(model: SAIModel, path: Path):
    tensors = {
        key: value.detach().contiguous()
        for key, value in model.state_dict().items()
        if key != TIED_KEY
    }
    save_file(tensors, str(path), metadata={"format": "pt"})

def core_config_from_hf(config: dict) -> SAIConfig:
    """Lấy các trường của ``SAI.modeling.SAIConfig`` từ ``config.json`` của Hugging Face."""
    if config.get("model_type") != "SAI":
        raise ValueError(f"config.json có model_type={config.get('model_type')!r}, cần 'SAI'")
    return SAIConfig(**{name: config[name] for name in CORE_FIELDS})

def load_hf_folder(folder: Path) -> SAIModel:
    """Dựng ``SAI.modeling.SAIModel`` từ một thư mục định dạng Hugging Face, không chạy code trong thư mục đó."""
    folder = Path(folder)
    with open(folder / "config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
    model = SAIModel(core_config_from_hf(config))
    state = load_file(str(folder / "model.safetensors"))
    state.setdefault(TIED_KEY, state[TIED_SOURCE_KEY])
    model.load_state_dict(state)
    return model
