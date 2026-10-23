"""Chuyển trọng số giữa định dạng của ``SAI`` (state_dict) và định dạng Hugging Face
(``config.json`` + ``model.safetensors``).

safetensors không lưu được hai tensor dùng chung bộ nhớ, nên ``lm_head.weight`` (weight tying với
``embed.weight``) bị bỏ khi ghi và được nối lại khi đọc. Prefix ``model.`` chỉ có trong bản HF,
không sửa tên khoá hoặc ghi lại checkpoint nguồn.
"""

import dataclasses
import json
from pathlib import Path
from safetensors.torch import load_file, save_file
from SAI.modeling import SAIConfig, SAIModel

CORE_FIELDS = tuple(field.name for field in dataclasses.fields(SAIConfig))
TIED_KEY = "lm_head.weight"
TIED_SOURCE_KEY = "embed.weight"

def save_safetensors(model: SAIModel, path: Path) -> None:
    tensors = {
        f"model.{key}": value.detach().contiguous()
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
    exported = load_file(str(folder / "model.safetensors"))
    if any(not key.startswith("model.") for key in exported):
        raise ValueError("Trọng số HF cần prefix model. của adapter chứa model lõi")
    state = {key.removeprefix("model."): value for key, value in exported.items()}
    state.setdefault(TIED_KEY, state[TIED_SOURCE_KEY])
    model.load_state_dict(state)
    return model
