"""Nạp model chính thức (``SAI/``) và tokenizer chính thức cho web chat.

Checkpoint trong ``model/`` đặt tên ``<stage>_<size>.pt`` (vd. ``sft1_35M.pt``) và dùng config
``config/models/sai_<size>.yaml``. Checkpoint là state_dict thuần (dev_2) hoặc checkpoint của lab.
Web chỉ phụ thuộc ``SAI``, không import training lab.
"""

from pathlib import Path
import sentencepiece as spm
import torch
import yaml
from SAI.modeling import SAIConfig, SAIModel
from web.backend.paths import MODEL_CONFIG_DIR, PROJECT_ROOT, TOKENIZER_MODEL

OFFICIAL_FAMILY = "SAI"

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps:0" if torch.backends.mps.is_available()
    else "cpu"
)
model_dir = PROJECT_ROOT / "model"

def model_config_name(model_file: Path) -> str:
    """``sft1_35M.pt`` → ``SAI_35M`` (tên file trong ``config/models/``)."""
    size = model_file.stem.rsplit("_", 1)[-1]
    return f"SAI_{size}"

def load_model_config(name: str) -> SAIConfig:
    """Đọc ``config/models/<name>.yaml`` của model chính thức thành ``SAIConfig``.

    Config chính thức là một file phẳng (``family: SAI``, không ``extends``); khoá thừa hoặc thiếu
    làm ``SAIConfig`` báo lỗi ngay.
    """
    path = MODEL_CONFIG_DIR / f"{name}.yaml"
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data.get("family") != OFFICIAL_FAMILY or "extends" in data:
        raise ValueError(f"{path}: web chỉ dùng config model chính thức (family: SAI, không extends)")
    fields = {key: value for key, value in data.items() if key != "family"}
    return SAIConfig(**fields)

def load_weights(model: SAIModel, model_file: Path):
    state = torch.load(model_file, map_location=device)
    # Checkpoint resume của lab bọc state_dict trong ``model_state_dict``; state_dict thuần (dev_2) dùng thẳng.
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)

def load_model(model_file: Path) -> tuple[SAIModel, spm.SentencePieceProcessor]:
    if not model_file.exists():
        raise FileNotFoundError(f"Model file không tìm thấy: {model_file}")
    if not TOKENIZER_MODEL.exists():
        raise FileNotFoundError(f"Tokenizer không tìm thấy: {TOKENIZER_MODEL}")

    config = load_model_config(model_config_name(model_file))
    model = SAIModel(config)
    load_weights(model, model_file)
    model.to(device)
    model.eval()

    params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded ({model_file.name}). Params: {params:,}")

    tokenizer = spm.SentencePieceProcessor(model_file=str(TOKENIZER_MODEL))
    return model, tokenizer

def generate_response(model: SAIModel, tokenizer: spm.SentencePieceProcessor, user_input: str) -> str:
    return model.generate_response(user_input, tokenizer)
