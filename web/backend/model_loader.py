"""Nạp model chính thức (``SAI/``) và tokenizer cho web chat, từ một trong hai nguồn.

- ``local``: checkpoint ``model/<stage>_<size>.pt`` (vd. ``sft1_35M.pt``) cùng config
  ``config/models/SAI_<size>.yaml`` và ``SAI/tokenizer.model``. Checkpoint là state_dict thuần
  hoặc checkpoint resume của lab.
- ``hub``: repo ``thongbuind/SAI_<size>`` trên Hugging Face. Chỉ tải config, trọng số và tokenizer rồi dựng
  bằng ``hf_release.checkpoint``; không chạy code trong repo nên không cần ``trust_remote_code``.

Web chỉ phụ thuộc ``SAI`` và ``hf_release.checkpoint``, không import training lab.
"""

from dataclasses import dataclass
from pathlib import Path
import sentencepiece as spm
import torch
import yaml
from huggingface_hub import snapshot_download
from SAI.generate import GenerationResult
from SAI.modeling import SAIConfig, SAIModel
from hf_release.checkpoint import load_hf_folder
from web.backend.paths import MODEL_CONFIG_DIR, MODEL_DIR, TOKENIZER_MODEL

OFFICIAL_FAMILY = "SAI"
# Mô hình web phục vụ, cũng là tên file trong config/models/ và tên repo trên Hub.
SERVED_MODELS = ("SAI_35M", "SAI_100M", "SAI_375M")
SOURCES = ("local", "hub")
HUB_OWNER = "thongbuind"  # repo trên Hub: thongbuind/<tên mô hình>
LOCAL_STAGE = "sft1"      # checkpoint local: model/sft1_<cỡ>.pt
HUB_FILES = ["config.json", "model.safetensors", "tokenizer.model"]

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps:0" if torch.backends.mps.is_available()
    else "cpu"
)

@dataclass
class LoadedModel:
    model: SAIModel
    tokenizer: spm.SentencePieceProcessor

    def generate(self, message: str) -> GenerationResult:
        return self.model.generate_response(message, self.tokenizer, return_result=True)

def hub_repo_id(name: str) -> str:
    return f"{HUB_OWNER}/{name}"

def local_checkpoint(name: str) -> Path:
    """``SAI_35M`` → ``model/sft1_35M.pt``."""
    size = name.removeprefix(f"{OFFICIAL_FAMILY}_")
    return MODEL_DIR / f"{LOCAL_STAGE}_{size}.pt"

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
    # Checkpoint resume của lab bọc state_dict trong ``model_state_dict``; state_dict thuần dùng thẳng.
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)

def _ready(model: SAIModel, tokenizer_file: Path) -> LoadedModel:
    model.to(device)
    model.eval()
    return LoadedModel(model, spm.SentencePieceProcessor(model_file=str(tokenizer_file)))

def load_local(name: str) -> LoadedModel:
    model_file = local_checkpoint(name)
    if not model_file.exists():
        raise FileNotFoundError(f"Model file không tìm thấy: {model_file}")
    if not TOKENIZER_MODEL.exists():
        raise FileNotFoundError(f"Tokenizer không tìm thấy: {TOKENIZER_MODEL}")
    model = SAIModel(load_model_config(name))
    load_weights(model, model_file)
    return _ready(model, TOKENIZER_MODEL)

def load_hf_snapshot(folder: Path) -> LoadedModel:
    """Dựng model từ thư mục đã tải về (config.json, model.safetensors, tokenizer.model)."""
    return _ready(load_hf_folder(folder), folder / "tokenizer.model")

def load_hub(name: str) -> LoadedModel:
    folder = snapshot_download(hub_repo_id(name), allow_patterns=HUB_FILES)
    return load_hf_snapshot(Path(folder))

def loader_for(source: str):
    if source not in SOURCES:
        raise ValueError(f"Nguồn mô hình phải là một trong {SOURCES}, nhận được: {source!r}")
    return load_local if source == "local" else load_hub
