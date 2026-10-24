from pathlib import Path
import sentencepiece as spm
import torch
from foundation_lab.architecture import build_model
from foundation_lab.paths import TOKENIZER_MODEL
from foundation_lab.state.checkpoint import load_model_weights
from foundation_lab.tokenizer import load_tokenizer

def load_for_inference(model_config: dict, checkpoint: Path, device: torch.device) -> tuple[torch.nn.Module, spm.SentencePieceProcessor]:
    """Dựng model theo config (mọi họ trong MODELS), nạp trọng số từ ``checkpoint``, chuyển sang eval trên ``device``.

    Tokenizer luôn là tokenizer chính thức: mọi checkpoint của lab được train trên cùng bộ id.
    """
    model = build_model(model_config)
    load_model_weights(model, checkpoint, device)
    model.to(device).eval()
    return model, load_tokenizer(TOKENIZER_MODEL)
