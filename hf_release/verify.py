"""Verify: tải thư mục export đúng như người dùng sẽ tải từ Hub (AutoModel + trust_remote_code), rồi so
với model chính thức dựng trực tiếp từ ``SAI.modeling``. Push chỉ được phép khi verify đạt.

Kiểm tra:
- tên khoá state_dict giống nhau, ``lm_head`` vẫn dùng chung trọng số với ``embed``;
- logits của forward, prefill và decode_step khớp (sai khác ≤ ``TOLERANCE``);
- tokenizer tải qua AutoTokenizer cho cùng token id với SentencePiece và decode lại đúng văn bản.
"""

import json
from pathlib import Path
import sentencepiece as spm
import torch
from foundation_lab.state.lineage import sha256_file
from hf_release.checkpoint import load_hf_folder

TOLERANCE = 1e-6
SAMPLE_TEXT = "xin chào, đây là mô hình ngôn ngữ sai của việt nam."

def _max_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).abs().max().item()

def verify_export(export_dir) -> dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    export_dir = Path(export_dir)
    reference = load_hf_folder(export_dir).eval()
    hf_model = AutoModelForCausalLM.from_pretrained(export_dir, trust_remote_code=True).eval()

    vocab_size = reference.embed.num_embeddings
    input_ids = torch.arange(24).remainder(vocab_size - 3).add(3).view(2, 12)   # (B, T) token id cố định
    with torch.no_grad():
        forward_diff = _max_diff(reference(input_ids), hf_model(input_ids))
        ref_cache = reference.init_cache(2, 12, input_ids.device)
        hf_cache = hf_model.init_cache(2, 12, input_ids.device)
        prefill_diff = _max_diff(
            reference.prefill(input_ids[:, :8], kv_cache=ref_cache)[0],
            hf_model.prefill(input_ids[:, :8], kv_cache=hf_cache)[0],
        )
        decode_diff = _max_diff(
            reference.decode_step(input_ids[:, 8], ref_cache, cache_len=8),
            hf_model.decode_step(input_ids[:, 8], hf_cache, cache_len=8),
        )

    tokenizer = AutoTokenizer.from_pretrained(export_dir, trust_remote_code=True)
    sentencepiece = spm.SentencePieceProcessor(model_file=str(export_dir / "tokenizer.model"))
    token_ids = tokenizer.encode(SAMPLE_TEXT, add_special_tokens=False)

    checks = {
        "state_dict_keys_match": set(hf_model.state_dict()) == set(reference.state_dict()),
        "weights_tied": hf_model.lm_head.weight.data_ptr() == hf_model.embed.weight.data_ptr(),
        "forward_logits_match": forward_diff <= TOLERANCE,
        "prefill_logits_match": prefill_diff <= TOLERANCE,
        "decode_logits_match": decode_diff <= TOLERANCE,
        "tokenizer_ids_match": token_ids == sentencepiece.encode(SAMPLE_TEXT),
        "tokenizer_roundtrip": tokenizer.decode(token_ids) == SAMPLE_TEXT,
    }
    report = {
        "passed": all(checks.values()),
        "checks": checks,
        "max_abs_diff": {"forward": forward_diff, "prefill": prefill_diff, "decode": decode_diff},
        "model_safetensors_sha256": sha256_file(export_dir / "model.safetensors"),
    }
    with open(export_dir / "verify.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return report
