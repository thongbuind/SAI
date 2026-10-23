"""Kiểm chứng bộ phát hành đã đóng băng qua giao diện Transformers và model lõi."""

from pathlib import Path
from importlib.metadata import version
import sys
import sentencepiece as spm
import torch
from SAI.generate import GenerationConfig, generate
from hf_release.artifact import MANIFEST_FILE, SCHEMA_VERSION, sha256_file, validate_artifact, write_json
from hf_release.checkpoint import load_hf_folder

TOLERANCE = 1e-6
PADDING_ATOL = 1e-5
PADDING_RTOL = 1e-5
SAMPLE_TEXT = "xin chào, đây là mô hình ngôn ngữ sai của việt nam."


def _max_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).abs().max().item()


def verify_export(export_dir) -> dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    export_dir = Path(export_dir)
    manifest = validate_artifact(export_dir)
    manifest_hash = sha256_file(export_dir / MANIFEST_FILE)
    reference = load_hf_folder(export_dir).eval()
    hf_model = AutoModelForCausalLM.from_pretrained(export_dir, trust_remote_code=True).eval()
    vocab_size = reference.embed.num_embeddings
    input_ids = torch.arange(24).remainder(vocab_size - 3).add(3).view(2, 12)
    with torch.no_grad():
        forward_diff = _max_diff(reference(input_ids), hf_model(input_ids, use_cache=False).logits)
        pref = hf_model(input_ids[:, :8], use_cache=True)
        ref_pref, ref_past = reference.forward_cached(input_ids[:, :8])
        prefill_diff = _max_diff(ref_pref, pref.logits)
        decoded = hf_model(input_ids[:, 8:9], past_key_values=pref.past_key_values, use_cache=True)
        ref_decoded, _ = reference.forward_cached(input_ids[:, 8:9], past_key_values=ref_past)
        decode_diff = _max_diff(ref_decoded, decoded.logits)
        padded = input_ids.clone()
        padded[0, :3] = 0
        mask = torch.ones_like(padded)
        mask[0, :3] = 0
        expected_padding = reference(input_ids[:1, 3:])
        actual_padding = hf_model(padded, attention_mask=mask).logits[:1, 3:]
        padded_diff = _max_diff(expected_padding, actual_padding)
        # Khác shape đổi thứ tự cộng dồn FP32; cùng tiêu chí với test padding của lõi.
        padding_matches = torch.allclose(expected_padding, actual_padding, atol=PADDING_ATOL, rtol=PADDING_RTOL)
        generated = hf_model.generate(input_ids=input_ids[:1, :4], max_new_tokens=3, do_sample=False)
        uncached = hf_model.generate(input_ids=input_ids[:1, :4], max_new_tokens=3, do_sample=False, use_cache=False)

    tokenizer = AutoTokenizer.from_pretrained(export_dir, trust_remote_code=True)
    sentencepiece = spm.SentencePieceProcessor(model_file=str(export_dir / "tokenizer.model"))
    token_ids = tokenizer.encode(SAMPLE_TEXT, add_special_tokens=False)
    config = GenerationConfig(strategy="greedy", max_new_tokens=2, no_repeat_ngram_size=0)
    native = generate(reference, "xin chào", sentencepiece, config=config, return_result=True)
    adapted = hf_model.generate_response("xin chào", tokenizer, config=config, return_result=True)
    checks = {
        "state_dict_keys_match": set(hf_model.model.state_dict()) == set(reference.state_dict()),
        "weights_tied": hf_model.get_output_embeddings().weight.data_ptr() == hf_model.get_input_embeddings().weight.data_ptr(),
        "forward_logits_match": forward_diff <= TOLERANCE,
        "prefill_logits_match": prefill_diff <= TOLERANCE,
        "decode_logits_match": decode_diff <= TOLERANCE,
        "left_padding_logits_match": padding_matches,
        "generation_cache_matches": torch.equal(generated, uncached),
        "native_generation_matches": native.token_ids == adapted.token_ids,
        "tokenizer_ids_match": token_ids == sentencepiece.encode(SAMPLE_TEXT),
        "tokenizer_roundtrip": tokenizer.decode(token_ids) == SAMPLE_TEXT,
        "version_matches": hf_model.config.sai_version == manifest["version"],
        "tokenizer_context_matches": tokenizer.model_max_length == reference.max_seq_len,
    }
    validate_artifact(export_dir)
    if sha256_file(export_dir / MANIFEST_FILE) != manifest_hash:
        raise ValueError("Manifest thay đổi trong lúc verify")
    report = {
        "schema_version": SCHEMA_VERSION,
        "repo_id": manifest["repo_id"],
        "version": manifest["version"],
        "passed": all(checks.values()),
        "checks": checks,
        "max_abs_diff": {"forward": forward_diff, "prefill": prefill_diff, "decode": decode_diff, "padding": padded_diff},
        "manifest_sha256": manifest_hash,
        "tolerances": {"same_shape_atol": TOLERANCE, "padding_atol": PADDING_ATOL, "padding_rtol": PADDING_RTOL},
        "environment": {
            "python": sys.version.split()[0],
            "dependencies": {name: version(name) for name in manifest["dependencies"]},
        },
    }
    write_json(export_dir / "verify.json", report)
    return report
