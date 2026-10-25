"""Dữ liệu và config tí hon cho test (chạy trên CPU, vài MB RAM)."""

import json
from pathlib import Path
import numpy as np
from SAI.generate import GenerationConfig

BOS, EOS = 1, 2
VOCAB = 64
OFFICIAL_VOCAB = 10000
# Số token dành cho prompt, ngoài max_new_tokens, trong cửa sổ của model tí hon chạy lệnh infer.
INFER_PROMPT_ROOM = 256

TINY_MODEL = {
    "family": "SAI",
    "vocab_size": VOCAB,
    "d_model": 32,
    "num_heads": 4,
    "num_kv_heads": 2,
    "num_layers": 2,
    "ff_dim": 64,
    "max_seq_len": 256,
    "dropout": 0.0,
}

# Model tí hon cho lệnh infer: vocab bằng tokenizer chính thức để giải mã thật bằng SAI.generate, cửa sổ đủ cho
# max_new_tokens mặc định cộng prompt vì spec infer luôn phải có biến thể GenerationConfig mặc định.
INFER_MODEL = {
    **TINY_MODEL,
    "vocab_size": OFFICIAL_VOCAB,
    "max_seq_len": GenerationConfig().max_new_tokens + INFER_PROMPT_ROOM,
}

def make_lm_samples(n, rng, min_len=3, max_len=40):
    """Mẫu pretrain kiểu ``foundation_lab.tokenizer.encode``: x = [BOS] + tokens, y = tokens + [EOS]."""
    samples = []
    for _ in range(n):
        tokens = rng.integers(3, VOCAB, size=int(rng.integers(min_len, max_len))).tolist()
        samples.append(([BOS] + tokens, tokens + [EOS]))
    return samples

def make_sft_samples(n, rng, min_len=4, max_len=40):
    """Mẫu SFT kiểu ``foundation_lab.tokenizer.encode``: x = [BOS] + hội thoại + [EOS], y = x[1:], mask chỉ phủ phần trả lời + EOS."""
    samples = []
    for _ in range(n):
        body = rng.integers(3, VOCAB, size=int(rng.integers(min_len, max_len))).tolist()
        split = int(rng.integers(1, len(body)))
        x = [BOS] + body + [EOS]
        y = x[1:]
        mask = [0] * split + [1] * (len(y) - split)
        samples.append((x, y, mask))
    return samples

def write_npz(path, samples, has_mask=False):
    """Ghi theo đúng định dạng ``foundation_lab.data.io.write_npz``."""
    xs = [s[0] for s in samples]
    ys = [s[1] for s in samples]

    def flat(seqs, dtype):
        offsets = np.zeros(len(seqs) + 1, dtype=np.int64)
        offsets[1:] = np.cumsum([len(s) for s in seqs])
        return np.concatenate([np.asarray(s, dtype=dtype) for s in seqs]), offsets

    x_flat, x_off = flat(xs, np.int32)
    y_flat, y_off = flat(ys, np.int32)
    out = {
        "X_flat": x_flat, "X_offsets": x_off,
        "Y_flat": y_flat, "Y_offsets": y_off,
        "lengths": np.array([len(x) for x in xs], dtype=np.int32),
    }
    if has_mask:
        m_flat, m_off = flat([s[2] for s in samples], np.int8)
        out["M_flat"] = m_flat
        out["M_offsets"] = m_off
    np.savez_compressed(path, **out)

def write_tiny_dataset(processed_dir: Path, seed: int = 0) -> dict:
    """Tạo đủ bộ file data/processed mà pipeline chuẩn cần; trả về đường dẫn từng nguồn."""
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    shards = []
    for name, count in (("pretrain_data_001", 30), ("pretrain_data_002", 30)):
        samples = make_lm_samples(count, rng)
        write_npz(processed_dir / f"{name}_ids.npz", samples)
        shards.append({
            "name": name, "file": f"{name}_ids.npz", "n_samples": count,
            "n_tokens": sum(len(s[0]) for s in samples),
        })
    manifest = {
        "shards": shards,
        "total_samples": sum(s["n_samples"] for s in shards),
        "total_tokens": sum(s["n_tokens"] for s in shards),
    }
    (processed_dir / "pretrain_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    write_npz(processed_dir / "continued_pretrain_data_ids.npz", make_lm_samples(10, rng))
    write_npz(processed_dir / "SFT1_data_ids.npz", make_sft_samples(40, rng), has_mask=True)
    write_npz(processed_dir / "SFT2_data_ids.npz", make_sft_samples(20, rng), has_mask=True)
    return {
        "pretrain": processed_dir / "pretrain_manifest.json",
        "continued": processed_dir / "continued_pretrain_data_ids.npz",
        "sft1": processed_dir / "SFT1_data_ids.npz",
        "sft2": processed_dir / "SFT2_data_ids.npz",
    }

def tiny_experiment(paths: dict, *, seed=0, dropout=0.0, epochs=2, accumulation_steps=2,
                    train_ratio=0.8, val_ratio=0.1, freeze_blocks=(0,), continued=True) -> dict:
    """Experiment tí hon có cùng cấu trúc với core/configs/base.yaml, trỏ vào dữ liệu ``paths``."""
    split = {"train_ratio": train_ratio, "val_ratio": val_ratio}
    return {
        "name": "tiny",
        "model": {**TINY_MODEL, "dropout": dropout},
        "runtime": {"device": "cpu", "seed": seed, "compile": False, "num_workers": 0},
        "stage_defaults": {
            "batch_size": 4,
            "accumulation_steps": accumulation_steps,
            "max_microbatch_tokens": 256,
        },
        "pipeline": {
            "pretrain": {
                "recipe": "causal_lm",
                "data": {"kind": "lm", "main": str(paths["pretrain"]), **split},
                "params": {"epochs": epochs, "learning_rate": 1e-3, "weight_decay": 0.2,
                           "warmup_ratio": 0.1, "adamw_fused": True, "final_eval": "last"},
            },
            "continued": {
                "enabled": continued,
                "recipe": "causal_lm",
                "data": {"kind": "lm", "main": str(paths["continued"]), "sub": str(paths["pretrain"]),
                         "mix_ratio": 3, **split},
                "init": "pretrain.best",
                "params": {"epochs": epochs, "learning_rate": 5e-4, "weight_decay": 0.1,
                           "warmup_ratio": 0.1, "adamw_fused": True, "final_eval": "last"},
            },
            "sft1": {
                "recipe": "sft",
                "data": {"kind": "sft", "main": str(paths["sft1"]), **split},
                "init": "pretrain.best",
                "params": {"epochs": epochs, "learning_rate": 5e-4, "weight_decay": 0.05,
                           "freeze_blocks": list(freeze_blocks)},
            },
            "sft2": {
                "recipe": "sft",
                "data": {"kind": "sft", "main": str(paths["sft2"]), "sub": str(paths["sft1"]),
                         "mix_ratio": 1, **split},
                "init": "sft1.best",
                "params": {"epochs": epochs, "learning_rate": 3e-4, "weight_decay": 0.05,
                           "freeze_blocks": list(freeze_blocks), "resample_per_epoch": True},
            },
        },
    }
