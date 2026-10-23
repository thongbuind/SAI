import json
import numpy as np
import pytest
from foundation_lab.tokenizer import load_tokenizer, train_tokenizer

# Đủ lớn để chứa 256 byte piece (byte_fallback) + token đặc biệt + ký tự của corpus, và không vượt
# số piece mà corpus tí hon dưới đây sinh ra được (SentencePiece báo tối đa 350).
TINY_VOCAB_SIZE = 300
SYLLABLES = [
    "xin", "chào", "việt", "nam", "mô", "hình", "ngôn", "ngữ", "học", "máy", "dữ", "liệu", "tiếng",
    "người", "bạn", "trợ", "lý", "câu", "hỏi", "trả", "lời", "thời", "gian", "hôm", "nay", "đẹp",
]

def _write_corpus(path, n_lines=2000, seed=0):
    rng = np.random.default_rng(seed)
    with open(path, "w", encoding="utf-8") as f:
        for _ in range(n_lines):
            words = rng.choice(SYLLABLES, size=int(rng.integers(3, 12)))
            f.write(json.dumps({"text": " ".join(words)}, ensure_ascii=False) + "\n")

def test_trained_tokenizer_keeps_official_special_token_ids(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    _write_corpus(corpus)
    model_path = train_tokenizer(corpus, tmp_path / "tokenizer", vocab_size=TINY_VOCAB_SIZE)

    tokenizer = load_tokenizer(model_path)
    assert tokenizer.get_piece_size() == TINY_VOCAB_SIZE
    assert [tokenizer.id_to_piece(i) for i in range(5)] == ["[UNK]", "[BOS]", "[EOS]", "<|im_start|>", "<|im_end|>"]

def test_train_refuses_existing_output_dir(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    _write_corpus(corpus, n_lines=10)
    existing = tmp_path / "tokenizer"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        train_tokenizer(corpus, existing, vocab_size=TINY_VOCAB_SIZE)
