import json
import pytest
from SAI.generate import GenerationConfig, _build_input
from foundation_lab.data.io import load_npz
from foundation_lab.paths import TOKENIZER_MODEL
from foundation_lab.tokenizer import ENCODERS, SpecialTokens, encode_file, encode_shards, load_tokenizer
from foundation_lab.tokenizer.chunking import split_windows
from foundation_lab.tokenizer.encode import (
    SKIP_MALFORMED_JSON, SKIP_NO_HISTORY, SKIP_NO_TEXT, SKIP_REJECTED, token_text,
)

MAX_SEQ_LEN = 64
LONG_TEXT = "rất dài " * 100      # chắc chắn vượt MAX_SEQ_LEN token

@pytest.fixture(scope="module")
def tokenizer():
    return load_tokenizer(TOKENIZER_MODEL)

@pytest.fixture(scope="module")
def special(tokenizer):
    return SpecialTokens.from_tokenizer(tokenizer)

def _segment(role, text, special, tokenizer):
    return [special.im_start] + tokenizer.encode(f"{role}\n{text}".lower(), out_type=int) + [special.im_end]

def test_official_tokenizer_special_token_ids(special):
    assert (special.bos, special.eos, special.im_start, special.im_end) == (1, 2, 3, 4)

def test_pretrain_sample_is_lowercased_and_shifted_by_one(tokenizer, special):
    [(x, y)] = ENCODERS.get("pretrain").encode("Xin Chào Việt Nam", special, tokenizer, MAX_SEQ_LEN)
    tokens = tokenizer.encode("xin chào việt nam", out_type=int)
    assert x == [special.bos] + tokens
    assert y == tokens + [special.eos]

def test_pretrain_text_fitting_max_seq_len_is_one_sample(tokenizer, special):
    n_tokens = len(tokenizer.encode("xin chào việt nam", out_type=int))
    assert len(ENCODERS.get("pretrain").encode("xin chào việt nam", special, tokenizer, n_tokens + 2)) == 1

def test_pretrain_text_longer_than_max_seq_len_is_split_after_paragraphs(tokenizer, special):
    paragraph = "hà nội là thủ đô. phố cổ rất đẹp."
    text = "\n\n".join([paragraph] * 20)
    tokens = tokenizer.encode(text, out_type=int)
    sequence = [special.bos] + tokens + [special.eos]
    windows = split_windows(tokens, MAX_SEQ_LEN, lambda token_id: token_text(tokenizer, token_id))

    samples = ENCODERS.get("pretrain").encode(text, special, tokenizer, MAX_SEQ_LEN)
    assert len(samples) == len(windows) > 1
    assert samples == [(sequence[start:end - 1], sequence[start + 1:end]) for start, end in windows]
    # Đoạn văn ngắn hơn vùng tìm điểm cuối, nên mọi cửa sổ (trừ cửa sổ cuối) phải kết thúc ngay sau hết đoạn.
    for _, end in windows[:-1]:
        assert token_text(tokenizer, sequence[end - 1]).endswith("\n")

def test_sft_mask_covers_model_turns_and_eos(tokenizer, special):
    sample = {
        "instruction": "Bạn là trợ lý",
        "history": [{"user": "Chào", "model": "Chào bạn"}, {"user": "Còn đó không", "model": ""}],
    }
    [(x, y, mask)] = ENCODERS.get("sft").encode(sample, special, tokenizer, MAX_SEQ_LEN)
    system = _segment("system", "Bạn là trợ lý", special, tokenizer)
    user = _segment("user", "Chào", special, tokenizer)
    model = _segment("model", "Chào bạn", special, tokenizer)
    assert x == [special.bos] + system + user + model + [special.eos]
    assert y == x[1:]
    assert mask == [0] * len(system) + [0] * len(user) + [1] * len(model) + [1]

def test_generate_prompt_is_prefix_of_sft_training_sample(tokenizer, special):
    """Prompt lúc sinh (``SAI.generate``) phải trùng từng token với phần đầu mẫu SFT cùng hội thoại.

    Hai nơi dựng chat template độc lập; lệch nhau (vd. ``"model\\n"`` tokenize riêng ra khác khi
    tokenize liền với câu trả lời) thì model gặp chuỗi token chưa từng thấy lúc train.
    """
    max_seq_len = 256
    instruction = "Bạn là trợ lý"
    earlier = {"user": "Chào", "model": "Chào bạn"}
    latest = {"user": "Bạn tên gì", "model": "Tôi là SAI"}
    [(x, _, _)] = ENCODERS.get("sft").encode(
        {"instruction": instruction, "history": [earlier, latest]}, special, tokenizer, max_seq_len,
    )
    prompt, _ = _build_input(
        latest["user"], tokenizer, max_seq_len=max_seq_len, config=GenerationConfig(max_new_tokens=8),
        instruction=instruction, history=[earlier],
    )
    assert x[:len(prompt)] == prompt

def test_encode_pretrain_file_round_trips_and_counts_skipped(tmp_path, tokenizer, special):
    raw = tmp_path / "raw.jsonl"
    raw.write_text("\n".join([
        json.dumps({"text": "xin chào"}, ensure_ascii=False),
        "không phải json",
        json.dumps({"other": 1}),
        "",
        json.dumps({"text": LONG_TEXT}, ensure_ascii=False),
        json.dumps({"text": "Việt Nam"}, ensure_ascii=False),
    ]) + "\n", encoding="utf-8")
    out = tmp_path / "out.npz"

    stats = encode_file("pretrain", raw, out, TOKENIZER_MODEL, MAX_SEQ_LEN)
    encode = ENCODERS.get("pretrain").encode
    # read_text_jsonl strip văn bản trước khi encode.
    texts = ["xin chào", LONG_TEXT.strip(), "Việt Nam"]
    expected = [sample for text in texts for sample in encode(text, special, tokenizer, MAX_SEQ_LEN)]
    assert len(expected) > 3                    # LONG_TEXT được chia thành nhiều mẫu, không bị bỏ
    assert stats["n_documents"] == 3
    assert stats["n_samples"] == len(expected)
    assert stats["skipped"] == {SKIP_MALFORMED_JSON: 1, SKIP_NO_TEXT: 1}

    X, Y, lengths = load_npz(out)
    assert [list(x) for x in X] == [x for x, _ in expected]
    assert [list(y) for y in Y] == [y for _, y in expected]
    assert list(lengths) == [len(x) for x, _ in expected]
    assert stats["n_tokens"] == sum(lengths)

    with pytest.raises(FileExistsError):
        encode_file("pretrain", raw, out, TOKENIZER_MODEL, MAX_SEQ_LEN)

def test_encode_sft_file_keeps_mask(tmp_path):
    raw = tmp_path / "sft.json"
    raw.write_text(json.dumps([
        {"instruction": "", "history": [{"user": "Hỏi", "model": "Đáp"}]},
        {"instruction": "", "history": []},
        "không phải object",
        {"instruction": "", "history": [{"user": LONG_TEXT, "model": "Đáp"}]},
    ], ensure_ascii=False), encoding="utf-8")
    out = tmp_path / "sft.npz"

    stats = encode_file("sft", raw, out, TOKENIZER_MODEL, MAX_SEQ_LEN)
    assert stats["n_samples"] == 1
    assert stats["skipped"] == {SKIP_NO_HISTORY: 2, SKIP_REJECTED: 1}     # hội thoại quá dài vẫn bị bỏ
    X, Y, M, _ = load_npz(out, load_mask=True)
    assert len(M[0]) == len(Y[0]) == len(X[0]) - 1

def test_encode_shards_writes_manifest_readable_as_one_dataset(tmp_path):
    for name, text in [("part_002", "việt nam"), ("part_001", "xin chào")]:
        (tmp_path / f"{name}.jsonl").write_text(json.dumps({"text": text}, ensure_ascii=False) + "\n", encoding="utf-8")
    out_dir = tmp_path / "processed"

    manifest = encode_shards(
        "pretrain", list(tmp_path.glob("part_*.jsonl")), out_dir, "manifest.json", TOKENIZER_MODEL, MAX_SEQ_LEN,
    )
    assert [shard["file"] for shard in manifest["shards"]] == ["part_001_ids.npz", "part_002_ids.npz"]
    assert manifest["total_samples"] == 2
    X, _, _ = load_npz(out_dir / "manifest.json")
    assert len(X) == 2
