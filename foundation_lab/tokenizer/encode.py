"""Encode văn bản thô thành chuỗi id theo định dạng dev_2.

- ``pretrain``: mỗi dòng JSONL ``{"text": ...}`` → x = [BOS] + tokens, y = tokens + [EOS]. Văn bản dài
  hơn ``max_seq_len`` được chia thành nhiều mẫu chồng lấn, cắt ở ranh giới tự nhiên (xem ``chunking``).
- ``sft``: file JSON là mảng ``{instruction, history: [{user, model}, ...]}`` →
  x = [BOS] + hội thoại + [EOS], y = x[1:] (đã shift), mask = 1 ở lượt trả lời của model và ở EOS.

Văn bản được viết thường trước khi encode (khớp ``GenerationConfig.lowercase`` mặc định). Mẫu hội
thoại ``<|im_start|>role\\n...<|im_end|>`` phải khớp với ``_build_input`` của ``SAI/generate.py``.
"""

import json
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
import sentencepiece as spm
from foundation_lab.core.log import log_progress
from foundation_lab.data.io import write_manifest, write_npz
from foundation_lab.tokenizer.chunking import split_windows
from foundation_lab.tokenizer.registry import ENCODERS, Encoder

KEY_TEXT = "text"           # khoá chứa văn bản trong mỗi dòng JSONL pretrain
SKIP_MALFORMED_JSON = "dòng JSON lỗi"
SKIP_NO_TEXT = "dòng không có text"
SKIP_NO_HISTORY = "hội thoại không có history"
SKIP_REJECTED = "mẫu bị loại khi encode (pretrain: quá ngắn; sft: quá dài hoặc rỗng)"
SHARD_SUFFIX = "_ids.npz"
MIN_PRETRAIN_TOKENS = 2     # văn bản ít token hơn bị bỏ
SPACE_MARKER = "▁"     # SentencePiece thay dấu cách bằng ký tự này trong piece
NON_ASCII_BYTE = "�"   # byte fallback ngoài ASCII: chỉ cần giữ độ dài, không cần đúng chữ

@dataclass(frozen=True)
class SpecialTokens:
    bos: int
    eos: int
    im_start: int
    im_end: int

    @classmethod
    def from_tokenizer(cls, tokenizer: spm.SentencePieceProcessor) -> "SpecialTokens":
        pieces = {"bos": "[BOS]", "eos": "[EOS]", "im_start": "<|im_start|>", "im_end": "<|im_end|>"}
        ids = {name: tokenizer.piece_to_id(piece) for name, piece in pieces.items()}
        # piece_to_id trả về id của [UNK] khi không có piece, nên phải đối chiếu ngược lại.
        missing = [piece for name, piece in pieces.items() if tokenizer.id_to_piece(ids[name]) != piece]
        if missing:
            raise ValueError(f"tokenizer thiếu token đặc biệt: {missing}")
        return cls(**ids)

def read_text_jsonl(path: Path, skipped: Counter) -> Iterator[str]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                skipped[SKIP_MALFORMED_JSON] += 1
                continue
            text = obj[KEY_TEXT].strip() if isinstance(obj, dict) and KEY_TEXT in obj else ""
            if not text:
                skipped[SKIP_NO_TEXT] += 1
                continue
            yield text

def read_sft_conversations(path: Path, skipped: Counter) -> Iterator[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        data = [data]
    for obj in data:
        if not isinstance(obj, dict) or not obj.get("history"):
            skipped[SKIP_NO_HISTORY] += 1
            continue
        yield obj

def token_text(tokenizer: spm.SentencePieceProcessor, token_id: int) -> str:
    """Phần chữ token chiếm trong văn bản; đủ để nhận ra khoảng trắng, xuống dòng và dấu câu."""
    piece = tokenizer.id_to_piece(token_id)
    if tokenizer.is_byte(token_id):
        value = int(piece[3:-1], 16)        # piece byte có dạng "<0x0A>"
        return chr(value) if value < 0x80 else NON_ASCII_BYTE
    return piece.replace(SPACE_MARKER, " ")

def encode_pretrain_text(text: str, special: SpecialTokens, tokenizer, max_seq_len: int) -> list[tuple[list[int], list[int]]]:
    tokens = tokenizer.encode(text.lower(), out_type=int)
    if len(tokens) < MIN_PRETRAIN_TOKENS:
        return []
    sequence = [special.bos] + tokens + [special.eos]
    windows = split_windows(tokens, max_seq_len, lambda token_id: token_text(tokenizer, token_id))
    # Văn bản vừa một cửa sổ cho đúng một mẫu x = [BOS] + tokens, y = tokens + [EOS] như trước.
    return [(sequence[start:end - 1], sequence[start + 1:end]) for start, end in windows]

def _chat_segment(role: str, text: str, special: SpecialTokens, tokenizer) -> list[int]:
    return [special.im_start] + tokenizer.encode(f"{role}\n{text}".lower(), out_type=int) + [special.im_end]

def encode_sft_conversation(sample: dict, special: SpecialTokens, tokenizer, max_seq_len: int) -> list[tuple[list[int], list[int], list[int]]]:
    instruction = sample.get("instruction", "").strip()
    all_tokens = []
    mask = []

    if instruction:
        system_ids = _chat_segment("system", instruction, special, tokenizer)
        all_tokens += system_ids
        mask += [0] * len(system_ids)

    for turn in sample["history"]:
        user_text = turn.get("user", "").strip()
        model_text = turn.get("model", "").strip()
        # Lượt không có câu trả lời của model thì không có gì để học, bỏ cả lượt.
        if not model_text:
            continue
        user_ids = _chat_segment("user", user_text, special, tokenizer)
        model_ids = _chat_segment("model", model_text, special, tokenizer)
        all_tokens += user_ids + model_ids
        mask += [0] * len(user_ids) + [1] * len(model_ids)

    if not all_tokens:
        return []

    input_ids = [special.bos] + all_tokens + [special.eos]
    if len(input_ids) > max_seq_len:
        return []

    target_ids = input_ids[1:]
    mask = mask + [1]       # EOS cuối cũng chịu loss
    assert len(mask) == len(target_ids), (
        f"Độ dài mask ({len(mask)}) khác target_ids ({len(target_ids)})"
    )
    return [(input_ids, target_ids, mask)]

ENCODERS.register("pretrain")(Encoder(read=read_text_jsonl, encode=encode_pretrain_text, has_mask=False))
ENCODERS.register("sft")(Encoder(read=read_sft_conversations, encode=encode_sft_conversation, has_mask=True))

def load_tokenizer(path: Path) -> spm.SentencePieceProcessor:
    return spm.SentencePieceProcessor(model_file=str(path))

def encode_file(kind: str, input_path: Path, output_path: Path, tokenizer_path: Path, max_seq_len: int) -> dict:
    """Encode một file thô thành một file ``.npz``; trả về số mẫu thô được giữ, số mẫu đã encode
    (một mẫu thô dài có thể thành nhiều mẫu), số token và số mẫu bị bỏ qua."""
    encoder = ENCODERS.get(kind)
    tokenizer = load_tokenizer(tokenizer_path)
    special = SpecialTokens.from_tokenizer(tokenizer)
    skipped = Counter()
    n_documents = 0

    def encoded_samples():
        nonlocal n_documents
        # write_npz duyệt hai lượt; đếm lại từ đầu mỗi lượt để bộ đếm phản ánh đúng một lượt.
        skipped.clear()
        n_documents = 0
        for sample in encoder.read(input_path, skipped):
            encoded = encoder.encode(sample, special, tokenizer, max_seq_len)
            if not encoded:
                skipped[SKIP_REJECTED] += 1
                continue
            n_documents += 1
            yield from encoded

    n_samples, n_tokens = write_npz(output_path, encoded_samples, encoder.has_mask)
    log_progress(f"{Path(output_path).name}: {n_documents:,} mẫu thô → {n_samples:,} mẫu | {n_tokens:,} token")
    for reason, count in sorted(skipped.items()):
        log_progress(f"  bỏ qua {count:,} {reason}")
    return {"n_documents": n_documents, "n_samples": n_samples, "n_tokens": n_tokens, "skipped": dict(skipped)}

def encode_shards(kind: str, input_paths: list[Path], output_dir: Path, manifest_name: str,
                  tokenizer_path: Path, max_seq_len: int) -> dict:
    """Encode mỗi file thô thành một shard ``<tên file>_ids.npz`` rồi ghi manifest liệt kê các shard."""
    output_dir = Path(output_dir)
    manifest_path = output_dir / manifest_name
    if manifest_path.exists():
        raise FileExistsError(f"{manifest_path} đã tồn tại; không ghi đè manifest")
    shards = []
    for input_path in sorted(Path(p) for p in input_paths):
        shard_file = f"{input_path.stem}{SHARD_SUFFIX}"
        stats = encode_file(kind, input_path, output_dir / shard_file, tokenizer_path, max_seq_len)
        shards.append({
            "name": input_path.stem,
            "file": shard_file,
            "n_samples": stats["n_samples"],
            "n_tokens": stats["n_tokens"],
        })
    manifest = write_manifest(manifest_path, shards)
    log_progress(f"{manifest_name}: {manifest['total_samples']:,} mẫu | {manifest['total_tokens']:,} token")
    return manifest
