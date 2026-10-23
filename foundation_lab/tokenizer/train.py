"""Train tokenizer SentencePiece của SAI từ văn bản pretrain thô (JSONL ``{"text": ...}``)."""

import tempfile
from collections import Counter
from pathlib import Path
import sentencepiece as spm
from foundation_lab.core.log import log_progress
from foundation_lab.tokenizer.encode import read_text_jsonl

OFFICIAL_VOCAB_SIZE = 10_000
MODEL_PREFIX = "tokenizer"            # SentencePiece ghi ra <prefix>.model và <prefix>.vocab
TRAINER_INPUT_FILE = "spm_input.txt"
# Tham số đã dùng để train SAI/tokenizer.model (ngoài input, model_prefix, vocab_size).
# pad_piece không có tác dụng: pad_id mặc định của SentencePiece là -1 nên tokenizer không có [PAD];
# id 0..4 là [UNK], [BOS], [EOS], <|im_start|>, <|im_end|>.
TRAINER_ARGS = {
    "model_type": "unigram",
    "character_coverage": 0.9999,
    "pad_piece": "[PAD]",
    "unk_piece": "[UNK]",
    "bos_piece": "[BOS]",
    "eos_piece": "[EOS]",
    "user_defined_symbols": ["<|im_start|>", "<|im_end|>"],
    "normalization_rule_name": "identity",
    "treat_whitespace_as_suffix": False,
    "input_sentence_size": 1_000_000,
    "shuffle_input_sentence": True,
    "remove_extra_whitespaces": False,
    "split_digits": True,
    "byte_fallback": True,
    "max_sentencepiece_length": 8,
}

def train_tokenizer(input_path: Path, output_dir: Path, vocab_size: int = OFFICIAL_VOCAB_SIZE) -> Path:
    """Train tokenizer mới vào ``output_dir`` (phải chưa tồn tại) và trả về đường dẫn ``tokenizer.model``.

    Văn bản được viết thường như lúc encode. Tokenizer chính thức ``SAI/tokenizer.model`` không bao
    giờ bị ghi đè ở đây; thay nó là việc làm tay sau khi đã quyết định train lại mô hình.
    """
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"{output_dir} đã tồn tại; tokenizer mới phải ghi vào thư mục mới")

    skipped = Counter()
    with tempfile.TemporaryDirectory() as tmp:
        text_path = Path(tmp) / TRAINER_INPUT_FILE
        n_lines = 0
        with open(text_path, "w", encoding="utf-8") as f:
            for text in read_text_jsonl(Path(input_path), skipped):
                f.write(text.lower() + "\n")
                n_lines += 1
        if n_lines == 0:
            raise ValueError(f"{input_path}: không có dòng văn bản hợp lệ nào để train tokenizer")
        log_progress(f"{n_lines:,} dòng văn bản đưa vào train")
        for reason, count in sorted(skipped.items()):
            log_progress(f"  bỏ qua {count:,} {reason}")

        output_dir.mkdir(parents=True)
        spm.SentencePieceTrainer.train(
            input=str(text_path),
            model_prefix=str(output_dir / MODEL_PREFIX),
            vocab_size=vocab_size,
            **TRAINER_ARGS,
        )
    return output_dir / f"{MODEL_PREFIX}.model"
