"""Phòng Tokenizer: train tokenizer SentencePiece và encode văn bản thô thành chuỗi id.

Tokenizer chính thức (thứ được phát hành cùng model) là ``SAI/tokenizer.model``; phòng này chỉ đọc
nó, không ghi đè. Kiểu encode (pretrain, SFT...) đăng ký vào ``ENCODERS``; file ``.npz`` đầu ra do
``foundation_lab.data.io`` ghi để định dạng chỉ được định nghĩa ở một chỗ.
"""

from foundation_lab.tokenizer.encode import SpecialTokens, encode_file, encode_shards, load_tokenizer
from foundation_lab.tokenizer.registry import ENCODERS, Encoder
from foundation_lab.tokenizer.train import OFFICIAL_VOCAB_SIZE, train_tokenizer

__all__ = [
    "ENCODERS", "Encoder", "OFFICIAL_VOCAB_SIZE", "SpecialTokens",
    "encode_file", "encode_shards", "load_tokenizer", "train_tokenizer",
]
