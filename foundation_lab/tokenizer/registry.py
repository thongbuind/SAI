from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from foundation_lab.core.registry import Registry

@dataclass(frozen=True)
class Encoder:
    """Một kiểu encode: đọc file thô thành các mẫu thô, rồi encode từng mẫu thành chuỗi id."""
    # (đường dẫn, bộ đếm mẫu bị bỏ qua) -> các mẫu thô; mẫu hỏng được đếm vào bộ đếm thay vì báo lỗi.
    read: Callable[[Path, Counter], Iterator]
    # (mẫu thô, SpecialTokens, tokenizer, max_seq_len) -> danh sách mẫu (x, y) hoặc (x, y, mask);
    # rỗng nếu mẫu bị loại, nhiều phần tử nếu một mẫu thô được chia thành nhiều mẫu.
    encode: Callable
    has_mask: bool

# Kiểu encode: Encoder.
ENCODERS = Registry("encoder")
