"""Phiên bản của từng thành phần và của mô hình SAI.

Mỗi file thành phần trong ``SAI/`` có hằng ``VERSION`` dạng ``devX.Y.Z``:

- X: giai đoạn phát triển (dev3).
- Y: tăng 1 khi thay đổi bắt buộc train lại mô hình (đổi tham số/state_dict, hoặc đổi phép tính
  làm output khác); khi Y tăng thì Z về 0.
- Z: tăng 1 khi thay đổi không cần train lại (vd. kernel nhanh hơn cho cùng output, sửa generate).

Khi sửa một thành phần, phiên bản mới của nó = ``MODEL_VERSION`` hiện tại + bậc tương ứng, không
tính từ phiên bản cũ của riêng thành phần đó. Nhờ vậy thành phần vừa sửa luôn có phiên bản lớn nhất
và ``MODEL_VERSION`` (phiên bản lớn nhất trong các thành phần) bằng phiên bản của nó. Ví dụ:
dev3.0.0 → sửa kernel attention → attention dev3.0.1 → đổi FFN → ffn dev3.1.0 → sửa generate →
generate dev3.1.1; phiên bản mô hình lần lượt là dev3.0.1, dev3.1.0, dev3.1.1.
"""

import re
from collections.abc import Iterable
# Import từng VERSION theo tên, không ``from . import generate``: ``SAI/__init__.py`` đã gán tên
# ``SAI.generate`` thành hàm generate nên import theo module sẽ lấy nhầm hàm.
from .attention import VERSION as ATTENTION_VERSION
from .block import VERSION as BLOCK_VERSION
from .ffn import VERSION as FFN_VERSION
from .generate import VERSION as GENERATE_VERSION
from .modeling import VERSION as MODELING_VERSION
from .rope import VERSION as ROPE_VERSION

# tokenizer.model không chứa được hằng số nên phiên bản của nó ghi ở đây; đổi tokenizer luôn phải
# train lại mô hình (tăng Y).
TOKENIZER_VERSION = "dev3.0.0"

VERSION_PATTERN = re.compile(r"dev(\d+)\.(\d+)\.(\d+)")

def parse_version(version: str) -> tuple[int, int, int]:
    """``"dev3.1.2"`` → ``(3, 1, 2)``; so sánh tuple này để biết phiên bản nào mới hơn."""
    match = VERSION_PATTERN.fullmatch(version)
    if match is None:
        raise ValueError(f"phiên bản {version!r} không đúng dạng devX.Y.Z")
    return tuple(int(part) for part in match.groups())

def latest_version(versions: Iterable[str]) -> str:
    return max(versions, key=parse_version)

COMPONENT_VERSIONS = {
    "rope": ROPE_VERSION,
    "attention": ATTENTION_VERSION,
    "ffn": FFN_VERSION,
    "block": BLOCK_VERSION,
    "modeling": MODELING_VERSION,
    "generate": GENERATE_VERSION,
    "tokenizer": TOKENIZER_VERSION,
}
MODEL_VERSION = latest_version(COMPONENT_VERSIONS.values())
