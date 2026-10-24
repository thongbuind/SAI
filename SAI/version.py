"""Phiên bản của từng thành phần và của mô hình SAI.

Mỗi file thành phần trong ``SAI/`` có hằng ``VERSION`` dạng ``dev3.Y.Z``:

- Y: tăng 1 khi thay đổi checkpoint mô hình (đổi tham số/state_dict, đổi tokenizer); khi Y tăng thì Z về 0.
- Z: tăng 1 khi không cần train lại nhưng hành vi bản phát hành đổi (vd. kernel nhanh hơn cho cùng output, sửa generate làm output khác, tối ưu thay đổi lượng ram sử dụng).

Thay đổi không đổi hành vi (refactor, comment, docstring, model card, đóng gói HF) không tăng phiên bản.
``MODEL_VERSION`` là phiên bản phát hành chung, kể cả thay đổi adapter HF làm hành vi khác.
Thành phần được sửa nhận phiên bản phát hành mới; thành phần không đổi giữ phiên bản cũ.
Mọi thành phần phải cùng giai đoạn và không lớn hơn phiên bản phát hành. Kiểm tra được gọi
tường minh trước khi export, không phát sinh thao tác kiểm tra khi import.
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
MODEL_VERSION = "dev3.0.1"

def validate_versions() -> None:
    """Kiểm tra quan hệ giữa phiên bản phát hành và các thành phần."""
    release = parse_version(MODEL_VERSION)
    for name, version in COMPONENT_VERSIONS.items():
        component = parse_version(version)
        if component[0] != release[0] or component > release:
            raise ValueError(f"{name}: {version} không tương thích phiên bản phát hành {MODEL_VERSION}")
