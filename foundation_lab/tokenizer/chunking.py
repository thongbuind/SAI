"""Chia văn bản pretrain dài hơn ``max_seq_len`` thành các cửa sổ chồng lấn, cắt ở ranh giới tự nhiên.

Văn bản được tokenize một lần cho cả bài (nên văn bản ngắn ra đúng token như khi không chia). Chuỗi được
chia là ``s = [BOS] + tokens + [EOS]``; mỗi cửa sổ là một đoạn ``s[start:end]`` với
``end - start <= max_len`` và thành một mẫu ``x = s[start:end - 1]``, ``y = s[start + 1:end]``.

Thuật toán (tham lam, đi từ đầu tới cuối văn bản):
1. Chấm điểm mọi vị trí cắt giữa hai token theo ranh giới chữ tại đó (``Boundary``): hết đoạn văn >
   xuống dòng > hết câu > hết vế (``; : ,``) > đầu một mục liệt kê > giữa hai từ > giữa một từ.
   Mọi vị trí nằm trong một khối liệt kê (kể cả dòng dẫn kết thúc bằng ``:``) bị hạ xuống "giữa hai
   từ", trừ chỗ bắt đầu một mục; vì vậy khối liệt kê chỉ bị cắt khi vùng tìm không còn chỗ nào khác,
   và khi buộc phải cắt thì cắt giữa hai mục.
2. Cuối cửa sổ: vị trí điểm cao nhất trong ``[start + MIN_FILL_RATIO·L, start + L]``; bằng điểm thì
   chọn vị trí xa nhất (cửa sổ dài nhất).
3. Đầu cửa sổ sau: vị trí điểm cao nhất trong vùng chồng lấn, cách cuối cửa sổ trước từ
   ``(OVERLAP_RATIO - OVERLAP_TOLERANCE)·L`` tới ``(OVERLAP_RATIO + OVERLAP_TOLERANCE)·L`` token;
   bằng điểm thì chọn vị trí có độ chồng lấn gần ``OVERLAP_RATIO·L`` nhất.
4. Lặp tới khi phần còn lại vừa một cửa sổ.

Hai cửa sổ liền nhau luôn chồng lấn ít nhất một token, nên mọi cặp (token, token kế tiếp) của ``s``
nằm trọn trong ít nhất một cửa sổ: không token đích nào bị mất. Token trong vùng chồng lấn được học
hai lần, với hai ngữ cảnh khác nhau.
"""

import math
import re
from collections.abc import Callable
from enum import IntEnum

OVERLAP_RATIO = 0.2          # độ chồng lấn mục tiêu giữa hai cửa sổ liền nhau, tính theo max_len
OVERLAP_TOLERANCE = 0.1      # độ chồng lấn được lệch khỏi mục tiêu để tìm chỗ bắt đầu đẹp hơn
MIN_FILL_RATIO = 0.6         # cửa sổ (trừ cửa sổ cuối) dài ít nhất chừng này phần max_len

class Boundary(IntEnum):
    """Điểm của một vị trí cắt; càng cao càng nên cắt ở đó."""
    INSIDE_WORD = 0
    WORD = 1
    LIST_ITEM = 2
    CLAUSE = 3
    SENTENCE = 4
    LINE = 5
    PARAGRAPH = 6

PARAGRAPH_BREAK = re.compile(r"\n[ \t]*\n[ \t\n]*")
LINE_BREAK = re.compile(r"\n[ \t]*")
SENTENCE_END = re.compile(r"[.!?…]+[\"'”’»)\]]*(?=\s)")
CLAUSE_END = re.compile(r"[;:,](?=\s)")
# Dòng mở đầu bằng dấu gạch/chấm tròn, "1." "1)" hoặc "a." "a)" rồi khoảng trắng và nội dung.
LIST_ITEM_LINE = re.compile(r"[ \t]*(?:[-*+•–—]|\d{1,3}[.)]|[a-zđ][.)])[ \t]+\S")
LIST_INTRO_SUFFIX = ":"

def split_windows(tokens: list[int], max_len: int, token_text: Callable[[int], str]) -> list[tuple[int, int]]:
    """Các cửa sổ ``[start, end)`` trên ``s = [BOS] + tokens + [EOS]`` (độ dài ``len(tokens) + 2``).

    ``token_text(token_id)``: phần chữ token chiếm trong văn bản; chỉ được gọi khi cần chia.
    """
    total = len(tokens) + 2
    if total <= max_len:
        return [(0, total)]

    min_fill = math.ceil(MIN_FILL_RATIO * max_len)
    overlap_min = max(1, int((OVERLAP_RATIO - OVERLAP_TOLERANCE) * max_len))
    overlap_max = int((OVERLAP_RATIO + OVERLAP_TOLERANCE) * max_len)
    overlap_target = round(OVERLAP_RATIO * max_len)
    # min_fill > overlap_max bảo đảm cửa sổ sau luôn bắt đầu muộn hơn cửa sổ trước (thuật toán dừng).
    if not min_fill > overlap_max >= overlap_min:
        raise ValueError(f"max_len={max_len} quá nhỏ để chia cửa sổ chồng lấn")

    levels = _cut_levels([token_text(token_id) for token_id in tokens])
    windows = []
    start = 0
    while total - start > max_len:
        end = _best_cut(levels, start + min_fill, start + max_len, prefer=start + max_len)
        windows.append((start, end))
        start = _best_cut(levels, end - overlap_max, end - overlap_min, prefer=end - overlap_target)
    windows.append((start, total))
    return windows

def _best_cut(levels: list[Boundary], lo: int, hi: int, prefer: int) -> int:
    """Vị trí trong ``[lo, hi]`` có điểm cao nhất; bằng điểm thì gần ``prefer`` nhất."""
    return max(range(lo, hi + 1), key=lambda p: (levels[p], -abs(p - prefer)))

def _cut_levels(texts: list[str]) -> list[Boundary]:
    """Điểm của vị trí cắt ngay trước ``s[p]``, với ``p`` từ 0 tới ``len(texts) + 2``."""
    starts = []
    offset = 0
    for piece in texts:
        starts.append(offset)
        offset += len(piece)
    char_levels = _char_levels("".join(texts))
    # p = 0 (trước BOS) không bao giờ được chọn; p = len(texts) + 1 là trước EOS, tức hết văn bản.
    return [Boundary.INSIDE_WORD] + [char_levels[c] for c in starts] + [Boundary.PARAGRAPH, Boundary.PARAGRAPH]

def _char_levels(text: str) -> list[Boundary]:
    """Điểm của vị trí cắt ngay trước ``text[c]``, với ``c`` từ 0 tới ``len(text)``."""
    levels = [Boundary.INSIDE_WORD] * (len(text) + 1)
    for c in range(len(text) + 1):
        if (c > 0 and text[c - 1].isspace()) or (c < len(text) and text[c].isspace()):
            levels[c] = Boundary.WORD
    # Sau dấu câu: cắt được ở bất kỳ đâu trong khoảng trắng theo sau (khoảng trắng thường dính vào token kế).
    for pattern, level in ((CLAUSE_END, Boundary.CLAUSE), (SENTENCE_END, Boundary.SENTENCE)):
        for match in pattern.finditer(text):
            _raise_levels(levels, match.end(), _whitespace_end(text, match.end()), level)
    # Xuống dòng: cắt được ở bất kỳ đâu sau ký tự xuống dòng đầu tiên của cụm.
    for pattern, level in ((LINE_BREAK, Boundary.LINE), (PARAGRAPH_BREAK, Boundary.PARAGRAPH)):
        for match in pattern.finditer(text):
            _raise_levels(levels, match.start() + 1, match.end(), level)

    for block_start, block_end, item_starts in _list_blocks(text):
        for c in range(block_start + 1, block_end):
            levels[c] = min(levels[c], Boundary.WORD)
        for item_start in item_starts:
            # Cắt ngay trước ký tự xuống dòng của mục, hoặc trước phần thụt lề/ký hiệu của mục.
            indent = len(text[item_start:]) - len(text[item_start:].lstrip(" \t"))
            for c in range(item_start - 1, item_start + indent + 1):
                levels[c] = Boundary.LIST_ITEM
    return levels

def _raise_levels(levels: list[Boundary], lo: int, hi: int, level: Boundary):
    for c in range(lo, hi + 1):
        levels[c] = max(levels[c], level)

def _whitespace_end(text: str, position: int) -> int:
    while position < len(text) and text[position].isspace():
        position += 1
    return position

def _list_blocks(text: str) -> list[tuple[int, int, list[int]]]:
    """Các khối liệt kê ``(đầu khối, cuối khối, vị trí đầu các mục trừ mục đầu tiên)`` theo offset ký tự.

    Khối gồm các dòng mục liên tiếp, dòng trống xen giữa hai mục, dòng thụt lề nối tiếp một mục, và
    dòng dẫn ngay trước mục đầu nếu dòng đó kết thúc bằng ``:``.
    """
    lines = []
    offset = 0
    for line in text.split("\n"):
        lines.append((offset, line))
        offset += len(line) + 1

    blocks = []
    index = 0
    while index < len(lines):
        if not LIST_ITEM_LINE.match(lines[index][1]):
            index += 1
            continue
        first = last = index
        item_starts = []
        for following in range(index + 1, len(lines)):
            line_start, line = lines[following]
            if LIST_ITEM_LINE.match(line):
                item_starts.append(line_start)
                last = following
            elif line.strip() and line[0] in " \t":
                last = following
            elif line.strip():
                break
        has_intro = first > 0 and lines[first - 1][1].rstrip().endswith(LIST_INTRO_SUFFIX)
        block_start = lines[first - 1][0] if has_intro else lines[first][0]
        block_end = lines[last][0] + len(lines[last][1])
        blocks.append((block_start, block_end, item_starts))
        index = last + 1
    return blocks
