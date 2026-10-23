"""Thuật toán chia văn bản dài, thử trên chuỗi piece giả (không cần tokenizer): token thứ i có chữ ``pieces[i]``."""

import math
import pytest
from foundation_lab.tokenizer.chunking import MIN_FILL_RATIO, OVERLAP_RATIO, OVERLAP_TOLERANCE, split_windows

MAX_LEN = 50

def _windows(pieces, max_len=MAX_LEN):
    return split_windows(list(range(len(pieces))), max_len, pieces.__getitem__)

def _sentences(count, words=6):
    """``count`` câu, mỗi câu ``words`` từ rồi dấu chấm: dấu chấm ở token 6, 13, 20, ..."""
    return ([" từ"] * words + ["."]) * count

def test_text_fitting_one_window_is_not_split():
    pieces = _sentences(2)
    assert _windows(pieces) == [(0, len(pieces) + 2)]

def test_windows_cover_every_target_with_bounded_length_and_overlap():
    pieces = []
    for index in range(30):
        pieces += _sentences(1, words=5 + index % 7) + (["\n"] if index % 4 == 3 else [])
    windows = _windows(pieces)
    total = len(pieces) + 2

    assert windows[0][0] == 0 and windows[-1][1] == total
    assert all(2 <= end - start <= MAX_LEN for start, end in windows)
    overlap_min = max(1, int((OVERLAP_RATIO - OVERLAP_TOLERANCE) * MAX_LEN))
    overlap_max = int((OVERLAP_RATIO + OVERLAP_TOLERANCE) * MAX_LEN)
    for (start, end), (next_start, next_end) in zip(windows, windows[1:]):
        assert end - start >= math.ceil(MIN_FILL_RATIO * MAX_LEN)
        assert start < next_start and end < next_end
        assert overlap_min <= end - next_start <= overlap_max
    # Mọi cặp (token, token kế tiếp) của s = [BOS] + tokens + [EOS] nằm trọn trong một cửa sổ.
    for position in range(total - 1):
        assert any(start <= position and position + 1 < end for start, end in windows)

def test_window_ends_at_paragraph_break_over_later_sentence_end():
    pieces = _sentences(15)
    # Vùng tìm điểm cuối là s[30..50]; không có hết đoạn thì cắt ở hết câu xa nhất: sau dấu chấm token 48.
    assert _windows(pieces)[0][1] == 50
    pieces[34] = "\n\n"     # token 34 nằm ở s[35]; cắt ngay sau nó
    assert _windows(pieces)[0][1] == 36

def test_list_with_intro_is_not_cut_when_another_boundary_exists():
    # Sau xuống dòng không có dấu cách nên token đầu dòng không mang khoảng trắng, như SentencePiece.
    intro = ["\n", "gồm", ":"]                      # token 28..30
    items = ["\n", "-", " mục", " một"] * 10        # token 31..70
    pieces = _sentences(4) + intro + items + ["\n", "hết"] + _sentences(5)
    # Trong vùng s[30..50] mọi chỗ khác đều nằm trong khối liệt kê; chỗ duy nhất ngoài khối là ngay
    # trước dòng dẫn (trước token 29, tức s[30]).
    assert _windows(pieces)[0][1] == 30

def test_list_longer_than_window_is_cut_only_between_items():
    pieces = [" -", " mục", " một"] + ["\n", "-", " mục", " một"] * 30
    windows = _windows(pieces)
    assert len(windows) > 2
    for start, end in windows[1:]:
        assert pieces[start - 1] in ("\n", "-")     # cửa sổ bắt đầu ở đầu một mục
    for start, end in windows[:-1]:
        assert pieces[end - 1] in ("\n", "-")       # token kế tiếp sau cửa sổ là đầu một mục

def test_text_without_spaces_falls_back_to_full_windows():
    windows = _windows(["a"] * 120)
    assert windows[0] == (0, MAX_LEN)
    assert windows[-1][1] == 122

def test_max_len_too_small_for_overlap_is_rejected():
    with pytest.raises(ValueError, match="quá nhỏ"):
        _windows(["a"] * 10, max_len=3)
