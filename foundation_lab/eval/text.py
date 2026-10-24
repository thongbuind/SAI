"""Thước đo trên văn bản sinh ra, không cần đáp án tham chiếu."""

def distinct_n(texts: list[str], n: int) -> float:
    """Tỷ lệ n-gram (theo từ, tách bằng khoảng trắng) khác nhau trên tổng số n-gram của mọi văn bản.

    Gần 1: ít lặp; gần 0: lặp nhiều. Không có n-gram nào (mọi văn bản ngắn hơn n từ) trả về 0.
    """
    if n < 1:
        raise ValueError(f"n phải >= 1, nhận {n}")
    ngrams = []
    for text in texts:
        words = text.split()
        ngrams.extend(tuple(words[i:i + n]) for i in range(len(words) - n + 1))
    if not ngrams:
        return 0.0
    return len(set(ngrams)) / len(ngrams)
