"""Đọc và ghi dữ liệu đã tokenize.

Một file ``.npz`` chứa ``X_flat/X_offsets``, ``Y_flat/Y_offsets``, ``lengths`` (và ``M_flat/M_offsets``
cho SFT). Một ``manifest.json`` liệt kê nhiều shard ``.npz`` ghép thành một dataset logic.
"""

import json
from collections.abc import Callable, Iterable
from pathlib import Path
import numpy as np
from foundation_lab.core.log import log_progress

PROGRESS_EVERY = 100_000      # số mẫu giữa hai dòng log tiến độ khi ghi

def write_npz(path, encoded_samples: Callable[[], Iterable[tuple]], has_mask: bool) -> tuple[int, int]:
    """Ghi các mẫu đã encode ra một file ``.npz``; trả về (số mẫu, tổng độ dài X).

    Mỗi mẫu là ``(x, y)`` hoặc ``(x, y, mask)`` khi ``has_mask``. ``encoded_samples`` phải trả về
    iterator mới mỗi lần gọi: dữ liệu được duyệt hai lượt (lượt 1 đếm độ dài để cấp phát mảng đúng
    cỡ, lượt 2 điền giá trị) để không phải giữ toàn bộ mẫu trong RAM.
    """
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} đã tồn tại; không ghi đè dữ liệu đã encode")

    x_offsets = [0]
    y_offsets = [0]
    m_offsets = [0]
    lengths = []
    for idx, sample in enumerate(encoded_samples()):
        if idx % PROGRESS_EVERY == 0:
            log_progress(f"[lượt 1] {idx:,} mẫu...")
        x, y = sample[0], sample[1]
        x_offsets.append(x_offsets[-1] + len(x))
        y_offsets.append(y_offsets[-1] + len(y))
        if has_mask:
            m_offsets.append(m_offsets[-1] + len(sample[2]))
        lengths.append(len(x))

    n = len(lengths)
    if n == 0:
        raise ValueError(f"{path.name}: không có mẫu hợp lệ nào để ghi")

    x_flat = np.empty(x_offsets[-1], dtype=np.int32)
    y_flat = np.empty(y_offsets[-1], dtype=np.int32)
    m_flat = np.empty(m_offsets[-1], dtype=np.int8) if has_mask else None
    for idx, sample in enumerate(encoded_samples()):
        if idx % PROGRESS_EVERY == 0:
            log_progress(f"[lượt 2] {idx:,}/{n:,} mẫu...")
        x_flat[x_offsets[idx]:x_offsets[idx + 1]] = sample[0]
        y_flat[y_offsets[idx]:y_offsets[idx + 1]] = sample[1]
        if has_mask:
            m_flat[m_offsets[idx]:m_offsets[idx + 1]] = sample[2]

    len_arr = np.array(lengths, dtype=np.int32)
    arrays = {
        "X_flat": x_flat, "X_offsets": np.array(x_offsets, dtype=np.int64),
        "Y_flat": y_flat, "Y_offsets": np.array(y_offsets, dtype=np.int64),
        "lengths": len_arr,
    }
    if has_mask:
        arrays["M_flat"] = m_flat
        arrays["M_offsets"] = np.array(m_offsets, dtype=np.int64)

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return n, int(len_arr.sum())

def write_manifest(path, shards: list[dict]) -> dict:
    """Ghi manifest liệt kê các shard; mỗi shard là ``{name, file, n_samples, n_tokens}``, ``file``
    tương đối theo thư mục của manifest."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} đã tồn tại; không ghi đè manifest")
    manifest = {
        "shards": shards,
        "total_samples": sum(shard["n_samples"] for shard in shards),
        "total_tokens": sum(shard["n_tokens"] for shard in shards),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return manifest

def _load_single_npz(path, load_mask=False):
    f = np.load(path)
    def reconstruct(name):
        flat, offsets = f[f"{name}_flat"], f[f"{name}_offsets"]
        return np.array([flat[offsets[i]:offsets[i+1]] for i in range(len(offsets)-1)], dtype=object)
    X = reconstruct("X")
    Y = reconstruct("Y")
    lengths = f["lengths"]
    if load_mask:
        M = reconstruct("M")
        f.close()
        return X, Y, M, lengths
    f.close()
    return X, Y, lengths

def _load_manifest_shards(manifest_path, load_mask=False):
    """Đọc manifest.json rồi ghép nhiều shard .npz lại thành 1 dataset logic."""
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    manifest_dir = Path(manifest_path).parent
    shards = manifest["shards"]

    X_list, Y_list, M_list, L_list = [], [], [], []
    for shard in shards:
        shard_path = manifest_dir / shard["file"]
        if load_mask:
            X_s, Y_s, M_s, L_s = _load_single_npz(shard_path, load_mask=True)
            M_list.append(M_s)
        else:
            X_s, Y_s, L_s = _load_single_npz(shard_path, load_mask=False)
        X_list.append(X_s)
        Y_list.append(Y_s)
        L_list.append(L_s)

    X = np.concatenate(X_list)
    Y = np.concatenate(Y_list)
    lengths = np.concatenate(L_list)

    if load_mask:
        M = np.concatenate(M_list)
        return X, Y, M, lengths
    return X, Y, lengths

def load_npz(path, load_mask=False):
    path = Path(path)
    if path.suffix == ".json":
        return _load_manifest_shards(path, load_mask=load_mask)
    return _load_single_npz(path, load_mask=load_mask)

def manifest_shard_paths(path) -> list[Path]:
    """Danh sách file thật sự được đọc khi load ``path`` (bản thân file, hoặc các shard của manifest)."""
    path = Path(path)
    if path.suffix != ".json" or not path.exists():
        return [path]
    with open(path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    return [path.parent / shard["file"] for shard in manifest["shards"]]
