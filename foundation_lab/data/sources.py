"""Nguồn dữ liệu đã tokenize: đọc file, trộn với nguồn phụ, split train/val/test.

Hai loại (``kind``):
- ``lm``: chuỗi token cho pretrain/continued pretrain (không có loss mask).
- ``sft``: hội thoại có loss mask.

Trộn với nguồn phụ (``sub``): lấy ngẫu nhiên ``mix_ratio × len(main)`` mẫu từ ``sub`` trên TOÀN BỘ
nguồn phụ, ghép vào ``main`` rồi xáo trộn, sau đó mới split. ``data/configs/continued_mix.yaml`` dùng
``mix_ratio=3``, ``data/configs/sft2_mix.yaml`` dùng ``mix_ratio=1``. Có lỗi đã biết: mẫu test của
nguồn phụ có thể vào train của nguồn chính.
"""

import gc
from dataclasses import dataclass
import numpy as np
from foundation_lab.core.config import ConfigError, from_dict
from foundation_lab.core.registry import Registry
from foundation_lab.data.io import load_npz, manifest_shard_paths
from foundation_lab.data.splits import split_train_val_test
from foundation_lab.paths import resolve_project_path

DATA_SOURCES = Registry("data_source")

@dataclass
class TokenizedSourceConfig:
    main: str                       # .npz hoặc manifest .json; tương đối theo thư mục gốc dự án
    sub: str | None = None          # nguồn phụ để trộn vào (tuỳ chọn)
    mix_ratio: float = 1.0          # số mẫu lấy từ sub = mix_ratio × số mẫu main
    mix_seed: int = 54
    train_ratio: float = 0.95
    val_ratio: float = 0.04
    split_seed: int = 54

@dataclass
class SplitArrays:
    X: np.ndarray
    Y: np.ndarray
    loss_masks: np.ndarray | None
    lengths: np.ndarray

    def __len__(self):
        return len(self.X)

def _mix_with_sub(main, sub, mix_ratio, seed):
    rng = np.random.default_rng(seed)
    shuffled_indices = rng.permutation(len(sub[0]))

    n_samples = min(int(mix_ratio * len(main[0])), len(sub[0]))
    selected_indices = shuffled_indices[:n_samples]
    sampled = [array[selected_indices] for array in sub]

    del sub, shuffled_indices, selected_indices
    gc.collect()

    combined = [np.concatenate([m, s]) for m, s in zip(main, sampled)]
    combined_indices = rng.permutation(len(combined[0]))
    return [array[combined_indices] for array in combined]

class TokenizedSource:
    kind = ""
    with_mask = False
    config_cls = TokenizedSourceConfig

    def __init__(self, config: TokenizedSourceConfig):
        self.config = config

    def files(self):
        paths = [resolve_project_path(self.config.main)]
        if self.config.sub is not None:
            paths.append(resolve_project_path(self.config.sub))
        return paths

    def read_files(self):
        """Mọi file thật sự được đọc (gồm các shard của manifest) — dùng cho lineage."""
        return [shard for path in self.files() for shard in manifest_shard_paths(path)]

    def load(self, mix_seed=None):
        """Trả về (X, Y, loss_masks hoặc None, lengths) sau khi trộn. ``mix_seed=None`` dùng ``config.mix_seed``."""
        cfg = self.config
        main = list(load_npz(resolve_project_path(cfg.main), load_mask=self.with_mask))
        if cfg.sub is not None:
            sub = list(load_npz(resolve_project_path(cfg.sub), load_mask=self.with_mask))
            seed = cfg.mix_seed if mix_seed is None else mix_seed
            main = _mix_with_sub(main, sub, cfg.mix_ratio, seed)
            del sub
        if self.with_mask:
            X, Y, M, L = main
        else:
            (X, Y, L), M = main, None
        return X, Y, M, L

    def load_splits(self, mix_seed=None) -> dict[str, SplitArrays]:
        cfg = self.config
        X, Y, M, L = self.load(mix_seed)
        parts = split_train_val_test(X, Y, M, L, cfg.train_ratio, cfg.val_ratio, seed=cfg.split_seed)
        return {
            name: SplitArrays(*parts[i * 4:(i + 1) * 4])
            for i, name in enumerate(("train", "val", "test"))
        }

@DATA_SOURCES.register("lm")
class LMSource(TokenizedSource):
    kind = "lm"
    with_mask = False

@DATA_SOURCES.register("sft")
class SFTSource(TokenizedSource):
    kind = "sft"
    with_mask = True

def resolve_data_config(data: dict, where: str = "data"):
    """Trả về (class nguồn dữ liệu, config đã kiểm tra) từ mapping có khoá ``kind``."""
    if not isinstance(data, dict) or "kind" not in data:
        raise ConfigError(f"{where}: cần mapping có khoá 'kind'")
    source_cls = DATA_SOURCES.get(data["kind"])
    fields = {key: value for key, value in data.items() if key != "kind"}
    return source_cls, from_dict(source_cls.config_cls, fields, where)

def build_data_source(data: dict, where: str = "data"):
    source_cls, config = resolve_data_config(data, where)
    return source_cls(config)
