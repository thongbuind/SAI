"""Experiment spec: một file mô tả đầy đủ một lượt chạy (model + pipeline các stage + runtime).

Quy trình resolve một experiment:
1. Đọc file, gộp chuỗi ``extends``.
2. Thay tham chiếu dạng chuỗi bằng nội dung config: ``model: SAI_35M`` → ``config/models/SAI_35M.yaml``,
   ``data: sft1`` → ``config/data/sft1.yaml``.
3. Áp override ``--set khoa.con=gia_tri``.
4. Gộp ``stage_defaults`` vào ``params`` của từng stage (giá trị riêng của stage được ưu tiên).

Kết quả là một dict tự đủ (không còn tham chiếu), được ghi thành ``resolved_config.yaml``
trong thư mục run; chạy lại một run chỉ cần file này.
"""

import copy
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from foundation_lab.core.config import ConfigError, apply_overrides, deep_merge, from_dict, load_with_extends
from foundation_lab.paths import DATA_CONFIG_DIR, MODEL_CONFIG_DIR

@dataclass
class RuntimeSpec:
    """Cấu hình môi trường chạy; không làm đổi thuật toán huấn luyện."""
    device: Literal["auto", "cuda", "mps", "cpu"] = "auto"   # auto: cuda > mps > cpu như dev_2
    seed: int | None = None             # None: không đặt seed (giống dev_2); có giá trị: stage thứ i dùng seed + i
    compile: bool = True                # torch.compile(forward_features, dynamic=True)
    limited_max_autotune: bool = False
    dynamo_cache_size_limit: int = 24
    profile_kernels: bool = False
    profile_vram: bool = False
    num_workers: int = 1

@dataclass
class StageSpec:
    recipe: str
    data: dict
    params: dict = field(default_factory=dict)
    init: str | None = None             # None: khởi tạo mới; "<stage>.best": best của stage trước; còn lại: đường dẫn .pt
    enabled: bool = True

@dataclass
class ExperimentSpec:
    name: str
    model: dict
    pipeline: dict[str, StageSpec]
    runtime: RuntimeSpec = field(default_factory=RuntimeSpec)
    description: str = ""

_STAGE_REF = re.compile(r"^(?P<stage>[A-Za-z0-9_\-]+)\.best$")

def parse_init_ref(init: str | None, stage_names) -> tuple[str, str] | None:
    """Phân loại ``init``: ("stage", tên stage) hoặc ("path", đường dẫn); None nghĩa là khởi tạo mới."""
    if init is None:
        return None
    match = _STAGE_REF.match(init)
    if match and match.group("stage") in stage_names:
        return "stage", match.group("stage")
    return "path", init

def _resolve_ref(value, config_dir: Path, where: str):
    if isinstance(value, str):
        path = config_dir / f"{value}.yaml"
        if not path.exists():
            raise ConfigError(f"{where}: không tìm thấy {path}")
        return load_with_extends(path)
    if isinstance(value, dict):
        return copy.deepcopy(value)
    raise ConfigError(f"{where}: cần tên config hoặc mapping, nhận {value!r}")

def resolve_experiment_dict(data: dict, overrides=None) -> dict:
    data = copy.deepcopy(data)
    data["model"] = _resolve_ref(data.get("model"), MODEL_CONFIG_DIR, "model")
    pipeline = data.get("pipeline")
    if not isinstance(pipeline, dict) or not pipeline:
        raise ConfigError("pipeline: cần mapping tên stage → cấu hình stage, không được rỗng")
    for name, stage in pipeline.items():
        if not isinstance(stage, dict):
            raise ConfigError(f"pipeline.{name}: cần mapping")
        if "data" in stage:
            stage["data"] = _resolve_ref(stage["data"], DATA_CONFIG_DIR, f"pipeline.{name}.data")

    data = apply_overrides(data, overrides)
    defaults = data.pop("stage_defaults", None) or {}
    for stage in data["pipeline"].values():
        stage["params"] = deep_merge(defaults, stage.get("params") or {})
    return data

def resolve_experiment(path, overrides=None) -> dict:
    data = load_with_extends(path)
    data.setdefault("name", Path(path).stem)
    return resolve_experiment_dict(data, overrides)

def validate_experiment(resolved: dict) -> ExperimentSpec:
    """Kiểm tra toàn bộ experiment trước khi nạp dữ liệu hay dựng model.

    Gồm: schema experiment, config model theo họ mô hình, params theo recipe, config dữ liệu theo
    loại nguồn, tương thích recipe ↔ model (capability) và recipe ↔ dữ liệu, tham chiếu ``init``.
    """
    from foundation_lab import catalog

    spec = from_dict(ExperimentSpec, resolved, "experiment")
    family, _ = catalog.resolve_model_config(spec.model)
    earlier = []
    for name, stage in spec.pipeline.items():
        where = f"pipeline.{name}"
        recipe_cls = catalog.RECIPES.get(stage.recipe)
        from_dict(recipe_cls.config_cls, stage.params, f"{where}.params")
        source_cls, _ = catalog.resolve_data_config(stage.data, f"{where}.data")
        if source_cls.kind not in recipe_cls.data_kinds:
            raise ConfigError(
                f"{where}: recipe '{stage.recipe}' nhận dữ liệu loại {sorted(recipe_cls.data_kinds)}, "
                f"nhưng data có kind '{source_cls.kind}'"
            )
        missing = set(recipe_cls.requires) - set(family.capabilities)
        if missing:
            raise ConfigError(
                f"{where}: recipe '{stage.recipe}' cần capability {sorted(missing)} "
                f"mà họ mô hình '{spec.model.get('family')}' không có"
            )
        ref = parse_init_ref(stage.init, spec.pipeline)
        if ref is not None and ref[0] == "stage" and ref[1] not in earlier:
            raise ConfigError(f"{where}.init: '{stage.init}' phải tham chiếu một stage đứng trước")
        earlier.append(name)
    return spec
