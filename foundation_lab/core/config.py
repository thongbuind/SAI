import copy
import dataclasses
import hashlib
import json
import types
import typing
from pathlib import Path
import yaml

class ConfigError(ValueError):
    """Config sai khoá, thiếu khoá bắt buộc hoặc sai kiểu."""

def read_config_file(path) -> dict:
    """Đọc một file YAML hoặc JSON; gốc của file phải là mapping."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f) if path.suffix == ".json" else yaml.safe_load(f)
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: gốc của config phải là mapping")
    return data

def deep_merge(base: dict, override: dict) -> dict:
    """Gộp hai mapping; mapping lồng nhau được gộp đệ quy, giá trị khác bị ghi đè."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged

def load_with_extends(path) -> dict:
    """Đọc config có khoá ``extends`` (đường dẫn tương đối theo file hiện tại) và gộp với file cha."""
    path = Path(path).resolve()
    data = read_config_file(path)
    parent = data.pop("extends", None)
    if parent is None:
        return data
    return deep_merge(load_with_extends(path.parent / parent), data)

def apply_overrides(data: dict, overrides) -> dict:
    """Áp các override dạng ``a.b.c=value``; value được đọc theo cú pháp YAML."""
    data = copy.deepcopy(data)
    for text in overrides or []:
        if "=" not in text:
            raise ConfigError(f"Override '{text}' phải có dạng khoa.con=gia_tri")
        key, raw = text.split("=", 1)
        keys = key.strip().split(".")
        node = data
        for part in keys[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise ConfigError(f"Override '{text}': '{part}' không phải mapping")
        node[keys[-1]] = yaml.safe_load(raw)
    return data

def config_hash(data: dict) -> str:
    """Hash ổn định của một config đã resolve (không phụ thuộc thứ tự khoá)."""
    payload = json.dumps(data, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

def from_dict(cls, data, where: str = ""):
    """Dựng dataclass ``cls`` từ mapping một cách nghiêm ngặt.

    - Khoá lạ → ConfigError (tránh gõ sai tên khoá mà không ai biết).
    - Thiếu khoá bắt buộc → ConfigError.
    - Sai kiểu → ConfigError. ``int`` được chấp nhận cho ``float``; ``bool`` không được coi là số.
    """
    label = where or cls.__name__
    if not isinstance(data, dict):
        raise ConfigError(f"{label}: cần mapping, nhận {type(data).__name__}")
    hints = typing.get_type_hints(cls)
    fields = {f.name: f for f in dataclasses.fields(cls) if f.init}
    unknown = sorted(set(data) - set(fields))
    if unknown:
        raise ConfigError(f"{label}: khoá không hợp lệ {unknown}; khoá hợp lệ: {sorted(fields)}")
    kwargs = {}
    for name, field in fields.items():
        path = f"{where}.{name}" if where else name
        if name in data:
            kwargs[name] = _coerce(hints[name], data[name], path)
        elif field.default is dataclasses.MISSING and field.default_factory is dataclasses.MISSING:
            raise ConfigError(f"{path}: thiếu khoá bắt buộc")
    return cls(**kwargs)

def _type_error(path, tp, value):
    return ConfigError(f"{path}: cần kiểu {getattr(tp, '__name__', tp)}, nhận {value!r}")

def _coerce(tp, value, path):
    origin = typing.get_origin(tp)
    args = typing.get_args(tp)
    if tp is typing.Any:
        return value
    if origin in (typing.Union, types.UnionType):
        if value is None:
            if type(None) in args:
                return None
            raise _type_error(path, tp, value)
        for arg in args:
            if arg is type(None):
                continue
            try:
                return _coerce(arg, value, path)
            except ConfigError:
                continue
        raise _type_error(path, tp, value)
    if origin is typing.Literal:
        if value in args:
            return value
        raise ConfigError(f"{path}: phải là một trong {list(args)}, nhận {value!r}")
    if dataclasses.is_dataclass(tp):
        return from_dict(tp, value, path)
    if origin is list or tp is list:
        if not isinstance(value, list):
            raise _type_error(path, tp, value)
        item_tp = args[0] if args else typing.Any
        return [_coerce(item_tp, item, f"{path}[{i}]") for i, item in enumerate(value)]
    if origin is dict or tp is dict:
        if not isinstance(value, dict):
            raise _type_error(path, tp, value)
        value_tp = args[1] if args else typing.Any
        return {key: _coerce(value_tp, item, f"{path}.{key}") for key, item in value.items()}
    if tp is bool:
        if not isinstance(value, bool):
            raise _type_error(path, tp, value)
        return value
    if tp is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise _type_error(path, tp, value)
        return value
    if tp is float:
        # PyYAML đọc "2e-4" (không có dấu chấm) thành chuỗi, nên chấp nhận chuỗi số thực.
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                raise _type_error(path, tp, value) from None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise _type_error(path, tp, value)
        return float(value)
    if tp is str:
        if not isinstance(value, str):
            raise _type_error(path, tp, value)
        return value
    raise ConfigError(f"{path}: kiểu {tp} chưa được loader hỗ trợ")

def to_dict(obj) -> dict:
    return dataclasses.asdict(obj)
