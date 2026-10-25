"""Xuất dữ liệu cho web tĩnh (web/site) từ code thật, để số liệu trên web luôn khớp với repo.

Chạy từ thư mục gốc dự án mỗi khi config mô hình, file phát hành, phiên bản thành phần hoặc registry
của lab thay đổi; ``tests/test_site_data.py`` báo lỗi nếu quên chạy:

    python -m web.scripts.export_site_data          # ghi web/site/src/data/*.json
    python -m web.scripts.export_site_data --check  # chỉ so sánh, thoát mã 1 nếu JSON đã lệch code

Script chỉ chạy lúc build nên được phép đọc ``foundation_lab`` (registry thật chỉ có ở đó);
backend web lúc chạy vẫn không phụ thuộc lab.
"""

import argparse
import dataclasses
import json
import re
import sys
from pathlib import Path
import torch
import foundation_lab
from foundation_lab.catalog import DEPARTMENTS
from foundation_lab.core.config import load_with_extends
from SAI.generate import GenerationConfig
from SAI.modeling import SAIConfig, SAIModel
from SAI.version import COMPONENT_VERSIONS, MODEL_VERSION

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "web" / "site" / "src" / "data"
MODEL_CONFIG_DIR = PROJECT_ROOT / "config" / "models"
RELEASE_DIR = PROJECT_ROOT / "hf_release" / "releases"
# File config hiển thị nguyên văn trên trang Lab: khoá trong snippets.json → đường dẫn từ gốc dự án.
SNIPPET_FILES = {
    "experiment_35m": "foundation_lab/core/configs/SAI_35M.yaml",
    "modular_35m": "foundation_lab/architecture/configs/SAI_modular_35M.yaml",
}
MODEL_FIELDS = ("vocab_size", "max_seq_len", "d_model", "num_heads", "num_kv_heads", "num_layers", "ff_dim")
# Tham số sinh hiện trên trang /try/ (giá trị mặc định lấy từ GenerationConfig), theo thứ tự hiển thị.
GENERATION_FIELDS_SHOWN = (
    "strategy", "max_new_tokens", "beam_size", "stable_steps", "length_penalty", "repetition_penalty",
    "no_repeat_ngram_size", "temperature", "top_k", "top_p", "lowercase",
)
HUGGING_FACE_BASE = "https://huggingface.co/"
# Dòng mô tả phòng ban trong docstring foundation_lab: "- ``mã``   Tên phòng: trách nhiệm".
DEPARTMENT_LINE = re.compile(r"^- ``(?P<code>\w+)``\s+(?P<name>[^:]+):\s*(?P<text>.+)$", re.MULTILINE)

def count_parameters(config: dict) -> int:
    """Đếm tham số của model chính thức mà không cấp bộ nhớ: dựng trên meta device, không có trọng số thật."""
    fields = {field.name: config[field.name] for field in dataclasses.fields(SAIConfig) if field.name in config}
    with torch.device("meta"):
        model = SAIModel(SAIConfig(**fields))
    # parameters() chỉ trả mỗi tensor một lần, nên embedding và lm head dùng chung (weight tying) không bị đếm đôi.
    return sum(parameter.numel() for parameter in model.parameters())

def export_models() -> list[dict]:
    """Mô hình đã có file phát hành trong hf_release/releases, kèm kiến trúc từ config/models; nhỏ trước."""
    models = []
    for release_path in sorted(RELEASE_DIR.glob("*.yaml")):
        release = load_with_extends(release_path)
        name = release["source"]["model"]
        config = load_with_extends(MODEL_CONFIG_DIR / f"{name}.yaml")
        if config.get("family") != "SAI":
            raise ValueError(f"{release_path.name}: chỉ phát hành model chính thức (family: SAI), nhận {config.get('family')!r}")
        models.append({
            "name": name,
            "label": name.removeprefix("SAI_"),
            "repo_id": release["repo_id"],
            "url": HUGGING_FACE_BASE + release["repo_id"],
            "license": release["card"]["license"],
            "parameters": count_parameters(config),
            **{field: config[field] for field in MODEL_FIELDS},
        })
    return sorted(models, key=lambda model: model["d_model"])

def export_version() -> dict:
    return {"model": MODEL_VERSION, "components": dict(COMPONENT_VERSIONS)}

def export_lab() -> dict:
    """Phòng ban theo thứ tự trong docstring lab; registry (tên đã đăng ký) lấy từ catalog.DEPARTMENTS."""
    documented = [match.groupdict() for match in DEPARTMENT_LINE.finditer(foundation_lab.__doc__)]
    codes = [department["code"] for department in documented]
    undocumented = sorted(set(DEPARTMENTS) - set(codes))
    if undocumented:
        raise ValueError(f"Phòng có registry nhưng thiếu trong docstring foundation_lab: {undocumented}")
    departments = []
    for department in documented:
        registries = DEPARTMENTS.get(department["code"], [])
        departments.append({
            "code": department["code"],
            "name": department["name"].strip(),
            "description": department["text"].strip(),
            "registries": [{"kind": registry.kind, "names": registry.names()} for registry in registries],
        })
    # Thứ tự in của "python -m foundation_lab list" (theo catalog), để khối lệnh trên web trùng output thật.
    return {"departments": departments, "list_order": list(DEPARTMENTS)}

def export_snippets() -> dict:
    return {
        key: {"path": path, "text": (PROJECT_ROOT / path).read_text(encoding="utf-8")}
        for key, path in SNIPPET_FILES.items()
    }

def export_generation() -> list[dict]:
    """Giá trị mặc định của GenerationConfig cho các tham số hiện trên trang /try/."""
    defaults = {field.name: field.default for field in dataclasses.fields(GenerationConfig)}
    missing = [name for name in GENERATION_FIELDS_SHOWN if name not in defaults]
    if missing:
        raise ValueError(f"GenerationConfig không còn tham số {missing}; cập nhật GENERATION_FIELDS_SHOWN")
    return [{"name": name, "default": defaults[name]} for name in GENERATION_FIELDS_SHOWN]

def build_site_data() -> dict[str, object]:
    """Tên file JSON (không đuôi) → dữ liệu."""
    return {
        "models": export_models(),
        "generation": export_generation(),
        "version": export_version(),
        "lab": export_lab(),
        "snippets": export_snippets(),
    }

def serialize(data) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2) + "\n"

def main(argv=None):
    parser = argparse.ArgumentParser(description="Xuất dữ liệu cho web/site từ code của repo")
    parser.add_argument("--check", action="store_true", help="chỉ so sánh với file đã có, không ghi")
    args = parser.parse_args(argv)

    outdated = []
    for name, data in build_site_data().items():
        path = DATA_DIR / f"{name}.json"
        text = serialize(data)
        if path.exists() and path.read_text(encoding="utf-8") == text:
            continue
        outdated.append(path.relative_to(PROJECT_ROOT).as_posix())
        if not args.check:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")

    if args.check and outdated:
        print("JSON đã lệch code, chạy lại python -m web.scripts.export_site_data:", *outdated, sep="\n  ")
        sys.exit(1)
    print("Đã ghi:" if outdated else "Không có thay đổi.", *outdated, sep="\n  ")

if __name__ == "__main__":
    main()
