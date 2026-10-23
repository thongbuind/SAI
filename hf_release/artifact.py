"""Manifest công khai và kiểm tra toàn vẹn bộ phát hành, độc lập working tree.

Manifest băm mọi file upload trừ chính nó. Báo cáo verify băm manifest.
Các file điều khiển nội bộ không được upload hoặc tính vào bộ file này.
"""

import ast
import hashlib
import json
from pathlib import Path, PurePosixPath

SCHEMA_VERSION = 1
MANIFEST_FILE = "release_manifest.json"
LOCAL_ONLY_FILES = {"release.json", "verify.json", "push.json", "push_plan.json"}


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        result = json.load(stream)
    if not isinstance(result, dict):
        raise ValueError(f"{path.name}: cần JSON object")
    return result


def write_json(path: Path, value: dict) -> None:
    """Ghi nguyên tử để lần chạy bị ngắt không để lại JSON dở dang."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def payload_files(folder: Path) -> list[str]:
    files = []
    for path in folder.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Bản phát hành không nhận symlink: {path}")
        name = path.relative_to(folder).as_posix()
        if path.is_file() and name not in LOCAL_ONLY_FILES | {MANIFEST_FILE}:
            files.append(name)
    return sorted(files)


def _assignment(path: Path, name: str) -> ast.expr:
    """Đọc khai báo snapshot mà không chạy code đó hoặc import từ working tree."""
    for statement in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(statement, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in statement.targets
        ):
            return statement.value
    raise ValueError(f"{path.name}: thiếu hằng số {name}")


def _literal(path: Path, name: str):
    return ast.literal_eval(_assignment(path, name))


def validate_artifact(folder: Path) -> dict:
    """Kiểm tra danh sách file, hash và quan hệ phiên bản trong bộ phát hành."""
    folder = Path(folder)
    manifest = read_json(folder / MANIFEST_FILE)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Định dạng manifest không được hỗ trợ; cần export lại")
    expected = manifest["files"]
    for name in expected:
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or path.as_posix() != name:
            raise ValueError(f"Đường dẫn file không hợp lệ: {name}")
    actual = payload_files(folder)
    if set(actual) != set(expected):
        raise ValueError(f"Danh sách file thay đổi: thiếu={sorted(set(expected) - set(actual))}, thêm={sorted(set(actual) - set(expected))}")
    for name in actual:
        if sha256_file(folder / name) != expected[name]:
            raise ValueError(f"File thay đổi sau export: {name}")
    config = read_json(folder / "config.json")
    tokenizer = read_json(folder / "tokenizer_config.json")
    version = manifest["version"]
    if config.get("sai_version") != version or _literal(folder / "version.py", "MODEL_VERSION") != version:
        raise ValueError("Phiên bản code SAI, config và manifest không đồng bộ")
    if config.get("component_versions") != manifest["component_versions"]:
        raise ValueError("Phiên bản thành phần trong config và manifest không đồng bộ")
    components = _assignment(folder / "version.py", "COMPONENT_VERSIONS")
    if not isinstance(components, ast.Dict):
        raise ValueError("COMPONENT_VERSIONS của snapshot cần là mapping khai báo tường minh")
    component_names = {ast.literal_eval(key) for key in components.keys}
    if component_names != set(manifest["component_versions"]):
        raise ValueError("Danh sách thành phần trong manifest không khớp code SAI")
    for name, component_version in manifest["component_versions"].items():
        source = "version.py" if name == "tokenizer" else f"{name}.py"
        constant = "TOKENIZER_VERSION" if name == "tokenizer" else "VERSION"
        if _literal(folder / source, constant) != component_version:
            raise ValueError(f"Phiên bản thành phần không khớp code: {name}")
    model_config = manifest["model_config"]
    if any(config.get(name) != value for name, value in model_config.items()):
        raise ValueError("Config model khác manifest")
    if tokenizer.get("model_max_length") != model_config["max_seq_len"]:
        raise ValueError("Context tokenizer và model không đồng bộ")
    return manifest


def check_verified(folder: Path) -> dict:
    manifest = validate_artifact(folder)
    report = read_json(folder / "verify.json")
    if report.get("schema_version") != SCHEMA_VERSION or report.get("passed") is not True:
        raise ValueError("Chưa có báo cáo verify đạt cho định dạng phát hành này")
    if report.get("manifest_sha256") != sha256_file(folder / MANIFEST_FILE):
        raise ValueError("Manifest thay đổi sau verify; cần kiểm chứng lại")
    if report.get("version") != manifest["version"] or report.get("repo_id") != manifest["repo_id"]:
        raise ValueError("Báo cáo verify không thuộc bản phát hành này")
    return manifest
