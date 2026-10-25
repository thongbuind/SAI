"""Lineage: mọi kết quả truy được về code, config, dữ liệu, tokenizer và môi trường đã tạo ra nó."""

import hashlib
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
import torch
from foundation_lab import __version__ as LAB_VERSION
from foundation_lab.paths import PROJECT_ROOT, TOKENIZER_MODEL, project_relative_path

def _git(*args) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), *args],
            capture_output=True, text=True, timeout=10, check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()

def sha256_file(path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()

def file_fingerprint(path: Path, hash_limit_bytes: int = 16 << 20) -> dict:
    """Dấu vân tay rẻ cho file dữ liệu: kích thước + mtime; chỉ băm nội dung với file nhỏ (manifest...).

    ``path`` trong kết quả tương đối theo gốc dự án nếu file nằm trong dự án (project_relative_path).
    """
    path = Path(path)
    if not path.exists():
        return {"path": project_relative_path(path), "exists": False}
    stat = path.stat()
    info = {"path": project_relative_path(path), "exists": True, "bytes": stat.st_size, "mtime": stat.st_mtime}
    if stat.st_size <= hash_limit_bytes:
        info["sha256"] = sha256_file(path)
    return info

def git_info() -> dict:
    """Commit, branch và cờ dirty của working tree hiện tại (None nếu không có git)."""
    status = _git("status", "--porcelain")
    return {
        "commit": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status) if status is not None else None,
    }

def collect_lineage(config_hash: str, data_files: dict[str, list[Path]]) -> dict:
    env = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        if torch.cuda.is_available() else [],
    }
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config_hash": config_hash,
        "lab_version": LAB_VERSION,
        "git": git_info(),
        "env": env,
        "tokenizer": file_fingerprint(TOKENIZER_MODEL),
        "data": {
            stage: [file_fingerprint(path) for path in paths]
            for stage, paths in data_files.items()
        },
    }
