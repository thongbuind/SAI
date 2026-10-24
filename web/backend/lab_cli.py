"""Bàn thử nghiệm (web lab): nút bấm trên web gọi lệnh ``python -m foundation_lab`` qua tiến trình con, rồi đọc
thư mục run mà lệnh ghi ra để hiển thị. Chỉ bật khi chạy ``python -m web --lab``.

Web phụ thuộc lab qua hai hợp đồng, không qua import:
- Lệnh: ``describe`` (JSON mô tả lệnh, schema spec, model, trọng số) và ``<lệnh> -`` (spec đọc từ stdin),
  mã thoát 0 xong / 2 lỗi config / 1 lỗi khi chạy.
- File trong ``runs/<lệnh>/<run_id>/``: ``resolved_config.yaml``, ``events.jsonl``, ``result.json``.
- File benchmark của phòng Suy luận (``benchmark_path`` trong ``describe``): YAML ``prompts: [...]``, web đọc và ghi.

File này gọi lệnh ``describe``; hàng đợi job ở lab_jobs.py, đọc thư mục run ở lab_runs.py, API ở lab_routes.py.
"""

import json
import subprocess
import sys
from pathlib import Path

# Lệnh lab chạy bằng đúng Python đang chạy server (cùng môi trường cài đặt).
LAB_COMMAND = (sys.executable, "-m", "foundation_lab")
# Phiên bản output "describe" và "result.json" mà web lab hiểu; lab tăng phiên bản thì web phải cập nhật.
DESCRIBE_FORMAT_VERSION = 1
RESULT_FORMAT_VERSION = 1
STDERR_TAIL_CHARS = 2000

class LabError(RuntimeError):
    """Lệnh lab thất bại hoặc trả về định dạng web không hiểu."""

def describe(lab_command: tuple[str, ...], project_root: Path) -> dict:
    completed = subprocess.run([*lab_command, "describe"], cwd=project_root, capture_output=True, text=True)
    if completed.returncode != 0:
        raise LabError(f"'describe' thoát mã {completed.returncode}: {completed.stderr[-STDERR_TAIL_CHARS:]}")
    description = json.loads(completed.stdout)
    if description.get("format_version") != DESCRIBE_FORMAT_VERSION:
        raise LabError(
            f"'describe' có format_version {description.get('format_version')!r}, web lab hiểu {DESCRIBE_FORMAT_VERSION}"
        )
    return description
