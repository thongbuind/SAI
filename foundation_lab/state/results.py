"""Kết quả của một lượt chạy tác vụ (inference...): ``result.json``.

Định dạng chung cho mọi lệnh để kết quả các lần chạy so sánh được với nhau:
- ``metrics``: số đo có tên, gom theo ``group`` (vd. tên biến thể), kèm đơn vị và chiều tốt hơn.
- ``records``: các bản ghi chi tiết (vd. mỗi câu hỏi × biến thể một bản ghi).
- ``timings``: thời gian các pha, mili giây.

Đổi ý nghĩa hoặc xoá khoá thì tăng ``RESULT_FORMAT_VERSION``.
"""

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

RESULT_FORMAT_VERSION = 1

@dataclass
class Metric:
    group: str
    name: str
    value: float
    unit: str = ""
    better: Literal["lower", "higher", "none"] = "none"

@dataclass
class RunResult:
    command: str
    metrics: list[Metric]
    records: list[dict]
    timings: dict[str, float]
    format_version: int = RESULT_FORMAT_VERSION

def write_result(path, result: RunResult):
    """Ghi qua file tạm rồi đổi tên: bên đọc thấy ``result.json`` là thấy bản đầy đủ."""
    path = Path(path)
    temporary = path.with_suffix(".json.tmp")
    with open(temporary, "w", encoding="utf-8") as f:
        json.dump(asdict(result), f, ensure_ascii=False, indent=2)
    os.replace(temporary, path)

def read_result(path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
