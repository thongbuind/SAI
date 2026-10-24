"""Nhật ký tiến trình của một lượt chạy: ``events.jsonl``, mỗi dòng một sự kiện ``{seq, time, kind, ...}``.

File được ghi dần trong lúc chạy để công cụ bên ngoài (vd. web lab) theo dõi tiến độ mà không cần
giao tiếp với tiến trình. Mỗi sự kiện được ghi bằng đúng một lần ``write`` có ``\\n`` ở cuối; bên đọc chỉ
lấy những dòng đã kết thúc bằng ``\\n``.

Các loại sự kiện:
- ``started``: bắt đầu, kèm ``total`` (số đơn vị công việc).
- ``progress``: ``done``/``total``, có thể kèm ``message``.
- ``record``: một bản ghi kết quả vừa xong (cũng có trong ``result.json`` khi chạy xong).
- ``finished``: chạy xong, ``result.json`` đã được ghi.
- ``failed``: dừng vì lỗi, kèm ``error``.
"""

import json
import time
from pathlib import Path

EVENT_KINDS = ("started", "progress", "record", "finished", "failed")

class EventLog:
    def __init__(self, path):
        self.path = Path(path)
        self._seq = 0

    def emit(self, kind: str, **data):
        if kind not in EVENT_KINDS:
            raise ValueError(f"Loại sự kiện '{kind}' không hợp lệ; các loại: {list(EVENT_KINDS)}")
        self._seq += 1
        line = {"seq": self._seq, "time": round(time.time(), 3), "kind": kind, **data}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

    @staticmethod
    def read(path) -> list[dict]:
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
