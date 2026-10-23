import json
import time
from pathlib import Path

class MetricsLogger:
    """Ghi metric thành ``metrics.jsonl``: mỗi dòng một bản ghi (stage, split, bộ đếm, giá trị)."""

    def __init__(self, path: Path | None):
        self.path = Path(path) if path is not None else None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: dict):
        if self.path is None:
            return
        line = {"time": round(time.time(), 3), **record}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

    @staticmethod
    def read(path) -> list[dict]:
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
