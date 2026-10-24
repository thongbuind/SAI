"""Thư mục của một lượt chạy::

    runs/<experiment>/<run_id>/
    ├── resolved_config.yaml     # config đã resolve đầy đủ; chạy lại chỉ cần file này
    ├── lineage.json             # git commit, môi trường, tokenizer, dấu vân tay dữ liệu
    ├── invocations.jsonl        # mỗi lần gọi CLI trên run này (lệnh, stage, init override, git)
    ├── metrics.jsonl            # metric theo thời gian của mọi stage
    ├── events.jsonl             # tác vụ (vd. infer): tiến trình ghi dần, xem state/events.py
    ├── result.json              # tác vụ: kết quả khi chạy xong, xem state/results.py
    └── stages/<stage>/          # experiment: mỗi stage một thư mục
        ├── best.pt              # trọng số best theo val (state_dict thuần)
        ├── last.ckpt.pt         # checkpoint resume
        └── done.json            # có file này nghĩa là stage đã xong, kèm kết quả
"""

import json
from datetime import datetime
from pathlib import Path
import yaml
from foundation_lab.core.config import config_hash
from foundation_lab.paths import RUNS_DIR

class RunDir:
    def __init__(self, path):
        self.path = Path(path)

    @classmethod
    def create(cls, experiment_name: str, root=None, run_id: str | None = None) -> "RunDir":
        root = Path(root) if root is not None else RUNS_DIR
        run_id = run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
        path = root / experiment_name / run_id
        path.mkdir(parents=True, exist_ok=False)
        return cls(path)

    @property
    def config_path(self) -> Path:
        return self.path / "resolved_config.yaml"

    @property
    def metrics_path(self) -> Path:
        return self.path / "metrics.jsonl"

    @property
    def events_path(self) -> Path:
        return self.path / "events.jsonl"

    @property
    def result_path(self) -> Path:
        return self.path / "result.json"

    def write_config(self, resolved: dict) -> str:
        digest = config_hash(resolved)
        with open(self.config_path, "w", encoding="utf-8") as f:
            f.write(f"# config_hash: {digest}\n")
            yaml.safe_dump(resolved, f, sort_keys=False, allow_unicode=True)
        return digest

    def read_config(self) -> dict:
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def write_json(self, name: str, data: dict):
        with open(self.path / name, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)

    def append_invocation(self, record: dict):
        line = {"time": datetime.now().isoformat(timespec="seconds"), **record}
        with open(self.path / "invocations.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(line, ensure_ascii=False, default=str) + "\n")

    def stage_dir(self, stage: str) -> Path:
        path = self.path / "stages" / stage
        path.mkdir(parents=True, exist_ok=True)
        return path

    def best_weights(self, stage: str) -> Path:
        return self.stage_dir(stage) / "best.pt"

    def last_checkpoint(self, stage: str) -> Path:
        return self.stage_dir(stage) / "last.ckpt.pt"

    def is_done(self, stage: str) -> bool:
        return (self.path / "stages" / stage / "done.json").exists()

    def mark_done(self, stage: str, result: dict):
        with open(self.stage_dir(stage) / "done.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=str)

    def read_result(self, stage: str) -> dict:
        with open(self.path / "stages" / stage / "done.json", "r", encoding="utf-8") as f:
            return json.load(f)
