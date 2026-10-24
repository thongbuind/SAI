"""Đọc thư mục run do lab ghi: ``runs/<lệnh>/<run_id>/`` (resolved_config.yaml, events.jsonl, result.json).

Web chỉ đọc, không ghi gì vào đây. Trạng thái của một run:
- Job còn trong bộ nhớ của server (đang chờ, đang chạy, vừa bị huỷ, lỗi config chưa tạo run): lấy từ job.
- Còn lại suy ra từ file: có ``result.json`` là ``succeeded``; sự kiện cuối là ``failed`` là ``failed``;
  không thì ``incomplete`` (bị dừng giữa chừng, vd. huỷ hoặc server tắt trước khi server này khởi động).
"""

import json
from pathlib import Path
import yaml
from web.backend.lab_cli import RESULT_FORMAT_VERSION, LabError
from web.backend.lab_jobs import Job

CONFIG_FILE = "resolved_config.yaml"
EVENTS_FILE = "events.jsonl"
RESULT_FILE = "result.json"
TERMINAL_EVENTS = ("finished", "failed")
# Trạng thái mà job trong bộ nhớ biết chắc hơn file (file chưa có, hoặc chưa ghi xong).
JOB_OWNED_STATES = ("queued", "running", "cancelled")

def read_events(path: Path, offset: int = 0) -> tuple[list[dict], int]:
    """Các sự kiện đã ghi xong (dòng kết thúc bằng ``\\n``) từ byte ``offset``; trả kèm offset mới.

    Lab ghi dần trong lúc chạy nên dòng cuối có thể đang ghi dở; dòng đó để lần đọc sau.
    """
    if not path.exists():
        return [], offset
    with open(path, "rb") as f:
        f.seek(offset)
        chunk = f.read()
    complete = chunk[:chunk.rfind(b"\n") + 1]
    events = [json.loads(line) for line in complete.decode("utf-8").splitlines() if line.strip()]
    return events, offset + len(complete)

def read_result(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        result = json.load(f)
    if result.get("format_version") != RESULT_FORMAT_VERSION:
        raise LabError(f"{path}: format_version {result.get('format_version')!r}, web lab hiểu {RESULT_FORMAT_VERSION}")
    return result

def read_config(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def file_state(run_path: Path) -> str:
    if (run_path / RESULT_FILE).exists():
        return "succeeded"
    events, _ = read_events(run_path / EVENTS_FILE)
    if events and events[-1]["kind"] == "failed":
        return "failed"
    return "incomplete"

def run_state(run_path: Path, job: Job | None) -> str:
    if job is None:
        return file_state(run_path)
    if job.state == "running":
        # Lab đã ghi sự kiện kết thúc nhưng tiến trình chưa thoát hẳn (Python còn dọn dẹp): file đã là kết quả cuối.
        state = file_state(run_path)
        return job.state if state == "incomplete" else state
    if job.state in JOB_OWNED_STATES or not run_path.exists():
        return job.state
    return file_state(run_path)

def run_summary(command: str, run_id: str, run_path: Path, job: Job | None) -> dict:
    config = read_config(run_path / CONFIG_FILE)
    return {
        "command": command,
        "run_id": run_id,
        "state": run_state(run_path, job),
        "config": config if config is not None else (job.spec if job is not None else None),
    }

def run_detail(command: str, run_id: str, run_path: Path, job: Job | None) -> dict:
    events, _ = read_events(run_path / EVENTS_FILE)
    return {
        **run_summary(command, run_id, run_path, job),
        "events": events,
        "result": read_result(run_path / RESULT_FILE),
        "stderr": list(job.stderr_tail) if job is not None else [],
    }

def list_run_ids(runs_root: Path, command: str, jobs: list[Job]) -> list[str]:
    """Run trên đĩa cộng job chưa có thư mục (đang chờ, lỗi config); mới nhất trước (run_id bắt đầu bằng thời gian)."""
    command_dir = runs_root / command
    on_disk = {path.name for path in command_dir.iterdir() if path.is_dir()} if command_dir.exists() else set()
    return sorted(on_disk | {job.run_id for job in jobs}, reverse=True)
