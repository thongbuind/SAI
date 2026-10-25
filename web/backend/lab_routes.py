"""API của phòng làm việc, gắn dưới ``/api/lab``; chỉ có khi chạy ``python -m web --lab``.

- ``GET  /describe``: output ``describe`` của lab (đọc một lần lúc server khởi động).
- ``GET  /jobs?command=infer``: lịch sử run của một lệnh, mới nhất trước.
- ``POST /jobs``: ``{command, spec}`` → 202 ``{command, run_id}``; spec được kiểm theo lệnh (``SPEC_CHECKS``) rồi
  job vào hàng đợi. Lệnh lab chưa có hàm kiểm tra ở đây thì web không chạy.
- ``GET  /jobs/{command}/{run_id}``: config, trạng thái, sự kiện, kết quả, stderr của job.
- ``GET  /jobs/{command}/{run_id}/events``: SSE, đọc dần ``events.jsonl``; kết thúc bằng sự kiện ``end``.
- ``POST /jobs/{command}/{run_id}/cancel``: huỷ job đang chờ hoặc đang chạy.
- ``GET  /benchmark``, ``PUT /benchmark``: đọc, ghi bộ câu hỏi mặc định của phòng Suy luận (file lab chỉ ra qua
  khoá ``benchmark_path`` của ``describe``).

Mọi route chỉ nhận Host là localhost: nút bấm chạy code trên máy chủ, và một trang web lạ trỏ tên miền của nó
về 127.0.0.1 (DNS rebinding) sẽ gửi Host là tên miền đó. Không bật lab thì ``/api/lab/*`` trả 503 kèm lý do.
"""

import asyncio
import json
import os
from dataclasses import dataclass
from itertools import takewhile
from pathlib import Path as FilePath
from typing import Annotated
from fastapi import APIRouter, Depends, Header, HTTPException, Path, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import yaml
from web.backend.lab_cli import BENCHMARK_KEY, read_benchmark
from web.backend.lab_jobs import ACTIVE_STATES, JobRunner, QueueFull
from web.backend.lab_runs import EVENTS_FILE, TERMINAL_EVENTS, list_run_ids, read_events, run_detail, run_state, run_summary

LOCAL_HOSTS = {"127.0.0.1", "localhost", "[::1]"}
LAB_DISABLED = "Server đang chạy không có phòng làm việc. Trên máy có foundation_lab, chạy: python -m web --lab"

@dataclass
class Lab:
    runner: JobRunner
    description: dict      # output "python -m foundation_lab describe", đọc lúc server khởi động
    runs_root: FilePath

def host_name(host_header: str) -> str:
    """``127.0.0.1:8000`` → ``127.0.0.1``; ``[::1]:8000`` → ``[::1]``."""
    if host_header.startswith("["):
        return host_header[:host_header.index("]") + 1]
    return host_header.split(":", 1)[0]

def require_local_host(request: Request):
    if host_name(request.headers.get("host", "")) not in LOCAL_HOSTS:
        raise HTTPException(403, "Phòng làm việc chỉ nhận truy cập từ localhost")

router = APIRouter(dependencies=[Depends(require_local_host)])
disabled_router = APIRouter()

@disabled_router.api_route("/{path:path}", methods=["GET", "POST"])
def lab_disabled(path: str):
    raise HTTPException(503, LAB_DISABLED)

# run_id thành một phần đường dẫn trên đĩa: chỉ nhận ký tự an toàn, chặn "..", "/".
RunId = Annotated[str, Path(pattern=r"^[A-Za-z0-9_\-]+$")]
EVENTS_POLL_SEC = 0.25

# Kiểm tra spec theo lệnh trước khi chạy: mọi tên model, trọng số... phải nằm trong danh sách của describe, để web
# không chuyển đường dẫn tuỳ ý từ trình duyệt xuống lệnh lab. Lab vẫn kiểm tra lại toàn bộ spec khi chạy.

def check_infer_spec(spec: dict, description: dict):
    """(model, checkpoint) phải là một cặp trong "checkpoints": không nạp trọng số vào model khác cỡ."""
    checkpoints = description["checkpoints"]
    model, checkpoint = spec.get("model"), spec.get("checkpoint")
    if not isinstance(model, str) or model not in checkpoints:
        raise HTTPException(400, f"spec.model: '{model}' không có trong 'checkpoints' của lab")
    if checkpoint not in checkpoints[model]:
        raise HTTPException(400, f"spec.checkpoint: '{checkpoint}' không phải trọng số của model '{model}'")

def check_kernel_spec(spec: dict, description: dict):
    """``models`` là danh sách tên trong "models": lab nối tên thành đường dẫn file config."""
    models = spec.get("models")
    if not isinstance(models, list) or not all(isinstance(name, str) and name in description["models"] for name in models):
        raise HTTPException(400, f"spec.models: cần danh sách tên model có trong 'models' của lab, nhận {models!r}")

def check_objective_spec(spec: dict, description: dict):
    """``experiments`` là danh sách tên trong "experiments": lab nối tên thành đường dẫn file experiment."""
    experiments = spec.get("experiments")
    if not isinstance(experiments, list) or not all(
        isinstance(name, str) and name in description["experiments"] for name in experiments
    ):
        raise HTTPException(400, f"spec.experiments: cần danh sách tên experiment có trong 'experiments' của lab, nhận {experiments!r}")

SPEC_CHECKS = {"infer": check_infer_spec, "compare-kernel": check_kernel_spec, "compare-objective": check_objective_spec}

class JobRequest(BaseModel):
    command: str
    spec: dict

class Benchmark(BaseModel):
    prompts: list[str]

def _lab(request: Request) -> Lab:
    return request.app.state.lab

def _runner(request: Request) -> JobRunner:
    return _lab(request).runner

def _check_command(request: Request, command: str):
    commands = _lab(request).description["commands"]
    if command not in commands:
        raise HTTPException(400, f"Lệnh '{command}' không có; các lệnh: {sorted(commands)}")

def _run_path(request: Request, command: str, run_id: str):
    _check_command(request, command)
    path = _lab(request).runs_root / command / run_id
    if not path.exists() and _runner(request).get(command, run_id) is None:
        raise HTTPException(404, f"Không có run {command}/{run_id}")
    return path

@router.get("/describe")
def get_describe(request: Request):
    return _lab(request).description

@router.get("/jobs")
def list_jobs(request: Request, command: str):
    _check_command(request, command)
    runner = _runner(request)
    runs_root = _lab(request).runs_root
    return {
        "runs": [
            run_summary(command, run_id, runs_root / command / run_id, runner.get(command, run_id))
            for run_id in list_run_ids(runs_root, command, runner.jobs(command))
        ],
    }

@router.post("/jobs", status_code=202)
def submit_job(request: Request, body: JobRequest):
    _check_command(request, body.command)
    if body.command not in SPEC_CHECKS:
        raise HTTPException(400, f"Web chưa có kiểm tra spec cho lệnh '{body.command}' nên chưa chạy được lệnh này")
    SPEC_CHECKS[body.command](body.spec, _lab(request).description)
    try:
        job = _runner(request).submit(body.command, body.spec)
    except QueueFull:
        raise HTTPException(503, "Hàng đợi đã đầy, chờ job đang chạy xong.") from None
    return {"command": job.command, "run_id": job.run_id}

@router.get("/jobs/{command}/{run_id}")
def get_job(request: Request, command: str, run_id: RunId):
    path = _run_path(request, command, run_id)
    return run_detail(command, run_id, path, _runner(request).get(command, run_id))

@router.post("/jobs/{command}/{run_id}/cancel")
async def cancel_job(request: Request, command: str, run_id: RunId):
    _run_path(request, command, run_id)
    job = _runner(request).get(command, run_id)
    if job is None or job.state not in ACTIVE_STATES:
        raise HTTPException(409, "Job không còn chờ hay chạy để huỷ")
    await _runner(request).cancel(job)
    return {"state": job.state}

@router.get("/jobs/{command}/{run_id}/events")
def stream_job_events(
    request: Request, command: str, run_id: RunId,
    last_event_id: Annotated[int | None, Header()] = None,
):
    path = _run_path(request, command, run_id)
    runner = _runner(request)

    async def stream():
        offset = 0
        last_seq = last_event_id or 0
        while True:
            # Đọc trạng thái trước khi đọc file: job đã kết thúc thì lần đọc này chắc chắn có mọi sự kiện cuối.
            job = runner.get(command, run_id)
            active = job is not None and job.state in ACTIVE_STATES
            events, offset = read_events(path / EVENTS_FILE, offset)
            for event in events:
                if event["seq"] <= last_seq:
                    continue
                last_seq = event["seq"]
                yield f"id: {event['seq']}\nevent: {event['kind']}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"
            ended = any(event["kind"] in TERMINAL_EVENTS for event in events)
            if ended or not active:
                state = run_state(path, job)
                yield f"event: end\ndata: {json.dumps({'state': state})}\n\n"
                return
            await asyncio.sleep(EVENTS_POLL_SEC)

    return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

def _benchmark_path(request: Request) -> FilePath:
    return _runner(request).project_root / _lab(request).description["benchmark_path"]

@router.get("/benchmark")
def get_benchmark(request: Request) -> Benchmark:
    return Benchmark(prompts=read_benchmark(_benchmark_path(request)))

@router.put("/benchmark")
def put_benchmark(request: Request, body: Benchmark) -> Benchmark:
    prompts = [prompt.strip() for prompt in body.prompts]
    if not prompts or not all(prompts):
        raise HTTPException(400, "prompts: cần ít nhất một câu hỏi và không có câu rỗng")
    path = _benchmark_path(request)
    # Giữ các dòng comment đầu file (lab viết giải thích ở đó), chỉ thay danh sách câu hỏi.
    with open(path, "r", encoding="utf-8") as f:
        header = list(takewhile(lambda line: line.startswith("#"), f))
    tmp_path = path.with_name(path.name + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write("".join(header) + yaml.safe_dump({BENCHMARK_KEY: prompts}, allow_unicode=True, sort_keys=False))
    # Ghi file tạm rồi đổi tên: file benchmark không bao giờ bị ghi dở.
    os.replace(tmp_path, path)
    return Benchmark(prompts=prompts)
