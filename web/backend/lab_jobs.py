"""Hàng đợi job của phòng làm việc: mỗi job là một lần chạy ``python -m foundation_lab <lệnh> -`` với spec gửi qua stdin.

Mỗi lúc chỉ chạy một job (máy 8GB RAM, mỗi job nạp một model). Trạng thái job chỉ giữ trong bộ nhớ;
kết quả thật nằm trong thư mục run do lab ghi, nên server khởi động lại vẫn đọc được lịch sử (xem runs.py).
Mỗi job kết thúc (xong, lỗi hay bị huỷ) thì gọi ``after_job`` nếu có: server dùng nó để chụp lại phòng làm việc cho
bản static (lab_snapshot.py).
"""

import asyncio
import json
import logging
import secrets
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

JobState = Literal["queued", "running", "succeeded", "failed", "cancelled"]
ACTIVE_STATES = ("queued", "running")
MAX_QUEUE = 4
# Sau SIGTERM, chờ tiến trình tự thoát chừng này giây rồi mới SIGKILL.
CANCEL_TIMEOUT_SEC = 10
STDERR_TAIL_LINES = 50
RUN_ID_SUFFIX_BYTES = 2

class QueueFull(Exception):
    """Đã có MAX_QUEUE job đang chờ."""

@dataclass
class Job:
    command: str
    run_id: str
    spec: dict
    state: JobState = "queued"
    returncode: int | None = None
    stderr_tail: deque = field(default_factory=lambda: deque(maxlen=STDERR_TAIL_LINES))
    cancel_requested: bool = False
    process: asyncio.subprocess.Process | None = None

def new_run_id() -> str:
    """Theo thời gian như run của lab, thêm hậu tố ngẫu nhiên để hai job cùng giây không trùng thư mục."""
    return f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(RUN_ID_SUFFIX_BYTES)}"

class JobRunner:
    def __init__(
        self, lab_command: tuple[str, ...], project_root: Path, runs_root: Path,
        after_job: Callable[[], None] | None = None,
    ):
        self.lab_command = lab_command
        self.project_root = project_root
        self.runs_root = runs_root
        self.after_job = after_job
        self._jobs: dict[tuple[str, str], Job] = {}
        self._queue: asyncio.Queue[Job] = asyncio.Queue()
        self._worker_task: asyncio.Task | None = None

    def start(self):
        self._worker_task = asyncio.create_task(self._worker())

    async def stop(self):
        for job in self._jobs.values():
            if job.state == "running":
                await self.cancel(job)
        if self._worker_task is not None:
            self._worker_task.cancel()

    def get(self, command: str, run_id: str) -> Job | None:
        return self._jobs.get((command, run_id))

    def jobs(self, command: str) -> list[Job]:
        return [job for (job_command, _), job in self._jobs.items() if job_command == command]

    def submit(self, command: str, spec: dict) -> Job:
        waiting = sum(job.state == "queued" for job in self._jobs.values())
        if waiting >= MAX_QUEUE:
            raise QueueFull()
        job = Job(command, new_run_id(), spec)
        self._jobs[(command, job.run_id)] = job
        self._queue.put_nowait(job)
        return job

    async def cancel(self, job: Job):
        job.cancel_requested = True
        if job.state == "queued":
            job.state = "cancelled"
            return
        if job.state != "running" or job.process is None:
            return
        job.process.terminate()
        try:
            await asyncio.wait_for(job.process.wait(), CANCEL_TIMEOUT_SEC)
        except TimeoutError:
            logger.warning("Job %s/%s không dừng sau SIGTERM, gửi SIGKILL", job.command, job.run_id)
            job.process.kill()

    async def _worker(self):
        while True:
            job = await self._queue.get()
            if job.state == "cancelled":
                continue
            await self._run(job)
            if self.after_job is not None:
                await self._call_after_job(job)

    async def _call_after_job(self, job: Job):
        # Chạy trong thread vì đọc, ghi file. Lỗi ở đây không được làm chết worker (các job đang chờ sẽ kẹt mãi):
        # ghi log kèm traceback rồi chạy job tiếp theo.
        try:
            await asyncio.to_thread(self.after_job)
        except Exception:
            logger.exception("after_job lỗi sau job %s/%s", job.command, job.run_id)

    async def _run(self, job: Job):
        job.state = "running"
        try:
            job.process = await asyncio.create_subprocess_exec(
                *self.lab_command, job.command, "-", "--run-id", job.run_id, "--runs-root", str(self.runs_root),
                cwd=self.project_root,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
        except OSError as error:
            # Không mở được tiến trình (sai đường dẫn Python...): job thất bại, hàng đợi vẫn chạy tiếp.
            logger.error("Không chạy được lệnh lab cho job %s/%s: %s", job.command, job.run_id, error)
            job.stderr_tail.append(f"Không chạy được lệnh lab: {error}")
            job.state = "failed"
            return
        job.process.stdin.write(json.dumps(job.spec, ensure_ascii=False).encode("utf-8"))
        await job.process.stdin.drain()
        job.process.stdin.close()
        async for line in job.process.stderr:
            job.stderr_tail.append(line.decode("utf-8", errors="replace").rstrip("\n"))
        job.returncode = await job.process.wait()
        if job.cancel_requested:
            job.state = "cancelled"
        else:
            job.state = "succeeded" if job.returncode == 0 else "failed"
        logger.info("Job %s/%s: %s (mã thoát %s)", job.command, job.run_id, job.state, job.returncode)
