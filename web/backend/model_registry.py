"""Giữ các mô hình web đang phục vụ: nạp nền theo thứ tự, theo dõi trạng thái, sinh văn bản tuần tự.

Server mở ngay khi khởi động; mô hình nào nạp xong thì chat được với mô hình đó. Mọi yêu cầu sinh văn bản
đi qua một hàng đợi và một worker, để các mô hình không tranh CPU/GPU với nhau.
"""

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal
from web.backend.model_loader import LoadedModel

logger = logging.getLogger(__name__)

ModelState = Literal["pending", "loading", "ready", "error"]
ServerStatus = Literal["loading", "ready", "error"]

# Quá số yêu cầu đang chờ này thì từ chối ngay thay vì để người dùng chờ vô hạn.
MAX_QUEUE = 8

class ServerBusy(Exception):
    """Hàng đợi sinh văn bản đã đầy."""

@dataclass
class Reply:
    text: str
    model: str
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int

class ModelRegistry:
    def __init__(self, names: list[str], loader: Callable[[str], LoadedModel], max_queue: int = MAX_QUEUE):
        if not names or len(set(names)) != len(names):
            raise ValueError(f"Danh sách mô hình phải khác rỗng và không trùng tên: {names}")
        self.names = list(names)
        # Mô hình đầu danh sách là mặc định và được nạp trước nên sẵn sàng sớm nhất.
        self.default = self.names[0]
        self._loader = loader
        self._max_queue = max_queue
        self._states: dict[str, ModelState] = {name: "pending" for name in self.names}
        self._models: dict[str, LoadedModel] = {}
        self._queue: asyncio.Queue | None = None
        self._tasks: list[asyncio.Task] = []

    def state(self, name: str) -> ModelState:
        return self._states[name]

    @property
    def status(self) -> ServerStatus:
        states = set(self._states.values())
        if states & {"pending", "loading"}:
            return "loading"
        return "error" if "error" in states else "ready"

    def start(self) -> None:
        """Bắt đầu nạp mô hình và chạy worker; không chờ nạp xong."""
        self._queue = asyncio.Queue(maxsize=self._max_queue)
        self._tasks = [asyncio.create_task(self._load_all()), asyncio.create_task(self._worker())]

    async def stop(self) -> None:
        for task in self._tasks:
            task.cancel()
        # Chỉ chờ các task kết thúc; CancelledError là kết quả mong đợi khi tắt server.
        await asyncio.gather(*self._tasks, return_exceptions=True)

    async def generate(self, name: str, message: str) -> Reply:
        if self._states.get(name) != "ready":
            raise ValueError(f"Mô hình {name} chưa sẵn sàng (trạng thái {self._states.get(name)})")
        future = asyncio.get_running_loop().create_future()
        try:
            self._queue.put_nowait((name, message, future))
        except asyncio.QueueFull:
            raise ServerBusy from None
        return await future

    async def _load_all(self) -> None:
        for name in self.names:
            self._states[name] = "loading"
            logger.info("Đang nạp %s...", name)
            try:
                self._models[name] = await asyncio.to_thread(self._loader, name)
            except Exception:
                # Một mô hình lỗi không chặn các mô hình còn lại; lỗi hiện qua /api/health và log.
                logger.exception("Nạp %s thất bại", name)
                self._states[name] = "error"
                continue
            self._states[name] = "ready"
            logger.info("Đã nạp %s", name)

    async def _worker(self) -> None:
        while True:
            name, message, future = await self._queue.get()
            try:
                if future.done():
                    continue  # người gửi đã huỷ yêu cầu trong lúc chờ
                start = time.perf_counter()
                result = await asyncio.to_thread(self._models[name].generate, message)
                latency_ms = (time.perf_counter() - start) * 1000
                if not future.done():
                    future.set_result(Reply(result.text, name, latency_ms, result.prompt_tokens, result.completion_tokens))
            except Exception as error:
                if not future.done():
                    future.set_exception(error)
            finally:
                self._queue.task_done()
