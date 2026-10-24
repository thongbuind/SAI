"""Route chat: trạng thái server, danh sách mô hình và sinh câu trả lời.

``ModelRegistry`` và ``RateLimiter`` nằm trong ``app.state`` (đặt ở ``create_app``).
"""

import asyncio
import time
from typing import Annotated
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, StringConstraints
from web.backend.analytics_middleware import log_model_usage
from web.backend.history_routes import ID_PATTERN
from web.backend.model_registry import ModelRegistry, ServerBusy

router = APIRouter()

MAX_MESSAGE_CHARS = 2000
RATE_LIMIT_SEC = 2.0

SafeId = Annotated[str, Field(pattern=f"^{ID_PATTERN}$")]

class RateLimiter:
    """Mỗi user_id gửi tối đa một tin trong ``interval`` giây."""
    def __init__(self, interval: float):
        self.interval = interval
        self._last: dict[str, float] = {}

    def allow(self, key: str, now: float) -> bool:
        # Bỏ các mốc đã hết hạn để dict không phình theo số user_id từng gửi.
        self._last = {k: t for k, t in self._last.items() if now - t < self.interval}
        if key in self._last:
            return False
        self._last[key] = now
        return True

class ChatRequest(BaseModel):
    message: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_MESSAGE_CHARS)]
    user_id: SafeId
    model_name: str
    session_id: SafeId | None = None

class ChatReply(BaseModel):
    reply: str
    model: str
    latency_ms: float

def _registry(request: Request) -> ModelRegistry:
    return request.app.state.registry

@router.get("/health")
def health(request: Request):
    return {"status": _registry(request).status}

@router.get("/models")
def list_models(request: Request):
    registry = _registry(request)
    return {
        "models": [{"name": name, "state": registry.state(name)} for name in registry.names],
        "default": registry.default,
    }

@router.post("/chat")
async def chat(req: ChatRequest, request: Request) -> ChatReply:
    registry = _registry(request)
    if req.model_name not in registry.names:
        raise HTTPException(400, f"Không có mô hình {req.model_name}.")
    state = registry.state(req.model_name)
    if state == "error":
        raise HTTPException(503, f"Mô hình {req.model_name} không nạp được.")
    if state != "ready":
        raise HTTPException(503, f"Mô hình {req.model_name} đang được nạp, thử lại sau.")
    if not request.app.state.rate_limiter.allow(req.user_id, time.time()):
        raise HTTPException(429, "Gửi quá nhanh, thử lại sau.")
    try:
        reply = await registry.generate(req.model_name, req.message)
    except ServerBusy:
        raise HTTPException(503, "Máy chủ đang bận, thử lại sau.") from None

    await asyncio.to_thread(
        log_model_usage,
        ts=time.time(),
        model_name=reply.model,
        user_id=req.user_id,
        prompt_len=len(req.message),
        reply_len=len(reply.text),
        latency_ms=reply.latency_ms,
        tokens_in=reply.prompt_tokens,
        tokens_out=reply.completion_tokens,
        session_id=req.session_id,
    )
    return ChatReply(reply=reply.text, model=reply.model, latency_ms=round(reply.latency_ms))
