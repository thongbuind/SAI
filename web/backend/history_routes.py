import json, logging, re
from pathlib import Path
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional
from web.backend.paths import HISTORY_DIR

router = APIRouter()
logger = logging.getLogger(__name__)

# user_id và session_id do trình duyệt sinh; chỉ nhận ký tự an toàn để làm tên file.
ID_PATTERN = r"[a-zA-Z0-9_\-]+"

def _ensure_dir():
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)

def _valid_id(value: str) -> bool:
    return bool(value) and bool(re.fullmatch(ID_PATTERN, value))

def _session_path(session_id: str) -> Path:
    return HISTORY_DIR / f"{session_id}.json"

def _read_sessions() -> list[dict]:
    """Đọc mọi file lịch sử; file hỏng được bỏ qua kèm cảnh báo để một file không làm hỏng cả danh sách."""
    sessions = []
    for f in HISTORY_DIR.glob("*.json"):
        try:
            sessions.append(json.loads(f.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            logger.warning("Bỏ qua file lịch sử hỏng: %s", f.name)
    return sessions

def _last_model(messages: list[dict]) -> str | None:
    return next((m.get("model") for m in reversed(messages) if m.get("role") == "ai"), None)

class Message(BaseModel):
    role: str
    text: str
    model: Optional[str] = None

class Session(BaseModel):
    id: str
    title: str
    messages: List[Message]
    created_at: float
    user_id: str
    shared: bool = False

@router.get("/history")
def get_my_history(user_id: str = Query(...)):
    if not _valid_id(user_id):
        raise HTTPException(400, "Invalid user_id")
    _ensure_dir()
    sessions = [
        {
            "id":         data["id"],
            "title":      data["title"],
            "created_at": data["created_at"],
            "shared":     data.get("shared", False),
        }
        for data in _read_sessions() if data.get("user_id") == user_id
    ]
    return {"sessions": sessions}

@router.get("/history/community")
def get_community():
    _ensure_dir()
    sessions = [
        {
            "id":         data["id"],
            "title":      data["title"],
            "created_at": data["created_at"],
            "user_id":    data["user_id"],
            "model":      _last_model(data["messages"]),
        }
        for data in _read_sessions() if data.get("shared")
    ]
    sessions.sort(key=lambda s: s["created_at"], reverse=True)
    return {"sessions": sessions}

@router.get("/history/{session_id}")
def get_session(session_id: str, user_id: str = Query(...)):
    """Load session đầy đủ — chỉ owner hoặc session shared mới đọc được."""
    if not _valid_id(session_id) or not _valid_id(user_id):
        raise HTTPException(400, "Invalid id")
    path = _session_path(session_id)
    if not path.exists():
        raise HTTPException(404, "Session not found")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("user_id") != user_id and not data.get("shared"):
        raise HTTPException(403, "Forbidden")
    return data

@router.post("/history/save")
def save_session(session: Session):
    """Lưu / cập nhật session. user_id trong body phải khớp file nếu đã tồn tại."""
    if not _valid_id(session.id) or not _valid_id(session.user_id):
        raise HTTPException(400, "Invalid id")
    _ensure_dir()
    path = _session_path(session.id)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("user_id") != session.user_id:
            raise HTTPException(403, "Forbidden")
    path.write_text(
        json.dumps(session.model_dump(), ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    return {"ok": True}

@router.patch("/history/{session_id}/share")
def toggle_share(session_id: str, user_id: str = Query(...)):
    if not _valid_id(session_id) or not _valid_id(user_id):
        raise HTTPException(400, "Invalid id")
    path = _session_path(session_id)
    if not path.exists():
        raise HTTPException(404, "Session not found")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("user_id") != user_id:
        raise HTTPException(403, "Forbidden")
    data["shared"] = not data.get("shared", False)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"ok": True, "shared": data["shared"]}

@router.delete("/history/{session_id}")
def delete_session(session_id: str, user_id: str = Query(...)):
    if not _valid_id(session_id) or not _valid_id(user_id):
        raise HTTPException(400, "Invalid id")
    path = _session_path(session_id)
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("user_id") != user_id:
            raise HTTPException(403, "Forbidden")
        path.unlink()
    return {"ok": True}
