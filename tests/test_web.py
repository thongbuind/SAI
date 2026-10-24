import dataclasses
import json
import time
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from fastapi.testclient import TestClient
from SAI.modeling import SAIConfig, SAIModel
from foundation_lab.architecture import official_model_config
from hf_release.checkpoint import save_safetensors
from hf_release.spec import load_release_spec
from helpers import TINY_MODEL
from web import __main__ as cli
from web.backend import analytics_middleware, history_routes, model_loader
from web.backend.main import create_app
from web.backend.model_loader import (
    SERVED_MODELS, hub_repo_id, load_hf_snapshot, load_model_config, local_checkpoint,
)
from web.backend.model_registry import ModelRegistry

RELEASES_DIR = Path(__file__).resolve().parents[1] / "hf_release" / "releases"
READY_TIMEOUT_SEC = 5.0

class FakeModel:
    """Thay mô hình thật: trả lời cố định để test HTTP không cần nạp trọng số."""
    def generate(self, message):
        return SimpleNamespace(text=f"trả lời: {message}", prompt_tokens=3, completion_tokens=2)

def fake_loader(name):
    if name == "SAI_375M":
        raise RuntimeError("giả lập nạp lỗi")
    return FakeModel()

@pytest.fixture
def site_dir(tmp_path):
    site = tmp_path / "site"
    (site / "try").mkdir(parents=True)
    (site / "index.html").write_text("trang chủ", encoding="utf-8")
    (site / "try" / "index.html").write_text("trải nghiệm", encoding="utf-8")
    (site / "404.html").write_text("không có trang", encoding="utf-8")
    (tmp_path / "secret.txt").write_text("bí mật", encoding="utf-8")
    return site

@pytest.fixture
def isolated_data(tmp_path, monkeypatch):
    """Không bao giờ ghi vào web/analytics.db hay web/chat_history thật.

    Middleware ghi log request trong thread nền sau khi trả response, có thể chạy sau khi test kết thúc,
    nên thay hẳn hàm ghi bằng hàm rỗng thay vì chỉ đổi đường dẫn DB.
    """
    monkeypatch.setattr(analytics_middleware, "log_request", lambda *args: None)
    monkeypatch.setattr(analytics_middleware, "DB_PATH", tmp_path / "analytics.db")
    monkeypatch.setattr(history_routes, "HISTORY_DIR", tmp_path / "chat_history")

@pytest.fixture
def idle_client(site_dir, isolated_data):
    """Không dùng ``with``: lifespan không chạy nên mô hình ở trạng thái pending."""
    return TestClient(create_app(site_dir, ModelRegistry(["SAI_100M", "SAI_35M"], fake_loader)))

@pytest.fixture
def client(site_dir, isolated_data):
    app = create_app(site_dir, ModelRegistry(["SAI_100M", "SAI_35M", "SAI_375M"], fake_loader))
    with TestClient(app) as test_client:
        deadline = time.monotonic() + READY_TIMEOUT_SEC
        while test_client.get("/api/health").json()["status"] == "loading":
            assert time.monotonic() < deadline, "mô hình giả không nạp xong"
            time.sleep(0.01)
        yield test_client

def chat(client, message="xin chào", user_id="tester", model_name="SAI_100M"):
    return client.post("/api/chat", json={"message": message, "user_id": user_id, "model_name": model_name})

def test_site_is_served_and_cannot_escape_site_dir(idle_client):
    assert idle_client.get("/").text == "trang chủ"
    assert idle_client.get("/try/").text == "trải nghiệm"
    missing = idle_client.get("/khong-co/")
    assert (missing.status_code, missing.text) == (404, "không có trang")
    assert idle_client.get("/..%2fsecret.txt").status_code == 404

def test_models_are_pending_before_startup(idle_client):
    assert idle_client.get("/api/health").json() == {"status": "loading"}
    assert idle_client.get("/api/models").json() == {
        "models": [{"name": "SAI_100M", "state": "pending"}, {"name": "SAI_35M", "state": "pending"}],
        "default": "SAI_100M",
    }
    assert chat(idle_client).status_code == 503

def test_failed_model_is_reported_without_blocking_others(client):
    assert client.get("/api/health").json() == {"status": "error"}
    states = {m["name"]: m["state"] for m in client.get("/api/models").json()["models"]}
    assert states == {"SAI_100M": "ready", "SAI_35M": "ready", "SAI_375M": "error"}
    assert chat(client, model_name="SAI_375M").status_code == 503

def test_chat_replies_with_model_and_latency(client):
    response = chat(client, message="  xin chào  ")
    assert response.status_code == 200
    body = response.json()
    assert (body["reply"], body["model"]) == ("trả lời: xin chào", "SAI_100M")
    assert body["latency_ms"] >= 0

def test_chat_rejects_bad_requests(client):
    assert chat(client, model_name="sft1_35M.pt", user_id="a").status_code == 400
    assert chat(client, message="   ", user_id="b").status_code == 422
    assert chat(client, message="x" * 2001, user_id="c").status_code == 422
    assert chat(client, user_id="../etc").status_code == 422

def test_chat_rate_limit_per_user(client):
    assert chat(client, user_id="fast").status_code == 200
    assert chat(client, user_id="fast").status_code == 429
    assert chat(client, user_id="other").status_code == 200

def test_history_flow_under_api(idle_client):
    session = {
        "id": "s1", "title": "chào", "created_at": 1.0, "user_id": "owner",
        "messages": [{"role": "user", "text": "chào"}, {"role": "ai", "text": "chào bạn", "model": "SAI_35M"}],
    }
    assert idle_client.post("/api/history/save", json=session).json() == {"ok": True}
    assert [s["id"] for s in idle_client.get("/api/history", params={"user_id": "owner"}).json()["sessions"]] == ["s1"]
    assert idle_client.get("/api/history/s1", params={"user_id": "stranger"}).status_code == 403

    assert idle_client.patch("/api/history/s1/share", params={"user_id": "owner"}).json()["shared"] is True
    community = idle_client.get("/api/history/community").json()["sessions"]
    assert [(s["id"], s["model"]) for s in community] == [("s1", "SAI_35M")]
    assert idle_client.get("/api/history/s1", params={"user_id": "stranger"}).status_code == 200

    assert idle_client.delete("/api/history/s1", params={"user_id": "stranger"}).status_code == 403
    assert idle_client.delete("/api/history/s1", params={"user_id": "owner"}).json() == {"ok": True}
    assert idle_client.get("/api/history/community").json() == {"sessions": []}

def test_admin_api_moved_under_api(idle_client):
    # Chưa đặt ADMIN_KEY thì admin bị tắt; route phải tồn tại ở đường dẫn mới.
    assert idle_client.get("/api/admin/web/overview").status_code in (401, 503)

def test_hub_snapshot_loads_into_official_model(tmp_path):
    """Thư mục tải từ Hub được dựng thành SAIModel bằng trọng số và tokenizer của chính thư mục đó."""
    torch.manual_seed(0)
    config = SAIConfig(**{k: v for k, v in TINY_MODEL.items() if k != "family"})
    source = SAIModel(config).eval()
    folder = tmp_path / "snapshot"
    folder.mkdir()
    save_safetensors(source, folder / "model.safetensors")
    hf_config = {"model_type": "SAI", **dataclasses.asdict(config)}
    (folder / "config.json").write_text(json.dumps(hf_config), encoding="utf-8")
    (folder / "tokenizer.model").write_bytes((Path(__file__).resolve().parents[1] / "SAI" / "tokenizer.model").read_bytes())

    loaded = load_hf_snapshot(folder)
    for key, value in source.state_dict().items():
        assert torch.equal(loaded.model.state_dict()[key].cpu(), value), key
    assert loaded.tokenizer.piece_to_id("[BOS]") >= 0

@pytest.mark.parametrize("name", SERVED_MODELS)
def test_served_models_match_release_specs_and_configs(name):
    """Web giữ bảng mô hình riêng (không phụ thuộc lab) nhưng phải khớp release spec và config lab dùng để train."""
    assert hub_repo_id(name) == load_release_spec(RELEASES_DIR / f"{name}.yaml").repo_id
    config = load_model_config(name)
    assert isinstance(config, SAIConfig)
    assert config == official_model_config(name, local_checkpoint(name).name)

def test_web_rejects_non_official_model_config(tmp_path, monkeypatch):
    # Config model thí nghiệm nằm trong lab, nên tự đặt một file như vậy vào thư mục config của web.
    (tmp_path / "SAI_modular_35M.yaml").write_text("family: SAI_modular\n", encoding="utf-8")
    monkeypatch.setattr(model_loader, "MODEL_CONFIG_DIR", tmp_path)
    with pytest.raises(ValueError, match="family: SAI"):
        load_model_config("SAI_modular_35M")

def test_cli_validates_models_and_site(tmp_path):
    assert cli.parse_args([]).models == ["SAI_100M", "SAI_35M", "SAI_375M"]
    assert cli.parse_args(["--models", "SAI_35M"]).models == ["SAI_35M"]
    for bad in ("SAI_500M", "SAI_35M,SAI_35M", ","):
        with pytest.raises(SystemExit):
            cli.parse_args(["--models", bad])
    with pytest.raises(SystemExit, match="npm run build:server"):
        cli.check_site(tmp_path)
