from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from SAI.modeling import SAIConfig
from foundation_lab.architecture import official_model_config
from web.backend import analytics_middleware, main
from web.backend.model_loader import load_model_config, model_config_name

@pytest.fixture
def client(monkeypatch):
    """TestClient không dùng ``with`` nên sự kiện startup (nạp mọi checkpoint thật) không chạy.

    Middleware ghi log trong thread nền sau khi trả response; thay hàm ghi bằng hàm rỗng để test
    không bao giờ ghi vào ``web/analytics.db`` thật, kể cả khi thread chạy sau lúc test kết thúc.
    """
    monkeypatch.setattr(analytics_middleware, "log_request", lambda *args: None)
    return TestClient(main.app)

def test_static_files_cannot_escape_frontend_dir(client):
    assert client.get("/theme.css").status_code == 200
    assert client.get("/..%2fanalytics.db").status_code == 404
    assert client.get("/..%2fbackend%2fmain.py").status_code == 404

def test_chat_without_loaded_model_fails_fast(client):
    response = client.post("/chat", json={"message": "xin chào", "user_id": "tester"})
    assert response.status_code == 503

def test_models_endpoint_uses_key_read_by_frontend(client):
    assert set(client.get("/models").json()) == {"models", "default"}

@pytest.mark.parametrize("checkpoint", ["sft1_35M.pt", "sft1_100M.pt", "sft1_375M.pt"])
def test_checkpoint_name_maps_to_official_model_config(checkpoint):
    """Web đọc config bằng code riêng (không phụ thuộc lab) nhưng phải ra đúng config lab dùng để train."""
    name = model_config_name(Path(checkpoint))
    config = load_model_config(name)
    assert isinstance(config, SAIConfig)
    assert config == official_model_config(name, checkpoint)

def test_web_rejects_non_official_model_config():
    with pytest.raises(ValueError, match="family: SAI"):
        load_model_config("SAI_modular_35M")
