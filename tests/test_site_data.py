"""Dữ liệu JSON của web phải khớp code; lệch thì chạy lại ``python -m web.scripts.export_site_data``."""

import pytest
from web.scripts.export_site_data import DATA_DIR, build_site_data, serialize

@pytest.mark.parametrize("name, data", build_site_data().items())
def test_site_data_matches_code(name, data):
    path = DATA_DIR / f"{name}.json"
    assert path.exists(), f"Thiếu {path}; chạy python -m web.scripts.export_site_data"
    assert path.read_text(encoding="utf-8") == serialize(data), f"{path.name} đã lệch code; chạy python -m web.scripts.export_site_data"
