import sys
from pathlib import Path
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))

from helpers import TINY_MODEL, write_tiny_dataset  # noqa: E402
from foundation_lab.catalog import build_model  # noqa: E402

@pytest.fixture
def tiny_model():
    """Model tí hon ở chế độ eval, FP32, seed cố định — dùng cho test bất biến."""
    torch.manual_seed(0)
    return build_model(TINY_MODEL).eval()

@pytest.fixture
def tiny_data(tmp_path):
    return write_tiny_dataset(tmp_path / "data" / "processed")
