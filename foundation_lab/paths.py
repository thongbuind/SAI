from pathlib import Path

# Các đường dẫn gốc của dự án; đường dẫn tương đối trong config được hiểu từ PROJECT_ROOT.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"                    # config của bản phát hành
MODEL_CONFIG_DIR = CONFIG_DIR / "models"                # model chính thức (family: SAI)
# Config thí nghiệm nằm trong phòng sở hữu nó.
LAB_DIR = Path(__file__).resolve().parent
LAB_MODEL_CONFIG_DIR = LAB_DIR / "architecture" / "configs"
# Lab tìm tên model ở cả model chính thức lẫn bản thí nghiệm; một tên chỉ được có ở một nơi.
MODEL_CONFIG_DIRS = (MODEL_CONFIG_DIR, LAB_MODEL_CONFIG_DIR)
DATA_CONFIG_DIR = LAB_DIR / "data" / "configs"
EXPERIMENT_CONFIG_DIR = LAB_DIR / "core" / "configs"
# Bộ câu hỏi mặc định của phòng Suy luận; web lab đọc và sửa file này (đường dẫn lấy qua "describe").
INFERENCE_BENCHMARK = LAB_DIR / "inference" / "configs" / "benchmark.yaml"
DATA_DIR = PROJECT_ROOT / "data"
RUNS_DIR = PROJECT_ROOT / "runs"
MODEL_DIR = PROJECT_ROOT / "model"             # checkpoint đã train (vd. sft1_35M.pt), không có trong git
TOKENIZER_MODEL = PROJECT_ROOT / "SAI" / "tokenizer.model"

def resolve_project_path(path) -> Path:
    """Đường dẫn tuyệt đối giữ nguyên; đường dẫn tương đối được nối vào PROJECT_ROOT."""
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path

def project_relative_path(path: str | Path) -> str:
    """Đường dẫn để ghi vào thư mục run: file trong dự án ghi tương đối theo PROJECT_ROOT (posix), để run không mang
    đường dẫn riêng của máy và so được giữa các máy; đường dẫn ngoài dự án hoặc vốn đã tương đối giữ nguyên."""
    absolute = Path(path)
    if absolute.is_absolute() and absolute.is_relative_to(PROJECT_ROOT):
        return absolute.relative_to(PROJECT_ROOT).as_posix()
    return str(path)
