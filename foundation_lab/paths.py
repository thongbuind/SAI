from pathlib import Path

# Các đường dẫn gốc của dự án; đường dẫn tương đối trong config được hiểu từ PROJECT_ROOT.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
MODEL_CONFIG_DIR = CONFIG_DIR / "models"
DATA_CONFIG_DIR = CONFIG_DIR / "data"
EXPERIMENT_CONFIG_DIR = CONFIG_DIR / "experiments"
DATA_DIR = PROJECT_ROOT / "data"
RUNS_DIR = PROJECT_ROOT / "runs"
TOKENIZER_MODEL = PROJECT_ROOT / "SAI" / "tokenizer.model"

def resolve_project_path(path) -> Path:
    """Đường dẫn tuyệt đối giữ nguyên; đường dẫn tương đối được nối vào PROJECT_ROOT."""
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path
