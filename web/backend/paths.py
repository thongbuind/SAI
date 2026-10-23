from pathlib import Path

# Đường dẫn của web tính theo vị trí file này, không theo thư mục đang chạy lệnh.
WEB_DIR = Path(__file__).resolve().parents[1]
FRONTEND_DIR = WEB_DIR / "frontend"
HISTORY_DIR = WEB_DIR / "chat_history"
ANALYTICS_DB = WEB_DIR / "analytics.db"

# Tài nguyên của model chính thức; web chỉ phụ thuộc ``SAI``, không dùng đường dẫn của training lab.
PROJECT_ROOT = WEB_DIR.parent
MODEL_CONFIG_DIR = PROJECT_ROOT / "config" / "models"
TOKENIZER_MODEL = PROJECT_ROOT / "SAI" / "tokenizer.model"
