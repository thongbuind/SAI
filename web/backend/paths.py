import os
from pathlib import Path

# Đường dẫn của web tính theo vị trí file này, không theo thư mục đang chạy lệnh.
WEB_DIR = Path(__file__).resolve().parents[1]
SITE_DIR = WEB_DIR / "site" / "dist" / "server"
ADMIN_PAGE = WEB_DIR / "backend" / "admin.html"

# Dữ liệu người dùng (lịch sử chat, analytics). Mặc định nằm trong web/; Docker đặt SAI_WEB_DATA_DIR
# trỏ vào một volume để dữ liệu còn lại khi dựng lại container.
DATA_DIR = Path(os.environ.get("SAI_WEB_DATA_DIR", WEB_DIR))
HISTORY_DIR = DATA_DIR / "chat_history"
ANALYTICS_DB = DATA_DIR / "analytics.db"

# Tài nguyên của model chính thức; web chỉ phụ thuộc ``SAI``, không dùng đường dẫn của training lab.
PROJECT_ROOT = WEB_DIR.parent
MODEL_DIR = PROJECT_ROOT / "model"
MODEL_CONFIG_DIR = PROJECT_ROOT / "config" / "models"
TOKENIZER_MODEL = PROJECT_ROOT / "SAI" / "tokenizer.model"
# Bàn thử nghiệm: thư mục run mà lệnh foundation_lab ghi ra (cùng mặc định với foundation_lab.paths.RUNS_DIR).
RUNS_DIR = PROJECT_ROOT / "runs"
