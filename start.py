"""Chạy web SAI trên máy bằng một lệnh: ``python start.py`` (từ thư mục nào cũng được).

1. Cài thư viện của site nếu chưa có (``npm ci``), build site bản server (``npm run build:server``).
2. Chạy ``python -m web --lab``: chat ở http://127.0.0.1:8000/try/, bàn thử nghiệm ở
   http://127.0.0.1:8000/lab/workbench/inference/.

Tham số thêm được chuyển thẳng cho ``python -m web``, ví dụ chỉ nạp mô hình nhỏ cho chat để đỡ RAM:

    python start.py --models SAI_35M
"""

import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SITE_DIR = PROJECT_ROOT / "web" / "site"

def build_site() -> None:
    if not (SITE_DIR / "node_modules").is_dir():
        subprocess.run(["npm", "ci"], cwd=SITE_DIR, check=True)
    subprocess.run(["npm", "run", "build:server"], cwd=SITE_DIR, check=True)

def main() -> None:
    build_site()
    os.chdir(PROJECT_ROOT)
    # Thay tiến trình hiện tại bằng server: Ctrl+C dừng thẳng server, không qua tiến trình trung gian.
    os.execv(sys.executable, [sys.executable, "-m", "web", "--lab", *sys.argv[1:]])

if __name__ == "__main__":
    main()
