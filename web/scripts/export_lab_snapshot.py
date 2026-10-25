"""Chụp lại phòng làm việc cho bản static (``web/site/src/data/lab_snapshot.json``, xem web/backend/lab_snapshot.py).

Server ``python -m web --lab`` tự chụp sau mỗi job; chạy script này khi run được tạo bằng dòng lệnh lab, khi đổi
benchmark hoặc có model, trọng số mới. Chạy từ thư mục gốc dự án:

    python -m web.scripts.export_lab_snapshot
"""

from web.backend.lab_cli import LAB_COMMAND, describe
from web.backend.lab_snapshot import LabSnapshot
from web.backend.paths import LAB_SNAPSHOT_FILE, PROJECT_ROOT, RUNS_DIR

def main() -> None:
    snapshot = LabSnapshot(describe(LAB_COMMAND, PROJECT_ROOT), PROJECT_ROOT, RUNS_DIR, LAB_SNAPSHOT_FILE)
    snapshot.write()
    print(f"Đã ghi {LAB_SNAPSHOT_FILE.relative_to(PROJECT_ROOT).as_posix()}")

if __name__ == "__main__":
    main()
