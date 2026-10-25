"""Chạy web SAI bản server: ``python -m web``.

Cần build site trước (``cd web/site && npm run build:server``). Ví dụ:

    python -m web                                  # nguồn local, cả 3 mô hình, 127.0.0.1:8000
    python -m web --models SAI_35M                 # thử nhanh, ít RAM
    python -m web --source hub --host 0.0.0.0      # trong Docker
    python -m web --lab                            # thêm phòng làm việc /lab/workbench/ (chỉ chạy ở 127.0.0.1)

Cách nhanh nhất trên máy: ``python start.py`` ở gốc repo (build site rồi chạy lệnh này kèm --lab).
"""

import argparse
import logging
from pathlib import Path
import uvicorn
from web.backend.lab_cli import LAB_COMMAND, describe
from web.backend.lab_jobs import JobRunner
from web.backend.lab_routes import Lab
from web.backend.lab_snapshot import LabSnapshot
from web.backend.main import create_app
from web.backend.model_loader import SERVED_MODELS, SOURCES, loader_for
from web.backend.model_registry import ModelRegistry
from web.backend.paths import LAB_SNAPSHOT_FILE, PROJECT_ROOT, RUNS_DIR, SITE_DIR

# Mô hình đầu danh sách là mặc định và được nạp trước.
DEFAULT_MODELS = "SAI_100M,SAI_35M,SAI_375M"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
# Phòng làm việc chạy lệnh lab trên máy chủ nên chỉ được nghe ở loopback.
LAB_HOSTS = ("127.0.0.1", "localhost", "::1")

def parse_models(value: str) -> list[str]:
    names = [name.strip() for name in value.split(",") if name.strip()]
    unknown = [name for name in names if name not in SERVED_MODELS]
    if not names or unknown or len(set(names)) != len(names):
        raise argparse.ArgumentTypeError(
            f"--models nhận danh sách không trùng, cách nhau bằng dấu phẩy, trong {', '.join(SERVED_MODELS)}; nhận được: {value!r}"
        )
    return names

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m web", description="Chạy web SAI bản server.")
    parser.add_argument("--source", choices=SOURCES, default="local", help="nơi lấy trọng số: model/ hoặc Hugging Face Hub")
    parser.add_argument("--models", type=parse_models, default=parse_models(DEFAULT_MODELS), help=f"mặc định {DEFAULT_MODELS}")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--lab", action="store_true", help="bật phòng làm việc: nút bấm gọi lệnh foundation_lab trên máy này")
    args = parser.parse_args(argv)
    if args.lab and args.host not in LAB_HOSTS:
        parser.error(f"--lab chỉ chạy với --host trong {', '.join(LAB_HOSTS)}: nút bấm chạy code trên máy chủ")
    return args

def check_site(site_dir: Path) -> None:
    if not (site_dir / "index.html").is_file():
        raise SystemExit(f"Chưa có site đã build ở {site_dir}. Chạy: cd web/site && npm run build:server")

def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    check_site(SITE_DIR)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    lab = None
    if args.lab:
        # describe chạy một lần lúc khởi động; có checkpoint hay config mới thì khởi động lại server.
        description = describe(LAB_COMMAND, PROJECT_ROOT)
        # Mỗi job xong thì chụp lại phòng làm việc cho bản static.
        snapshot = LabSnapshot(description, PROJECT_ROOT, RUNS_DIR, LAB_SNAPSHOT_FILE)
        lab = Lab(JobRunner(LAB_COMMAND, PROJECT_ROOT, RUNS_DIR, after_job=snapshot.write), description, RUNS_DIR)
    app = create_app(SITE_DIR, ModelRegistry(args.models, loader_for(args.source)), lab)
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()
