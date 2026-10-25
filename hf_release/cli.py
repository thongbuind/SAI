"""Dòng lệnh quản lý Hugging Face.

    python -m hf_release export hf_release/releases/SAI_35M.yaml
    python -m hf_release verify DUONG_DAN_EXPORT
    python -m hf_release push   DUONG_DAN_EXPORT            # lưu kế hoạch để duyệt
    python -m hf_release push   DUONG_DAN_EXPORT --confirm  # cần export từ commit sạch

Thay DUONG_DAN_EXPORT bằng đường dẫn được lệnh export in ra; không khai báo version trong YAML.
"""

import argparse
import json
import sys
from foundation_lab.core.config import ConfigError

def cmd_export(args):
    from hf_release.export import export_release

    out_dir = export_release(args.spec, out_dir=args.out)
    print(f"Đã export: {out_dir}")
    print(f"Bước tiếp theo: python -m hf_release verify {out_dir}")

def cmd_verify(args):
    from hf_release.verify import verify_export

    report = verify_export(args.export_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        sys.exit(1)

def cmd_push(args):
    from hf_release.push import push_release

    push_release(args.export_dir, confirm=args.confirm, create_repo=args.create_repo)

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m hf_release", description="Quản lý phát hành SAI trên Hugging Face")
    sub = parser.add_subparsers(dest="command", required=True)

    export = sub.add_parser("export", help="dựng thư mục repo HF từ một release spec")
    export.add_argument("spec", help="file trong hf_release/releases/")
    export.add_argument("--out", help="thư mục đích (mặc định runs/releases/<spec>/<version>)")
    export.set_defaults(func=cmd_export)

    verify = sub.add_parser("verify", help="tải thư mục export qua Transformers và so với model chính thức")
    verify.add_argument("export_dir")
    verify.set_defaults(func=cmd_verify)

    push = sub.add_parser("push", help="đưa thư mục export lên Hub (mặc định chỉ xem kế hoạch)")
    push.add_argument("export_dir")
    push.add_argument("--confirm", action="store_true", help="push thật sau khi gõ xác nhận repo_id")
    push.add_argument("--create-repo", action="store_true", help="tạo repo nếu chưa tồn tại")
    push.set_defaults(func=cmd_push)
    return parser

def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except ConfigError as error:
        print(f"Lỗi config: {error}", file=sys.stderr)
        sys.exit(2)
