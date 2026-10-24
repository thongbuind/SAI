"""Chạy một script của pipeline cũ trong môi trường cố định để làm bản đối chiếu.

    python legacy_runner.py <legacy_root> <seed> <script> [args...]

Chỉ cố định môi trường, không sửa thuật toán của pipeline cũ:
- ``torch.manual_seed(seed)`` trước khi script chạy (pipeline cũ không tự đặt seed).
- Ép CPU (ẩn CUDA/MPS) và tắt ``torch.compile`` (trả về hàm gốc).
- ``persistent_workers=False`` khi ``num_workers=0`` (pipeline cũ luôn bật nên DataLoader báo lỗi).
"""

import runpy
import sys
from pathlib import Path
import torch
import torch.utils.data as tud

def main():
    root, seed, script, *script_args = sys.argv[1:]
    root = Path(root).resolve()
    sys.path.insert(0, str(root))

    torch.manual_seed(int(seed))
    torch.compile = lambda fn=None, **kwargs: fn
    torch.cuda.is_available = lambda: False
    torch.backends.mps.is_available = lambda: False

    original_init = tud.DataLoader.__init__

    def patched_init(self, *args, **kwargs):
        if kwargs.get("num_workers", 0) == 0:
            kwargs["persistent_workers"] = False
        original_init(self, *args, **kwargs)

    tud.DataLoader.__init__ = patched_init

    script_path = root / script
    sys.argv = [str(script_path), *script_args]
    runpy.run_path(str(script_path), run_name="__main__")

if __name__ == "__main__":
    main()
