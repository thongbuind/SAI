"""Tác vụ: lệnh chạy một lần (không qua pipeline stage) theo một hợp đồng chung, để công cụ bên ngoài
(web lab) chạy mọi tác vụ theo cùng một cách.

Hợp đồng, do ``foundation_lab.cli`` thực thi theo thứ tự:
1. Spec là file YAML/JSON (có ``extends``), ``-`` (đọc từ stdin) hoặc bỏ trống (bắt đầu từ ``{}``).
2. ``resolve(data, overrides)`` áp ``--set`` và thay mọi tham chiếu bằng nội dung, trả về dict tự đủ;
   dict này được ghi thành ``resolved_config.yaml``, nên chạy lại một run chỉ cần đưa file đó làm spec.
3. ``validate(resolved)`` dựng spec và báo ``ConfigError`` trước khi tốn RAM hay thời gian (mã thoát 2,
   chưa tạo thư mục run).
4. ``run(spec, run_dir, events)`` ghi ``events.jsonl`` dần dần (``state/events.py``) và ``result.json`` khi xong
   (``state/results.py``), trả về ``RunResult``.
5. ``inputs(spec)``: nhãn → file đầu vào, để ghi dấu vân tay vào ``lineage.json``.

Thư mục run: ``runs/<tên tác vụ>/<run_id>/``. Phòng sở hữu tác vụ đăng ký nó vào ``TASKS`` với tên lệnh;
``department`` là mã phòng trong docstring của ``foundation_lab`` (vd. ``arch``), nơi web đặt trang của tác vụ.
"""

from dataclasses import dataclass
from typing import Callable
from foundation_lab.core.registry import Registry

@dataclass(frozen=True)
class Task:
    department: str
    spec_cls: type          # dataclass spec; ``describe`` mô tả nó cho web bằng describe_dataclass
    resolve: Callable       # (data: dict, overrides: list[str] | None) -> dict tự đủ
    validate: Callable      # (resolved: dict) -> spec_cls
    run: Callable           # (spec, RunDir, EventLog) -> RunResult
    inputs: Callable        # (spec) -> dict[str, list[Path]]
    help: str               # một dòng, hiện trong --help

# Tên lệnh → Task.
TASKS = Registry("task")
