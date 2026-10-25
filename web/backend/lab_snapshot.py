"""Ảnh chụp phòng làm việc cho bản static (GitHub Pages): ``web/site/src/data/lab_snapshot.json``.

Bản static không có API lab nên trang phòng làm việc đọc file này thay cho ``/api/lab``: output ``describe``, bộ câu
hỏi benchmark và mọi run trên đĩa của từng lệnh, mỗi run đúng như ``GET /api/lab/jobs/{command}/{run_id}`` trả về.
Run chỉ đọc từ đĩa (không qua job trong bộ nhớ) nên không có stderr và trạng thái suy ra từ file (lab_runs.py).

File được ghi lại sau mỗi job của phòng làm việc (lab_jobs.py) và bằng tay qua
``python -m web.scripts.export_lab_snapshot`` (run chạy thẳng bằng dòng lệnh lab). Muốn bản GitHub Pages cập nhật
thì commit và push file này.
"""

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from web.backend.lab_cli import read_benchmark
from web.backend.lab_runs import list_run_ids, run_detail

@dataclass
class LabSnapshot:
    description: dict      # output "python -m foundation_lab describe"
    project_root: Path
    runs_root: Path
    out_path: Path

    def build(self) -> dict:
        return {
            "captured_at": datetime.now().isoformat(timespec="seconds"),
            "description": self.description,
            "benchmark": read_benchmark(self.project_root / self.description["benchmark_path"]),
            # Lệnh → run trên đĩa, mới nhất trước (như danh sách lịch sử của trang).
            "runs": {
                command: [
                    run_detail(command, run_id, self.runs_root / command / run_id, None)
                    for run_id in list_run_ids(self.runs_root, command, [])
                ]
                for command in self.description["commands"]
            },
        }

    def write(self) -> None:
        text = json.dumps(self.build(), ensure_ascii=False, indent=2) + "\n"
        # Ghi file tạm rồi đổi tên: dev server của site đang theo dõi file không đọc phải bản ghi dở.
        tmp_path = self.out_path.with_name(self.out_path.name + ".tmp")
        tmp_path.write_text(text, encoding="utf-8")
        os.replace(tmp_path, self.out_path)
