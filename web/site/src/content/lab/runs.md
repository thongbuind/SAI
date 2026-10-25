---
title: Thư mục run
description: Mỗi lần chạy ghi ra một thư mục tự đủ, truy được về code, config, dữ liệu và môi trường đã tạo ra nó, và chạy tiếp được khi bị ngắt.
group: Bắt đầu
order: 13
---

## Cấu trúc

Mỗi lần `run` tạo một thư mục `runs/<tên experiment>/<thời gian>/`:

```text
runs/SAI_375M/20261023-091500/
├── resolved_config.yaml   # config đã resolve; chạy lại chỉ cần file này
├── lineage.json           # nguồn gốc của run, xem bên dưới
├── invocations.jsonl      # mỗi lần gọi run/resume: argv, stage, init, commit git
├── metrics.jsonl          # metric theo thời gian của mọi stage
└── stages/
    └── pretrain/
        ├── best.pt        # trọng số tốt nhất theo val loss
        ├── last.ckpt.pt   # checkpoint để chạy tiếp
        └── done.json      # có file này nghĩa là stage đã xong, kèm kết quả
```

Tác vụ (`infer`, `compare-kernel`) ghi vào `runs/<lệnh>/<thời gian>/`, thay `stages/` bằng `events.jsonl` (tiến trình ghi dần) và `result.json` (kết quả khi xong).

Đổi thư mục gốc bằng `--runs-root`, đặt tên thư mục run bằng `--run-id` thay vì theo thời gian.

## lineage.json

Ghi một lần lúc tạo run:

| Khoá | Nội dung |
| --- | --- |
| `created_at` | Thời điểm tạo run |
| `config_hash` | Hash của config đã resolve, không phụ thuộc thứ tự khoá |
| `lab_version` | Phiên bản lab, xem [Phiên bản](../versioning/#phiên-bản-lab) |
| `git` | Commit, branch, và working tree có thay đổi chưa commit hay không |
| `env` | Phiên bản Python, hệ điều hành, PyTorch, CUDA, tên GPU |
| `tokenizer` | Dấu vân tay của `SAI/tokenizer.model` |
| `data` | Dấu vân tay của từng file dữ liệu mỗi stage đọc |

Dấu vân tay gồm đường dẫn, kích thước và thời điểm sửa; file nhỏ (manifest...) có thêm SHA-256 nội dung.

`lineage.json` chỉ ghi lúc tạo run; `invocations.jsonl` thêm một dòng cho mỗi lần gọi, nên biết được code nào đã chạy ở từng lần resume.

## Chạy tiếp với --resume

Khi bị ngắt, chạy lại với thư mục run. Lab đọc `resolved_config.yaml` của run đó, bỏ qua stage đã có `done.json`, và tiếp tục stage dở dang từ `last.ckpt.pt`, kể cả giữa epoch.

```bash
python -m foundation_lab run --resume runs/SAI_375M/20261021-120000
```

Checkpoint lưu cả trạng thái RNG, sampler và GradScaler, nên run chạy tiếp cho kết quả như run không bị ngắt.

> Khi resume không truyền file experiment hay `--set`: run phải chạy tiếp đúng config đã ghi. Muốn đổi config thì tạo run mới.
