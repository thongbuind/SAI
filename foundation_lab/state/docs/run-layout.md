---
schema_version: 1
department: state
page_id: run-layout
title: Cấu trúc thư mục run
description: Phân biệt artifact của pipeline huấn luyện với artifact của tác vụ và dấu hoàn thành của mỗi loại.
order: 10
kind: reference
updated: "2026-10-25"
tags: [run, artifact, stage, file]
sources:
  - foundation_lab/state/run.py
  - foundation_lab/core/tasks.py
  - foundation_lab/cli.py
  - foundation_lab/pipeline.py
  - web/backend/lab_runs.py
---

## Run huấn luyện

Ví dụ cấu trúc, không phải một run thật đã được đọc:

```text
runs/<experiment>/<run_id>/
├── resolved_config.yaml
├── lineage.json
├── invocations.jsonl
├── metrics.jsonl
└── stages/<stage>/
    ├── best.pt
    ├── last.ckpt.pt
    └── done.json
```

| Artifact | Nội dung và thời điểm |
| --- | --- |
| `resolved_config.yaml` | Cấu hình sau khi resolve; có comment `config_hash` đầu file |
| `lineage.json` | Nguồn gốc lúc tạo run |
| `invocations.jsonl` | Thêm một dòng cho từng lần gọi CLI trên run |
| `metrics.jsonl` | Metric train/val/test ghi theo tiến độ |
| `best.pt` | State dict của trọng số tốt nhất theo val đã quan sát |
| `last.ckpt.pt` | Trạng thái huấn luyện ở lần lưu gần nhất |
| `done.json` | Stage đã trả về kết quả thành công cho pipeline |

Pipeline chỉ ghi `done.json` khi recipe chạy xong. `best.pt` hoặc `last.ckpt.pt` có thể xuất hiện trong lúc stage vẫn chạy, vì vậy không dùng chúng làm dấu “xong”. Một stage xong có thể cùng tồn tại với stage khác chưa chạy trong cùng run.

## Run tác vụ

```text
runs/<command>/<run_id>/
├── resolved_config.yaml
├── lineage.json
├── invocations.jsonl
├── events.jsonl
└── result.json
```

Tác vụ như `infer` ghi sự kiện dần và kết quả cuối theo `RunResult`. `result.json` được ghi nguyên tử qua file tạm rồi đổi tên; sự kiện `finished` được phát sau khi kết quả đã ghi. Nếu dừng sớm, có thể chỉ có vài event/record và chưa có `result.json`.

Backend tác vụ hiện suy trạng thái từ job đang quản lý và file: khi không còn job trong bộ nhớ, có `result.json` được xem là thành công; event cuối `failed` nghĩa là lỗi; còn lại là `incomplete`. Đây là quy tắc cho run tác vụ, không thay thế `done.json` của stage huấn luyện.

## Định danh và vị trí

`RunDir.create` tạo mới thư mục và không chấp nhận ghi vào thư mục đã tồn tại. `run_id` CLI mặc định dùng thời gian; web có cơ chế tên job riêng. Tên thư mục giúp tìm run, nhưng không phải fingerprint hay bằng chứng tất cả nội dung đã hoàn chỉnh.

`--runs-root` đổi nơi ghi run mới; `--run-id` đặt định danh. Không đổi tên hoặc di chuyển run cũ để làm mới lịch sử. Các đường dẫn được ghi trong config và kết quả có thể còn tham chiếu vị trí cũ.

## Cách đọc một run bằng mắt

1. Xác định đó là pipeline hay tác vụ.
2. Đọc config để biết model, stage hoặc variants.
3. Đọc lineage và invocation để biết code/môi trường ban đầu và các lần gọi tiếp.
4. Kiểm tra dấu hoàn thành đúng loại.
5. Đọc metric/result cùng artifact trọng số tương ứng.

Việc đọc metadata không cần mở tensor checkpoint. Xem [trọng số và checkpoint](checkpoints.md) khi quyết định nạp file nào.
