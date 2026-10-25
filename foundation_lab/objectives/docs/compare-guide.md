---
schema_version: 1
department: objectives
page_id: compare-guide
title: "Hướng dẫn so objective trên web và CLI"
description: "Chọn experiment, stage và chunk để đối chiếu với mốc tự lấy từ cấu hình hiện tại."
order: 40
kind: guide
updated: "2026-10-25"
tags: [hướng dẫn, web, CLI, compare-objective, baseline]
sources:
  - foundation_lab/eval/compare.py
  - foundation_lab/cli.py
  - foundation_lab/recipes/base.py
  - web/site/src/data/lab_fields.json
  - web/site/src/components/lab/TaskBench.astro
  - web/site/src/lib/lab/views/compareObjective.ts
---

## Chọn một câu hỏi cụ thể

Ví dụ hữu ích với registry hiện có: “Nếu giảm chunk của `lm_ce`, tổng loss và gradient còn khớp mốc, và thời gian thay đổi ra sao?” Chọn trước quy mô thử; trên máy ít RAM, chỉ một experiment và số token nhỏ để làm quen, sau đó mới xin chạy workload gần thực tế theo quy định tài nguyên.

## Luồng trên web

1. Mở `/lab/workbench/objectives/` trên bản server Lab.
2. Chọn **Objective ứng viên**; snapshot hiện có `lm_ce`.
3. Chọn **Experiment** và **Stage làm mốc** có trong các experiment đã chọn.
4. Điền **Chunk ứng viên** hoặc để trống dùng chunk mốc. Chọn **Số token (N)** hoặc để trống dùng ngân sách stage làm quy mô tensor.
5. Chọn dtype, thiết bị, số lần lặp; bấm **Chạy** và đọc hộp xác nhận workload trước khi gửi.
6. Đọc từng dòng theo experiment. Có thể chọn lịch sử và lấy lệnh chạy lại từ config đã resolve.

Tác vụ tạo tensor ngẫu nhiên; không đọc dữ liệu train, không load checkpoint và không tự thay `params.objective` trong experiment.

## Mốc được hình thành thế nào?

```text
experiment → resolve model + pipeline + stage_defaults
           → chọn stage
           → dựng dataclass params của recipe để bổ sung mặc định
           → objective, loss_chunk_size, max_microbatch_tokens
model      → d_model, vocab_size
```

Vì vậy stage không ghi `objective` vẫn có mốc từ mặc định recipe. Hãy xem mốc của từng dòng kết quả thay vì mặc định rằng nó luôn là `lm_ce`, dù đó là lựa chọn ở snapshot hiện tại. Tên experiment còn là nhãn từng dòng; cùng cỡ model nhưng stage khác vẫn có thể có params khác.

## Tra nhanh các trường

| Trường | Ý nghĩa |
| --- | --- |
| `candidate` | Tên hàm trong `OBJECTIVES` |
| `experiments` | Danh sách experiment; resolver cũng hỗ trợ mapping nhãn tới tên/mốc |
| `stage` | Stage dùng lấy mốc; phải tồn tại và có recipe chọn objective qua params |
| `candidate_chunk_size` | `null`/trống nghĩa bằng chunk mốc của từng dòng |
| `tokens` | `null`/trống nghĩa `N = max_microbatch_tokens` mốc |
| `dtype`, `device` | Điều kiện tạo tensor và chạy phép đo |
| `repeats` | Số lượt đo lấy trung vị sau khởi động |

## Ví dụ CLI

Ví dụ quy mô nhỏ, chưa chạy trong phiên soạn nội dung:

```bash
python -m foundation_lab compare-objective --set candidate=lm_ce --set 'experiments=[SAI_35M]' --set stage=pretrain --set candidate_chunk_size=16 --set tokens=32 --set device=cpu --set dtype=float32 --set repeats=3
```

Kết quả vào `runs/compare-objective/<run_id>/`. Dùng lệnh chạy lại do web cung cấp với `resolved_config.yaml` để tái lập mốc cũ. Muốn so với mặc định **hiện tại**, tạo spec mới theo tên experiment; file đã resolve cũ giữ mốc lúc nó được tạo.

CLI hiện có mã thoát 0 khi xong, 2 khi lỗi cấu hình, 1 khi lỗi chạy. Không coi “xong” là objective đã đạt tiêu chuẩn: cần đọc [kết quả loss và gradient](results.md).

Hướng dẫn đối chiếu source ngày 25/10/2026; schema đang chạy quyết định trường hợp lệ và registry quyết định lựa chọn.
