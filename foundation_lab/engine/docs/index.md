---
schema_version: 1
department: engine
page_id: overview
title: Sổ tay Vận hành
description: Hiểu một bước huấn luyện, lựa chọn tài nguyên và đọc phép đo của engine SAI.
order: 0
kind: overview
updated: "2026-10-25"
tags: [vận hành, huấn luyện, bộ nhớ]
sources:
  - foundation_lab/engine/__init__.py
  - foundation_lab/engine/loop.py
  - web/site/src/lib/lab/workbench.ts
---

## Phòng này giúp giải quyết việc gì?

Vận hành trả lời câu hỏi: **thực hiện một thuật toán huấn luyện đã chọn bằng tài nguyên hiện có như thế nào?** Nó quản lý thiết bị, precision, compile, chia microbatch, tích luỹ gradient, cập nhật optimizer, vòng epoch và phép đo tài nguyên. Recipe chọn dữ liệu, objective, mẫu số chuẩn hoá loss và lịch học; engine thực thi những lựa chọn đó.

Một lượt train đi theo luồng:

```text
Batch từ phòng Dữ liệu
  → chuyển sang device
  → chia microbatch
  → forward → tính loss theo chunk → backward
  → đủ số batch tích luỹ: clip → optimizer → scheduler
  → log → cuối epoch đánh giá val → lưu best/checkpoint
```

## Trạng thái hiện tại

Snapshot ngày 25/10/2026: engine đã được pipeline huấn luyện sử dụng. Trang web Vận hành vẫn là **phòng đang được xây dựng**; `bench-step` chỉ là lệnh dự kiến, chưa phải lệnh CLI đã có. Sổ tay có thể đọc độc lập ngay cả khi phòng chưa có Control center.

Sổ tay không tự khởi chạy train hay profiler. Các lệnh minh hoạ trong bài là nội dung để đọc và sao chép; chưa được chạy trong phiên soạn tài liệu này.

## Nên đọc theo câu hỏi nào?

| Bạn đang muốn hiểu | Bài phù hợp |
| --- | --- |
| Batch, microbatch và token budget khác nhau thế nào? | [Batch và chuẩn hoá loss](batches-and-loss.md) |
| Vì sao CUDA, MPS và CPU có cách chạy khác nhau? | [Device, AMP và compile](runtime-and-precision.md) |
| Learning rate, weight decay và clip tác động gì? | [Optimizer và lịch học](optimizer-and-schedule.md) |
| VRAM hoặc TFLOPS trong log có nghĩa gì? | [Đọc profiling](profiling.md) |
| Pipeline chuẩn đang sử dụng giá trị nào? | [Mặc định hiện tại](current-defaults.md) |
| Cần làm gì trước một thử nghiệm vận hành? | [Quy trình sử dụng](workflow.md) |
| OOM, loss và tốc độ đang gây khó hiểu? | [Câu hỏi thường gặp](faq.md) |

## Ba điều cần nhớ

1. Giảm microbatch chủ yếu giảm lượng activation cùng tồn tại; nó không làm model hay trạng thái AdamW nhỏ đi.
2. Tốc độ chỉ có ý nghĩa khi biết model, shape, padding, dtype, device và phạm vi thời gian được đo.
3. Mốc so sánh phải lấy từ experiment đã resolve ở code hiện tại. Các bảng trong sổ tay là snapshot kiến thức, không phải nguồn mặc định cho form.
