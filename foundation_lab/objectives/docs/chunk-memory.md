---
schema_version: 1
department: objectives
page_id: chunk-memory
title: "Chunk, logits và bộ nhớ"
description: "Hiểu vì sao loss chia chunk, activation checkpointing làm gì và cách chọn phép thử công bằng."
order: 30
kind: concept
updated: "2026-10-25"
tags: [chunk, bộ nhớ, logits, checkpointing, hiệu năng]
sources:
  - foundation_lab/objectives/lm_ce.py
  - foundation_lab/eval/compare.py
  - foundation_lab/engine/step.py
  - tests/test_objectives.py
  - tests/test_compare.py
---

## Vì sao logits có thể lớn?

Nếu có `N` vị trí và từ vựng `V`, logits đầy đủ có `N × V` phần tử. Hidden chỉ có `N × d_model` phần tử, mà `V` thường lớn hơn `d_model`. Chỉ riêng tensor logits float32 cần khoảng `4 × N × V` byte, chưa tính gradient, softmax và workspace.

Ví dụ toán học: `N=1024`, `V=10000` tạo khoảng 39 MiB logits float32. Đây là ước lượng một tensor, không phải mức RAM của cả lệnh. Benchmark Objective tạo cả hidden và ma trận head; tên experiment lớn không có nghĩa nó train model đầy đủ, nhưng cũng không có nghĩa phép đo luôn rẻ.

## lm_ce chia theo hàng

Implementation ngày 25/10/2026 duyệt các đoạn tối đa `chunk_size` hàng của hidden, tạo logits cho đoạn, tính CE từng token, nhân token weight rồi cộng vào tổng scalar. Toàn bộ từ vựng vẫn được xét cho từng token; chunk không cắt bớt vocabulary, không làm mất token và không đổi thứ tự target.

```text
N hàng hidden
  → chunk 1 → logits → tổng loss 1
  → chunk 2 → logits → tổng loss 2
  → ...
  → cộng các tổng → một loss_sum
```

Khi gradient được bật, từng chunk dùng activation checkpointing: giữ ít activation hơn và tính lại phần cần thiết khi backward. Từ “checkpoint” ở đây là kỹ thuật tính lại đồ thị, **khác** file trọng số `.pt` để lưu model. Khi `torch.no_grad()` được dùng trong eval, đường checkpoint activation không cần chạy.

## Đánh đổi kích thước chunk

| Chunk | Điều thường thay đổi | Điều cần kiểm chứng |
| --- | --- | --- |
| Nhỏ hơn | Logits tạm nhỏ hơn; nhiều lần gọi phép tính hơn | Peak và thời gian thực tế; sai khác cộng số thực |
| Lớn hơn | Ít chunk hơn; tensor logits tạm lớn hơn | Không vượt tài nguyên; chưa chắc nhanh hơn mọi backend |
| Lớn hơn `N` | Một chunk cho toàn bộ hàng | Không có lợi giảm tensor logits theo hàng |

Mục tiêu khi đổi chunk của cùng `lm_ce` là giữ tổng loss và gradient tương đương trong dung sai. Thứ tự cộng số thực có thể gây sai khác nhỏ. Test hiện có đối chiếu nhiều chunk và so với cách tạo logits đầy đủ trên tensor tí hon; sổ tay không khẳng định mọi kích thước/thiết bị đã được test.

## Phân biệt ba cách chia

`chunk_size` chia bước chiếu hidden sang logits. `max_microbatch_tokens` điều khiển engine chia batch cho forward model. `accumulation_steps` tích luỹ gradient qua logical batch trước optimizer step. Thay một giá trị không thay thế hoàn toàn hai giá trị còn lại.

Trong `compare-objective`, để trống **Chunk ứng viên** nghĩa dùng `loss_chunk_size` mốc riêng của từng experiment; để trống **Số token** lấy `max_microbatch_tokens` stage làm số hàng ngẫu nhiên `N`. Đây là quy ước workload của phép đo, không có padding hay sampler train thực sự trong benchmark đó.

Giữ cùng `N`, `d_model`, `V`, dtype và thiết bị khi so các chunk. Để đọc số đo, xem [Kết quả](results.md).
