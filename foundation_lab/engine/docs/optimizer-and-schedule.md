---
schema_version: 1
department: engine
page_id: optimizer-and-schedule
title: AdamW, lịch học và gradient clipping
description: Phân biệt learning rate, weight decay, lịch warmup và đóng băng tham số trong một bước train.
order: 30
kind: concept
updated: "2026-10-25"
tags: [adamw, learning rate, warmup, clipping, freeze]
sources:
  - foundation_lab/engine/optim.py
  - foundation_lab/engine/step.py
  - foundation_lab/engine/freeze.py
  - foundation_lab/recipes/base.py
  - foundation_lab/recipes/causal_lm.py
  - foundation_lab/recipes/sft.py
---

## Những tham số này không thay thế nhau

| Tham số | Ý nghĩa | Đánh đổi cần quan sát |
| --- | --- | --- |
| `learning_rate` | Quy mô cập nhật cơ sở | Quá lớn có thể mất ổn định; quá nhỏ học chậm |
| `weight_decay` | Co trọng số trong AdamW | Điều chuẩn mạnh/yếu ảnh hưởng nghiệm học được |
| `max_grad_norm` | Giới hạn norm gradient trước update | Ngăn cập nhật đột ngột; không sửa nguyên nhân loss sai |
| `warmup_ratio` | Tỷ lệ bước đầu tăng LR | Ổn định giai đoạn đầu, đổi lượng học sớm |
| `freeze_blocks` | Block không nhận cập nhật gradient | Giảm tham số được tối ưu, hạn chế khả năng thích nghi |

AdamW duy trì thống kê moment của gradient và tách weight decay khỏi các moment đó. Engine dùng `torch.optim.AdamW`; hàm dựng không chia riêng nhóm “không decay” cho norm hoặc embedding: mọi tham số được truyền vào đều nhận cùng `weight_decay`. [Tài liệu AdamW](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html).

## Lịch đang được đăng ký

Registry `LR_SCHEDULES` hiện đăng ký `warmup_stable_decay`. Lịch là multiplier nhân với learning rate cơ sở:

1. Trong `warmup_steps`: tăng tuyến tính từ 0.
2. Sau warmup và trước 40% tổng step: multiplier 1.
3. Từ 40% đến 70%: giảm tuyến tính từ 1 xuống 0,1.
4. Phần còn lại: giữ multiplier 0,1.

Recipe tính `warmup_steps = int(total_steps × warmup_ratio)` và `total_steps = ceil(batches_per_epoch / accumulation_steps) × epochs`. Đây là số **optimizer step**, không phải token hoặc microbatch. Các tỷ lệ warmup chuẩn hiện tại 0,1 và 0,15 nằm trước mốc 40%; không suy diễn một warmup tùy ý rất dài vẫn có đầy đủ bốn pha như sơ đồ trên.

Ví dụ minh hoạ: LR cơ sở `2e-4` và 100 optimizer step với warmup 10 step cho vùng giữ LR trước mốc 40, giảm tới `2e-5` ở mốc 70. Giá trị minh hoạ giúp đọc lịch, không được dùng để đặt mốc form.

## Thứ tự một lần cập nhật

```text
unscale gradient → clip norm → optimizer step
  → update scaler → xóa gradient → scheduler step
```

Clip phải hiểu trên gradient đã unscale để ngưỡng có đúng ý nghĩa. Lab gọi clip trên tham số model, còn AdamW chỉ cập nhật những tham số đã đưa vào optimizer. Ở nhánh FP16, `GradScaler` có thể bỏ qua update khi gradient không hữu hạn; engine hiện vẫn tiến scheduler sau lời gọi scaler. Vì vậy bộ đếm bước theo lịch không luôn đồng nghĩa có một thay đổi trọng số hữu hiệu.

## Fused và freeze trong SAI

`adamw_fused` chọn implementation fused của AdamW, không đổi sang một thuật toán optimizer khác. `null` trong recipe nghĩa là chỉ bật fused khi runtime dùng CUDA AMP; `true`/`false` là lựa chọn tường minh. Khả năng backend vẫn phải được kiểm tra trên môi trường thật.

Recipe `causal_lm` truyền toàn bộ tham số model. Recipe `sft` đặt `requires_grad=false` cho các block trong `freeze_blocks` rồi chỉ đưa tham số còn trainable vào optimizer. Các block frozen vẫn nằm trong forward và vẫn chiếm bộ nhớ trọng số; freeze không xoá chúng khỏi mạng.

Đọc [snapshot cấu hình](current-defaults.md) trước khi thay đổi một tham số. Khi so lựa chọn vận hành, giữ nguyên dữ liệu, objective và mẫu số chuẩn hoá để không nhầm một thay đổi thuật toán thành tối ưu tốc độ.
