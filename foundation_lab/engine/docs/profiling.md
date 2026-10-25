---
schema_version: 1
department: engine
page_id: profiling
title: Đọc VRAM, tốc độ và kernel profile
description: Hiểu phạm vi đo của log vận hành và tránh diễn giải quá mức các con số hiệu năng.
order: 40
kind: interpretation
updated: "2026-10-25"
tags: [profiling, vram, tflops, tốc độ]
sources:
  - foundation_lab/engine/profiling.py
  - foundation_lab/engine/loop.py
---

## Đo gì trước khi kết luận nhanh hơn?

Một con số thời gian cần đi kèm model, device, precision, shape `B × T`, tỷ lệ padding, objective, loss chunk, microbatch, accumulation, trạng thái compile và phạm vi đo. Khi các yếu tố này khác nhau, chênh lệch có thể do lượng việc khác nhau.

## Các mốc VRAM

`profile_vram=true` ghi mốc trong 20 batch đầu của epoch đầu đang chạy nếu device là CUDA. Các mốc gồm `start`, `h2d`, `forward`, `loss`, `backward`, `update`, `released`.

| Ký hiệu | Ý nghĩa |
| --- | --- |
| Allocated (`A`) | Bộ nhớ tensor do allocator đang sử dụng |
| Reserved (`R`) | Bộ nhớ allocator giữ lại, gồm phần chưa cấp cho tensor sống |
| Peak allocated/reserved | Đỉnh từ lần reset thống kê đỉnh tương ứng |
| GiB | Đơn vị nhị phân: 1 GiB bằng 1.024³ byte |

Reserved cao hơn allocated không tự chứng minh rò rỉ bộ nhớ. `released` là mốc sau khi xóa biến của batch, không phải toàn bộ model, optimizer và allocator đã được giải phóng. Peak của microbatch được engine chọn theo allocated cao nhất cho pha; nó không phải ảnh chụp đồng thời mọi microbatch.

Nếu forward là pha nặng, chú ý shape, padding và microbatch. Nếu loss nặng, xem vocab, head và `loss_chunk_size`. Nếu update tăng mạnh, nhớ optimizer có trạng thái riêng. Đây là hướng đọc log, chưa phải kết luận nguyên nhân từ một con số duy nhất.

## TFLOPS là ước tính theo công thức

Engine ước tính chi phí mỗi token-slot từ số tham số cùng số layer, context tối đa và chiều rộng model. Mặc định vòng lặp đo các logical batch có chỉ số 500–999 sau warmup 500 batch, trong epoch đầu còn chạy; tên “step” trong bản log hiệu năng này là batch của vòng lặp, cần phân biệt với `optimizer_step` trong metrics.

Tốc độ token-slot/s tính trên kích thước tensor `B × T`, có cả padding. TFLOPS được suy ra từ công thức FLOPs model và thời gian đo; nó không phải số phần cứng trực tiếp báo, cũng không tự cho biết token hữu ích/s. Ước tính thời gian toàn bộ train còn bỏ qua các biến động như val/test, ghi checkpoint và thay đổi độ dài giữa batch.

Run ngắn chưa tới vùng đo có thể không in báo cáo TFLOPS. Resume giữa epoch chủ động tắt phép đo benchmark đó để không ghép vùng thời gian không liên tục.

## Kernel profile hiện có

`profile_kernels=true` bật profiler tại batch có chỉ số 1.000 trong epoch đầu đang chạy. Báo cáo gom operator theo shape, lọc đóng góp nhỏ, giữ tối đa 20 dòng và lưu bản tóm tắt JSON trong thư mục stage. Đây không phải timeline toàn bộ lượt train.

`self` là thời gian quy cho chính operator; `total` có thể bao gồm lời gọi con. Không cộng mọi cột total để suy ra tổng thời gian vì có thể tính lặp. Profiler tự thêm chi phí thu thập: so tốc độ thông thường nên tách run đo profile khỏi kết luận thông lượng ổn định.

Phòng `bench-step` trên web chưa được triển khai. Các số đo vừa mô tả hiện xuất phát từ vòng huấn luyện. Đọc [quy trình](workflow.md) để dùng chúng với đúng quyền chạy và giới hạn tài nguyên.
