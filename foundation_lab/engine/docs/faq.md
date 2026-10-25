---
schema_version: 1
department: engine
page_id: faq
title: Câu hỏi thường gặp về vận hành
description: Giải thích các nhầm lẫn thường gặp về OOM, batch, precision, loss và thông lượng.
order: 70
kind: faq
updated: "2026-10-25"
tags: [faq, oom, loss, hiệu năng]
sources:
  - foundation_lab/engine/step.py
  - foundation_lab/engine/runtime.py
  - foundation_lab/engine/loop.py
  - foundation_lab/engine/profiling.py
  - foundation_lab/core/configs/base.yaml
---

## Đã giảm microbatch mà vẫn hết bộ nhớ?

Microbatch chủ yếu tác động phần công việc theo batch. Trọng số, gradient, optimizer state và một mẫu dài vẫn phải tồn tại. Phép chia cân bằng cũng không làm `max_microbatch_tokens` thành giới hạn bộ nhớ cứng. Xem pha nào gây đỉnh qua [profiling](profiling.md); không liên tục giảm một số mà bỏ qua shape và model.

## Tăng accumulation có làm một mẫu chạy nhẹ hơn không?

Không trực tiếp. Nó trì hoãn optimizer step để gom nhiều logical batch. Muốn giảm lượng activation cùng lúc, xem microbatch hoặc logical batch. Nếu giảm batch và tăng accumulation để giữ effective batch gần bằng, vẫn phải kiểm tra sampler, mẫu số chuẩn hoá, batch cuối và tổng số step.

## Loss không đổi thì có thể coi mọi cấu hình tương đương?

Một vài chữ số loss giống nhau chưa chứng minh gradient, trạng thái optimizer hay đường học giống nhau. Với tối ưu chỉ nhằm hiệu năng, cần kiểm chứng phù hợp trên model tí hon trước khi dùng kết quả. Khi thay dtype hoặc kernel, dùng dung sai số được xác định theo phạm vi thử nghiệm, không hứa bitwise bằng nhau.

## Vì sao log train loss khác scalar dùng backward?

Log chia tổng loss thực tế cho tổng token chịu loss thực tế. Backward chia mỗi loss sum cho mẫu số cố định do recipe cung cấp và cho accumulation. Hai số phục vụ hai mục đích. Sau resume giữa epoch, trung bình train log chỉ bao phủ phần batch mới xử lý. Xem [batch và loss](batches-and-loss.md).

## `auto` có tự chọn batch phù hợp không?

Không. Nó chỉ chọn CUDA → MPS → CPU. Nó không benchmark RAM, không giảm model, không tự chuyển sang checkpoint nhỏ hơn và không tự sửa cấu hình fused/compile theo dung lượng bộ nhớ.

## Bật compile mà lượt đầu chậm có bất thường không?

Khởi động compile có chi phí và không cùng ý nghĩa với thời gian chạy ổn định. Cần nhìn phạm vi đo và shape trước khi kết luận. Sổ tay không bảo đảm compile nhanh hơn trên mọi máy.

## Không có VRAM hoặc TFLOPS report nghĩa là lỗi?

VRAM profile chỉ chạy ở CUDA và cửa sổ batch được quy định. Báo cáo TFLOPS cần đi tới vùng đo; run quá ngắn hoặc resume giữa epoch có thể không tạo nó. Phân biệt thiếu phép đo với giá trị bằng 0.

## Frozen block có biến mất khỏi mô hình không?

Không. Nó vẫn được dùng trong forward, có trọng số và vẫn ảnh hưởng đầu ra. Freeze chỉ bỏ cập nhật các tham số đó. Với SFT, optimizer được dựng trên phần trainable.

## Có thể bấm Sổ tay để chạy thử ngay không?

Nút Sổ tay chỉ mở nội dung. Nó không train, không nạp checkpoint và không tự bật profiler. Trang Vận hành hiện vẫn đang được xây dựng; `bench-step` là dự kiến. Khi có runner sau này, thao tác chạy phải đi qua luồng thực thi riêng và thông tin tài nguyên rõ ràng.
