---
schema_version: 1
department: engine
page_id: runtime-and-precision
title: Device, AMP và compile
description: Hiểu cách engine chọn thiết bị và precision, cùng đánh đổi của việc biên dịch forward.
order: 20
kind: concept
updated: "2026-10-25"
tags: [device, amp, precision, compile]
sources:
  - foundation_lab/engine/runtime.py
  - foundation_lab/core/experiment.py
  - foundation_lab/pipeline.py
---

## Chọn nơi chạy

`runtime.device` nhận `auto`, `cuda`, `mps`, `cpu`. `auto` thử CUDA trước, rồi MPS, cuối cùng CPU. Chọn tường minh CUDA hoặc MPS mà thiết bị không sẵn có sẽ báo lỗi; nó không âm thầm đổi sang CPU.

| Thiết bị | Vai trò | Điều cần xét |
| --- | --- | --- |
| CUDA | Chạy trên GPU NVIDIA tương thích | VRAM, kernel, phiên bản PyTorch/CUDA |
| MPS | Backend GPU trên máy Apple hỗ trợ | Bộ nhớ dùng chung và khả năng hỗ trợ toán tử |
| CPU | Chạy trên bộ xử lý chính | RAM, thời gian tính và số worker |

`auto` chọn backend, không tự tìm cấu hình batch phù hợp với lượng RAM của máy. Cùng tên model cũng không có nghĩa footprint giống nhau giữa train và infer: train còn giữ activation, gradient và trạng thái optimizer.

## AMP trong engine hiện tại

AMP là mixed precision: một số phép toán dùng precision thấp hơn, còn thao tác cần độ ổn định có thể dùng precision cao hơn. `autocast` quyết định theo từng phép toán; đây không phải chuyển tất cả tensor sang cùng một dtype. `GradScaler` giúp xử lý thang gradient khi dùng FP16. [Hướng dẫn AMP của PyTorch](https://docs.pytorch.org/docs/stable/notes/amp_examples.html).

Chính sách của engine tại snapshot 25/10/2026:

| Device | Autocast | GradScaler |
| --- | --- | --- |
| CUDA hỗ trợ BF16 | BF16 | Tắt |
| CUDA không hỗ trợ BF16 | FP16 | Bật |
| MPS hoặc CPU | Tắt | Tắt |

`RuntimeSpec` chưa có field chọn `dtype` hoặc bật/tắt AMP riêng. Modal phải mô tả chính sách này, không vẽ selector BF16/FP16 như một khả năng đã có. `amp_dtype` nội bộ có thể mang giá trị FP16 khi autocast bị tắt; điều đó không có nghĩa MPS/CPU đang thực thi mixed precision.

## Compile đang biên dịch phần nào?

`runtime.compile=true` bọc `model.forward_features` bằng `torch.compile(dynamic=True)`. Vòng Python quản lý microbatch, optimizer và lưu file vẫn có trách nhiệm riêng. Không nên gọi toàn bộ pipeline là một graph đã compile.

`limited_max_autotune=true` thêm tập tuỳ chọn ATEN/TRITON, search space `DEFAULT`, fusion phần epilogue và tắt CUDA Graph trong cấu hình đó. Nếu `compile=false`, nhánh biên dịch không được dùng. Khả năng chạy và lợi ích phải được đo trên backend thực tế.

Đánh đổi thường gặp: quá trình chuẩn bị graph có chi phí đầu; shape thay đổi có thể làm khác hành vi cache/compile; lợi ích chỉ nên kết luận sau khi tách thời gian khởi động khỏi vùng chạy ổn định. Dòng `dynamic=True` không phải cam kết không bao giờ có compile lại.

## Seed và worker

`runtime.seed=null` nghĩa là pipeline không đặt seed theo cấu hình. Khi có số nguyên, stage ở vị trí `i` dùng `seed + i`. Vị trí này tính trong pipeline, không chỉ trong tập stage được chọn chạy. `runtime.num_workers` là worker của DataLoader; tăng worker có thể giúp đọc dữ liệu nhưng cũng dùng thêm RAM và tiến trình.

Cùng seed không bảo đảm kết quả bitwise giống nhau khi đổi thiết bị, kernel hoặc phiên bản thư viện. PyTorch nêu rõ giới hạn tái lập giữa nền tảng và phiên bản. [Reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html).

Đọc [mặc định](current-defaults.md) để biết snapshot pipeline và [quy trình](workflow.md) để chuẩn bị thử nghiệm có mốc so sánh.
