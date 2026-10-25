---
schema_version: 1
department: arch
page_id: kernel-guide
title: "Thực hành so attention kernel"
description: "Chọn cấu hình trên web hoặc CLI và giữ phép so có baseline đúng, quy mô phù hợp."
order: 60
kind: guide
updated: "2026-10-25"
tags: [hướng dẫn, compare-kernel, CLI, web, baseline]
sources:
  - foundation_lab/eval/compare.py
  - foundation_lab/cli.py
  - web/site/src/data/lab_fields.json
  - web/site/src/components/lab/TaskBench.astro
  - web/site/src/lib/lab/views/compareKernel.ts
---

## Trước khi bấm Chạy

Mục tiêu dễ bắt đầu là so `eager` với kernel trong config của một model modular. Công cụ tạo Q, K, V và gradient đầu ra ngẫu nhiên theo shape của model; không nạp checkpoint hay train toàn model. Tuy vậy, kích thước tensor tăng theo batch, số head và độ dài chuỗi; bản eager tạo ma trận score cỡ `B × H × T × T`.

Máy ít RAM nên bắt đầu một model, `float32`, CPU và chuỗi ngắn. Đây là hướng dẫn cấu hình, không phải một benchmark đã được chạy. Khi chạy thực tế, làm theo quy định tài nguyên của workspace.

## Luồng trên web hiện có

1. Mở phòng **Kiến trúc** tại `/lab/workbench/arch/` trên bản server Lab.
2. Trong **Control center**, chọn **Kernel ứng viên** và ít nhất một **Mô hình** thuộc họ `SAI_modular`.
3. Chọn batch, độ dài, kiểu số, thiết bị, causal mask và số lần lặp. Kiểm tra lại số model được chọn.
4. Bấm **Chạy**; hộp xác nhận hiện workload chạy trên máy này. Xác nhận mới gửi tác vụ.
5. Đọc từng dòng kết quả khi model tương ứng hoàn tất. Có thể mở lịch sử, hủy tác vụ đang chờ/chạy, hoặc sao chép lệnh chạy lại khi run đã được tạo.

Bạn có thể đọc Sổ tay để hiểu các trường trước khi quyết định cấu hình phép so.

## Ý nghĩa từng trường

| Trường | Cách dùng |
| --- | --- |
| `candidate` | Tên kernel đã đăng ký; hiện có `eager`, `sdpa` |
| `models` | Lấy `H`, `Hkv`, `D` và kernel mốc từ từng config; model `SAI` chính thức không có trường chọn kernel |
| `batch_size` | Số chuỗi tensor ngẫu nhiên `B`; không phải số model |
| `seq_len` | Độ dài `T` của Q và K; không phải độ dài dữ liệu train được đọc |
| `dtype` | `float32`, `float16`, `bfloat16`; khả năng thực thi còn phụ thuộc backend |
| `device` | `cpu`, `cuda`, `mps`, `auto`; chọn rõ thiết bị khi muốn so số đo |
| `causal` | Bật để mỗi query chỉ nhìn hiện tại/quá khứ trong phép đo vuông |
| `repeats` | Số lượt đo lấy trung vị sau một lần khởi động mỗi bên |

Mốc không phải một ô tự gõ: Lab lấy `attention_kernel` trong config đã resolve của từng model. Khi dùng một resolved spec cũ, mốc là snapshot đã nhúng trong file đó; nó phục vụ tái lập run cũ, không tự cập nhật theo config mới.

## Ví dụ CLI có thật

Lệnh sau chỉ minh họa phép thử quy mô nhỏ, chưa được thực thi trong lúc viết sổ tay:

```bash
python -m foundation_lab compare-kernel --set candidate=eager --set 'models=[SAI_modular_35M]' --set batch_size=1 --set seq_len=32 --set device=cpu --set dtype=float32 --set repeats=3
```

Kết quả được ghi trong `runs/compare-kernel/<run_id>/`. Giao diện cung cấp lệnh chạy lại trỏ tới `resolved_config.yaml` đúng của run; giữ file này cùng `result.json`, sự kiện và lineage khi đối chiếu. Không tự thay `<run_id>` bằng một ID giả rồi chạy.

Đọc tiếp [Kết quả và giới hạn phép đo](results.md). Hướng dẫn phản ánh CLI và web ngày 25/10/2026; schema hiện hành quyết định trường hợp lệ.
