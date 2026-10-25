---
schema_version: 1
department: eval
page_id: overview
title: "Sổ tay phòng Đánh giá"
description: "Hiểu CE loss, chất lượng văn bản và cách thiết kế phép đánh giá công bằng cho SAI."
order: 0
kind: overview
updated: "2026-10-25"
tags: [đánh giá, validation, test, chất lượng, bắt đầu]
sources:
  - foundation_lab/eval/__init__.py
  - foundation_lab/eval/lm_loss.py
  - foundation_lab/eval/text.py
  - foundation_lab/eval/compare.py
  - web/site/src/lib/lab/workbench.ts
---

## Đánh giá trả lời câu hỏi nào?

Đánh giá là đo một hệ thống trong điều kiện đã định trước để đưa ra quyết định. Với Lab, cần phân biệt: phép tính có đúng không, mô hình dự đoán token tốt đến đâu, văn bản sinh có hữu ích không và chi phí chạy là bao nhiêu. Một con số không trả lời đồng thời cả bốn câu hỏi.

| Câu hỏi | Công cụ/thước đo hiện có | Giới hạn |
| --- | --- | --- |
| Kernel/objective có khớp mốc? | `compare-kernel`, `compare-objective` | Tensor ngẫu nhiên; kiểm chứng số học |
| Checkpoint dự đoán dữ liệu giữ lại tốt không? | CE val/test được recipe gọi | Phụ thuộc dữ liệu, split, mask và checkpoint |
| Văn bản có lặp, dài, bị cắt không? | `distinct_n` và số đo ở phòng Suy luận | Không tự chấm đúng kiến thức/hữu ích |
| Chạy nhanh hoặc ít bộ nhớ hơn? | Timing, bộ đếm CUDA theo công cụ | Chỉ đại diện workload/thiết bị đã đo |

Các helper so sánh nằm trong package `eval`, nhưng tác vụ web của chúng nằm ở phòng Kiến trúc và Objective. Tên thư mục source không đồng nghĩa vị trí mọi nút trên giao diện.

## Trạng thái phòng web

Ngày 25/10/2026, `/lab/workbench/eval/` là **phòng đang được xây dựng**. `eval-loss` là tên lệnh dự kiến trong cấu hình giao diện, chưa là CLI đã đăng ký. Phòng này chưa có form chọn checkpoint hay nút chạy eval độc lập. Sổ tay vẫn có thể mở để học và đọc quy trình hiện tại.

Đánh giá val/test hiện diễn ra trong recipe huấn luyện; phép xem các cách sinh văn bản dùng phòng [Suy luận](../../inference/docs/index.md). Không chạy lại một pipeline train chỉ để xem một con số đã có trong run.

## Chọn bài cần đọc

| Nhu cầu | Bài |
| --- | --- |
| Hiểu loss và perplexity | [CE trên dữ liệu giữ lại](lm-loss.md) |
| Hiểu distinct-2, độ dài, latency | [Thước đo văn bản và chi phí](text-metrics.md) |
| So checkpoint/biến thể cho công bằng | [Thiết kế phép đánh giá](evaluation-protocol.md) |
| Tìm kết quả đang có và hiểu quy trình thực tế | [Cách dùng hiện tại](current-workflow.md) |
| Đọc train/val/test và các giới hạn đã biết | [Diễn giải kết quả](results.md) |
| Tránh các kết luận quá mức | [Câu hỏi thường gặp](faq.md) |

## Nguyên tắc đọc số đo

Luôn gắn kết quả với checkpoint, dữ liệu, tokenizer, mask, code version và cấu hình đã resolve. Với phép so mới, mốc phải đại diện cấu hình hiện tại của model đang dùng; với run cũ, giữ đúng snapshot mốc của run. Không thay bản ghi lịch sử bằng mặc định mới.

Nội dung này dựa trên working tree ngày 25/10/2026, không phải kết quả đánh giá mới được chạy trong phiên soạn sổ tay.
