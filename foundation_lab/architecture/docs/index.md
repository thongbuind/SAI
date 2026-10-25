---
schema_version: 1
department: arch
page_id: overview
title: "Sổ tay phòng Kiến trúc"
description: "Hiểu cấu trúc Transformer của SAI và cách kiểm chứng attention kernel trong Lab."
order: 0
kind: overview
updated: "2026-10-25"
tags: [kiến trúc, transformer, attention, bắt đầu]
sources:
  - foundation_lab/architecture/__init__.py
  - foundation_lab/architecture/registry.py
  - foundation_lab/eval/compare.py
  - web/site/src/lib/lab/workbench.ts
---

## Phòng này giúp bạn trả lời điều gì?

Kiến trúc mô hình là cách các phép tính và tham số được tổ chức để biến token đầu vào thành dự đoán token tiếp theo. Sổ tay giải thích toàn bộ kiến trúc SAI, còn công cụ web hiện có tập trung vào một câu hỏi hẹp: **attention kernel ứng viên có cho output và gradient khớp kernel mốc, với chi phí bao nhiêu?**

Một kernel là cách thực hiện phép tính attention; nó không phải toàn bộ attention layer và không chứa trọng số học được. So kernel bằng tensor ngẫu nhiên không đo khả năng trả lời câu hỏi, không huấn luyện model và không chứng minh một kiến trúc mới tốt hơn.

## Chọn lộ trình đọc

| Bạn đang muốn biết | Bài nên đọc |
| --- | --- |
| Transformer làm gì với một câu? | [Từ token đến logits](transformer.md) |
| Query, key, value; MHA, GQA, MQA là gì? | [Attention và các lựa chọn](attention.md) |
| Vì sao thứ tự token quan trọng? | [Vị trí, RoPE và cache](position-cache.md) |
| RMSNorm, residual và SwiGLU có vai trò gì? | [Chuẩn hoá và FFN](norm-ffn.md) |
| SAI đang dùng gì, Lab cho chọn gì? | [Kiến trúc hiện tại của SAI](sai-current.md) |
| Cần chọn các ô trên web như thế nào? | [Thực hành so kernel](kernel-guide.md) |
| “Khớp”, “Lệch”, “×” có nghĩa gì? | [Đọc kết quả](results.md) |
| Đã đổi kernel nhưng chưa biết bước tiếp? | [Câu hỏi thường gặp](faq.md) |

## Bốn cấp thay đổi

| Cấp | Ví dụ | Nơi chịu trách nhiệm |
| --- | --- | --- |
| Giá trị cấu hình | Số tầng, số KV head | Config kiến trúc |
| Phép tính không tham số | `sdpa`, `eager` | `ATTENTION_KERNELS` |
| Thành phần có tham số | Attention, FFN, norm | `ATTENTIONS`, `FFNS`, `NORMS` |
| Họ mô hình | `SAI`, `SAI_modular` | `MODELS` |

`SAI/` là model chính thức. `SAI_modular` là bản lắp ghép để thử các thành phần qua registry trong Lab. Một biến thể có mặt trong Lab không tự trở thành một phần của bản phát hành.

## Một phép thử có thể kết luận đến đâu?

```text
Đọc kiến trúc → xác định phần muốn thay → so output và gradient
                                           ↓
                                 đo chi phí cùng điều kiện
                                           ↓
                              thí nghiệm phù hợp với thay đổi
                                           ↓
                           đánh giá chất lượng trên dữ liệu giữ lại
```

Nếu thay đổi thực sự làm đổi hàm mô hình, đi tiếp sang [Phương pháp huấn luyện](../../recipes/docs/index.md) và [Đánh giá](../../eval/docs/index.md). Nếu quan tâm kiểu trả lời của cùng checkpoint, xem [Suy luận](../../inference/docs/index.md).

Nội dung mô tả working tree được đọc ngày 25/10/2026. Các lựa chọn và mặc định đang chạy phải lấy từ registry, schema và config hiện tại; bảng trong sổ tay là ảnh chụp để học, không thay thế cấu hình của run.
