---
schema_version: 1
department: objectives
page_id: overview
title: "Sổ tay phòng Objective"
description: "Hiểu hàm loss, token chịu loss và cách đối chiếu một implementation objective trong Lab."
order: 0
kind: overview
updated: "2026-10-25"
tags: [objective, loss, cross-entropy, bắt đầu]
sources:
  - foundation_lab/objectives/__init__.py
  - foundation_lab/objectives/registry.py
  - foundation_lab/objectives/lm_ce.py
  - foundation_lab/eval/compare.py
  - web/site/src/lib/lab/workbench.ts
---

## Objective là gì?

Objective là đại lượng quá trình train muốn tối ưu. Với ngôn ngữ tự hồi quy, objective hiện có của SAI là cross-entropy dự đoán token kế tiếp: model bị phạt khi gán xác suất thấp cho token đúng. Gradient của loss cho optimizer biết cách điều chỉnh tham số.

Hàm objective của Lab nhận hidden state, trọng số đầu ra, target và trọng số từng token; nó trả về **tổng loss có trọng số**, chưa chia mẫu số. Recipe quyết định token nào học và mẫu số; engine thực hiện forward/backward và tối ưu.

## Công cụ hiện tại làm gì?

Phòng web chạy `compare-objective`: tạo tensor ngẫu nhiên có cỡ hidden/head giống model trong experiment, rồi so objective ứng viên với objective của một stage làm mốc. Nó đo tổng loss, gradient, thời gian và bộ nhớ; không train hoặc chấm câu trả lời của checkpoint.

Ngày 25/10/2026, registry `OBJECTIVES` chỉ có `lm_ce`. Bạn vẫn có thể dùng phòng để so cùng `lm_ce` với kích thước chunk khác nhau. Một tên objective lý thuyết không thể gõ vào form để tự có implementation.

## Bắt đầu từ nhu cầu của bạn

| Câu hỏi | Đọc tiếp |
| --- | --- |
| Logits, cross-entropy, gradient nghĩa là gì? | [Loss và hợp đồng objective](loss-contract.md) |
| Vì sao SFT không tính loss trên prompt? | [Mask, target và mẫu số](mask-normalization.md) |
| Chunk có làm đổi bài toán học không? | [Chunk và bộ nhớ logits](chunk-memory.md) |
| Làm sao chọn experiment và stage trên web? | [Hướng dẫn so objective](compare-guide.md) |
| “Khác mốc” có luôn là lỗi không? | [Diễn giải kết quả](results.md) |
| Vì sao chỉ có một objective hoặc thiếu cột bộ nhớ? | [Câu hỏi thường gặp](faq.md) |

## Giữ rõ ba câu hỏi

```text
Implementation tính đúng hợp đồng? → test loss/gradient/mask
Implementation có tiết kiệm hơn?  → benchmark cùng workload
Mô hình học được tốt hơn?          → train + đánh giá dữ liệu giữ lại
```

Phép so trong phòng trả lời chủ yếu hai câu đầu. Nếu thay **bản chất** objective, loss/gradient khác mốc có thể là mong đợi; để biết có ích phải thiết kế thí nghiệm train và đánh giá bằng thước đo chung. Xem [Phương pháp huấn luyện](../../recipes/docs/index.md) và [Đánh giá](../../eval/docs/index.md).

Sổ tay phản ánh source/config được đọc ngày 25/10/2026. Baseline thực tế luôn lấy từ experiment đã resolve và mặc định dataclass của recipe, không từ con số gõ tay trong bài.
