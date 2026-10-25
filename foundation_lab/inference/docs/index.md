---
schema_version: 1
department: inference
page_id: overview
title: Sổ tay Suy luận
description: Hiểu cách sinh câu trả lời, điều chỉnh giải mã và đọc phép so sánh trên cùng checkpoint.
order: 0
kind: overview
updated: "2026-10-25"
tags: [suy luận, giải mã, benchmark]
sources:
  - foundation_lab/inference/__init__.py
  - foundation_lab/inference/bench.py
  - web/site/src/lib/lab/workbench.ts
  - web/site/src/lib/lab/views/infer.ts
---

## Phòng này làm gì?

Phòng Suy luận so sánh **cách sinh câu trả lời** trên cùng model, checkpoint và bộ câu hỏi. Bạn thay decoder hoặc cấu hình giải mã, rồi đọc đầu ra và phép đo. Đây là cách kiểm tra tác động của greedy, beam hoặc sampling mà không phải huấn luyện lại trọng số.

Luồng hiện có:

```text
Chọn model + checkpoint
  → chọn câu hỏi và các biến thể, giữ một mốc mặc định
  → xác nhận chạy trên máy này
  → nạp model một lần
  → mỗi câu hỏi chạy qua từng biến thể
  → xem câu trả lời ghi dần → xem metric tổng hợp khi xong
```

## Trạng thái hiện tại

Snapshot 25/10/2026: phòng web đã có tác vụ `infer`. Registry decoder đang có `sai`, dùng `SAI.generate`; các chiến lược là `auto`, `beam`, `greedy`, `sample`. Sổ tay không chạy suy luận khi mở, đóng hoặc chuyển bài.

Mọi so sánh phải giữ một biến thể đúng `GenerationConfig()` hiện tại với decoder `sai`. Tên thường là `default`, nhưng tên không làm một cấu hình trở thành mốc. Mốc phản ánh cách bản phát hành đang sinh; không lấy các số trong tài liệu để ghi đè mặc định code.

## Bạn nên đọc gì?

| Bạn muốn biết | Bài |
| --- | --- |
| Greedy, beam và sample khác nhau thế nào? | [Các cách giải mã](decoding.md) |
| Temperature, top-p, repetition và stop dùng ra sao? | [Tra cứu cấu hình sinh](generation-options.md) |
| Vì sao prompt dài tốn bộ nhớ hoặc bị từ chối? | [Nạp model, context và KV cache](cache-and-loading.md) |
| SAI hiện đang dùng mặc định nào? | [Mốc SAI hiện tại](current-defaults.md) |
| Thao tác trên web và CLI thế nào? | [Quy trình sử dụng](workflow.md) |
| Latency, tokens/s, distinct và truncated đo điều gì? | [Đọc kết quả](interpreting-results.md) |
| Không thấy checkpoint, đổi tham số không đổi đầu ra? | [Câu hỏi thường gặp](faq.md) |

## Mục đích của benchmark này

Phòng giúp quan sát đánh đổi giữa tốc độ, độ dài, lặp và hành vi câu trả lời. Metric hiện tại không đo trực tiếp tính đúng sự thật, độ hữu ích hoặc khả năng suy luận. Đọc câu trả lời thực tế cùng con số; không chọn biến thể chỉ vì một cột được đánh dấu tốt hơn.
