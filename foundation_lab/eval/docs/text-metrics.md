---
schema_version: 1
department: eval
page_id: text-metrics
title: "Thước đo văn bản sinh và chi phí"
description: "Đọc distinct-n, tỷ lệ bị cắt, độ dài và tốc độ cùng với chất lượng câu trả lời thực tế."
order: 20
kind: reference
updated: "2026-10-25"
tags: [distinct-n, đa dạng, latency, tốc độ, chất lượng văn bản]
sources:
  - foundation_lab/eval/text.py
  - foundation_lab/inference/bench.py
  - foundation_lab/state/results.py
---

## Distinct-n đo sự lặp của các cụm từ

Implementation `distinct_n(texts,n)` tách từng văn bản bằng khoảng trắng, tạo các n-gram trong mỗi văn bản rồi tính:

```text
số n-gram khác nhau / tổng số n-gram
```

N-gram là một dãy `n` đơn vị liên tiếp. `n=1` nhìn đơn vị đơn, `n=2` nhìn cặp. Code không nối n-gram qua ranh giới hai câu trả lời. Nó không dùng tokenizer SAI và không phân tích từ tiếng Việt: một từ có nhiều âm tiết cách nhau bằng khoảng trắng được đếm thành nhiều đơn vị. Dấu câu và hoa/thường vẫn nằm trong chuỗi vì hàm này chỉ dùng `split()`.

Ví dụ `a b a b` có ba bigram: `(a,b)`, `(b,a)`, `(a,b)`; distinct-2 là `2/3`. Nếu mọi văn bản đều ngắn hơn `n`, kết quả là 0 do không có n-gram, không phải bằng chứng model đang lặp. `n<1` là đầu vào không hợp lệ.

Phòng Suy luận hiện tổng hợp `distinct_2` trên toàn bộ câu trả lời của từng biến thể. Đây là dấu hiệu độ đa dạng bề mặt, không là điểm đúng kiến thức. Sinh chuỗi ngẫu nhiên có thể có distinct cao nhưng hoàn toàn vô ích. Ý tưởng đo đa dạng n-gram xuất hiện trong [nghiên cứu về đa dạng hội thoại](https://aclanthology.org/N16-1014/); công thức cụ thể ở đây lấy từ code Lab ngày 25/10/2026.

## Các số đo ở phòng Suy luận

| Tên metric | Công thức/phạm vi hiện có | Cách diễn giải |
| --- | --- | --- |
| `latency_ms_mean` | Trung bình thời gian gọi decoder của từng prompt | Thấp hơn thường phản hồi nhanh hơn; không gồm pha load model riêng |
| `tokens_per_second` | Tổng completion token chia tổng thời gian decoder | Không phải trung bình đơn giản tốc độ của từng câu |
| `completion_tokens_mean` | Tổng completion token chia số câu | Mô tả độ dài; không có chiều tốt hơn cố định |
| `truncated_rate` | Tỷ lệ `finish_reason == "length"` | Bị chạm giới hạn sinh; cần đọc câu có bị dở dang không |
| `distinct_2` | Distinct bigram trên tập câu trả lời | Phát hiện lặp bề mặt; phụ thuộc độ dài và bộ câu hỏi |

Pha load có `load_ms`, thời gian toàn tác vụ có `total_ms`. Không cộng/gộp tùy ý rồi so với latency mỗi câu. Những số đo này phản ánh đường gọi hiện tại; không coi là benchmark latency phần cứng chuẩn hoá hay thời gian đến token đầu tiên nếu công cụ chưa đo riêng đại lượng đó.

## Đọc cùng câu trả lời

Khi một biến thể có distinct cao hơn, xem nó còn trả lời đúng và bám câu hỏi không. Khi latency thấp hơn, xem câu có ngắn hơn hoặc bị cắt không. Khi token/s cao hơn, giữ cùng checkpoint, thiết bị, câu hỏi và điều kiện giải mã cần so.

Một bảng đánh giá thủ công hữu ích có các cột: đúng ý chính, lỗi thực tế, bám yêu cầu, lặp, thiếu phần kết và ghi chú ví dụ. Đó là khung đọc do sổ tay đề xuất, chưa là công cụ chấm tự động của Lab. Với câu hỏi kiến thức, cần đáp án hoặc nguồn đối chiếu riêng; không gán nhãn “đúng” chỉ vì văn bản trôi chảy.

Xem [Suy luận](../../inference/docs/index.md) để chạy các biến thể hiện có và [Thiết kế phép đánh giá](evaluation-protocol.md) để chọn điều kiện so sánh.
