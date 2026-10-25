---
schema_version: 1
department: inference
page_id: interpreting-results
title: Đọc latency, token và chất lượng đầu ra
description: Diễn giải metric hiện có của infer và giới hạn của benchmark không có đáp án tham chiếu.
order: 60
kind: interpretation
updated: "2026-10-25"
tags: [latency, tokens per second, distinct, truncated, metric]
sources:
  - foundation_lab/inference/bench.py
  - foundation_lab/eval/text.py
  - foundation_lab/state/results.py
  - web/site/src/lib/lab/views/infer.ts
  - tests/test_inference.py
  - SAI/generate.py
---

## Đọc từng câu trước bảng tổng hợp

Một ô câu trả lời hiển thị text, `completion_tokens`, latency và `finish_reason`. So trên cùng hàng để biết tác động decoder đối với cùng prompt. Có thể một biến thể nhanh hơn vì sinh ngắn hơn hoặc dừng sớm; cần đọc nội dung trước khi xem đó là cải thiện.

`prompt_tokens` tính cả định dạng chat; `completion_tokens` đếm token đã sinh, có thể gồm token kết thúc đã bị bỏ khỏi text. `decode_steps` là số lần gọi decode, không đồng nhất số từ và không luôn bằng số token trả về.

## Công thức metric hiện tại

| Metric | Công thức/phạm vi | Chiều được gắn |
| --- | --- | --- |
| `latency_ms_mean` | Tổng thời gian gọi decoder / số câu | Thấp hơn |
| `tokens_per_second` | Tổng completion token / tổng giây gọi decoder | Cao hơn |
| `completion_tokens_mean` | Tổng completion token / số câu | Không gán hướng |
| `truncated_rate` | Tỷ lệ record có `finish_reason == "length"` | Thấp hơn |
| `distinct_2` | Bigram từ khác nhau / tổng bigram từ của mọi câu trả lời | Cao hơn |

`tokens_per_second` là tỷ số hai tổng, không phải trung bình đơn giản tokens/s của từng câu. Latency bao quanh lời gọi decoder, gồm chuẩn bị input, prefill, decode và chuyển kết quả của lời gọi đó; nó không là tốc độ riêng một decode kernel.

`load_ms` đo nạp/dựng model và tokenizer riêng. `total_ms` bắt đầu trước nạp rồi kéo dài qua việc xử lý các lượt sinh đến lúc dựng kết quả; nó không phải riêng tổng latency của từng câu, và không bao gồm toàn bộ thời gian đợi hàng đợi hoặc mọi chi phí HTTP của web.

## Ví dụ tính tay

Hai câu có 4 token trong 1.000 ms và 2 token trong 500 ms. Khi đó latency trung bình 750 ms; throughput `6 / 1,5 = 4 token/s`; độ dài trung bình 3 token. Nếu một record có reason length, truncated rate là 0,5.

Hai text `a b a b` và `c d` có các bigram `(a,b)`, `(b,a)`, `(a,b)`, `(c,d)`: 3 loại trên 4 lần xuất hiện, distinct-2 bằng 0,75. Biên giữa hai câu không tạo thêm bigram. Đây là ví dụ tính metric, không phải kết quả SAI đo được.

## Đọc lý do kết thúc

| Reason | Ý nghĩa hiện tại |
| --- | --- |
| `stop` | Đường greedy/sample gặp token hoặc dãy stop |
| `length` | Vòng sinh kết thúc ở giới hạn theo nhánh hiện tại; benchmark dùng nhãn này cho truncated |
| `stable_beam` | Beam thỏa tiêu chí ổn định đã cài |
| `all_beams_finished` | Không còn nhánh beam sống để mở rộng |
| `empty` | Đường beam không có ứng viên để chọn |

Reason của beam mô tả trạng thái tìm kiếm, không phải bộ chấm ngữ nghĩa câu trả lời. `length` là nhãn vận hành mà metric đếm; không khẳng định mọi câu có reason này đều thiếu ý, hoặc mọi câu có reason khác đều hoàn chỉnh.

## Distinct không đo đúng sai

Hàm hiện tách từ bằng khoảng trắng, không dùng token SentencePiece và không phân tích từ ghép tiếng Việt. Nó gom các n-gram trong từng câu, rồi tính tập khác nhau trên cả bộ câu trả lời. Bộ câu quá ngắn không có bigram trả về 0; không coi đó là lặp hoàn toàn.

Distinct cao có thể đến từ văn bản đa dạng hoặc văn bản vô nghĩa. Distinct thấp cũng có thể do nhiều câu cần cùng một cụm chính xác. Không tối ưu con số này bằng cách làm mất nội dung cần lặp.

## Giới hạn phép đo tốc độ

Infer hiện gọi mỗi cặp prompt × variant một lần theo thứ tự, không có warmup, repeats, median/p95, khoảng tin cậy, TTFT hay profiler GPU riêng trong spec. Thời gian đầu có thể chịu chi phí khởi động, độ dài đầu ra giữa variant có thể khác, và tải máy cũng biến động. Không suy một lần đo thành bảo đảm SLA hoặc xếp hạng kernel.

Để báo kết quả: nêu checkpoint/config, bộ prompt, device, phiên bản/lineage, số câu, độ dài, reason và các quan sát nội dung. Khi cần kết luận mạnh hơn, thiết kế phép đo lặp phù hợp trong tác vụ được duyệt, giữ mốc mặc định; đừng giả định runner hiện đã làm các bước này.
