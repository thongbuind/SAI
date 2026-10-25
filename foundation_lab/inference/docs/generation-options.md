---
schema_version: 1
department: inference
page_id: generation-options
title: Tra cứu cấu hình sinh
description: Giải thích miền giá trị, phạm vi áp dụng và tác động của từng field trong GenerationConfig.
order: 20
kind: reference
updated: "2026-10-25"
tags: [temperature, top-p, top-k, stop, repetition]
sources:
  - SAI/generate.py
  - foundation_lab/inference/bench.py
  - web/site/src/data/lab_fields.json
---

## Chiến lược và độ dài

| Field | Miền trong validator hiện tại | Tác động |
| --- | --- | --- |
| `strategy` | `auto`, `beam`, `greedy`, `sample` | Chọn thuật toán giải mã |
| `max_new_tokens` | Số nguyên ít nhất 1 | Số token sinh tối đa; đồng thời lấy chỗ khỏi ngân sách prompt |
| `beam_size` | Số nguyên ít nhất 1 | Số nhánh beam; không điều khiển greedy/sample |
| `stable_steps` | Số nguyên ít nhất 1 hoặc `null` | Tiêu chí ổn định của beam; `null` tắt riêng tiêu chí này |
| `length_penalty` | Số không âm | Số mũ chuẩn hóa điểm beam theo độ dài |

Validator có thể kiểm tra cả field không được chiến lược đang chọn sử dụng. Ví dụ temperature vẫn phải hợp lệ dù strategy là greedy. Giới hạn slider của web là phạm vi UI, không nhất thiết trùng toàn bộ miền CLI.

`max_new_tokens` không phải “độ dài chắc chắn sẽ sinh”. EOS, kết thúc lượt model hoặc stop sequence có thể dừng sớm. Tăng giới hạn có thể làm prompt đang hợp lệ trở nên quá dài vì budget prompt bằng `max_seq_len - max_new_tokens`.

## Lấy mẫu

| Field | Miền | Hiểu đúng |
| --- | --- | --- |
| `temperature` | Lớn hơn 0 | Điều chỉnh độ tập trung của xác suất; chỉ sampling dùng |
| `top_k` | Số nguyên không âm | 0 tắt lọc top-k; giá trị dương dùng ngưỡng top-k |
| `top_p` | Trong `(0, 1]` | 1 tắt lọc nucleus; nhỏ hơn 1 cắt phần đuôi xác suất |
| `seed` | Số nguyên hoặc `null` theo schema | Generator lấy mẫu riêng nếu có seed; null không cố định seed theo config này |

Top-k dùng ngưỡng logit; nếu nhiều token bằng điểm ở ranh giới, số token được giữ có thể nhiều hơn k. Top-p giữ token vượt ngưỡng tích luỹ đầu tiên để không loại hết khối cần thiết. Lọc top-p được tính sau temperature và top-k nếu bật.

`min_p`, typical sampling và beam sampling không phải field/strategy được hỗ trợ trong config hiện tại. Không thêm chúng vào YAML rồi kỳ vọng tự bị bỏ qua; schema của lab kiểm tra khóa cấu hình.

## Chống lặp theo token

`repetition_penalty` phải lớn hơn 0. Giá trị 1 là trung tính; lớn hơn 1 giảm xu hướng chọn lại token đã sinh bằng phép điều chỉnh logit theo dấu. Nó chỉ xét phần token đã sinh trong helper hiện tại, không phạt toàn bộ token xuất hiện trong prompt. Giá trị dưới 1 không phải “phạt nhẹ hơn” theo nghĩa thông thường: nó có thể ưu tiên token đã xuất hiện.

`no_repeat_ngram_size` là số nguyên không âm. 0 tắt; 1 cấm lặp token đã sinh; n lớn hơn cấm token khiến chuỗi n-gram token lặp lại. Đây là n-gram **token**, khác `distinct_2` tính theo từ ở kết quả. Chặn quá mạnh có thể làm mất khả năng lặp tên riêng hoặc cụm từ cần thiết.

Các quy tắc này sửa bước chọn token, không xóa câu lặp sau khi đã sinh và không bảo đảm đầu ra luôn trôi chảy. Đọc [kết quả](interpreting-results.md) để quan sát thay vì chỉ nhìn tỷ lệ distinct.

## Viết thường và dừng

`lowercase=true` viết thường nội dung được mã hóa trong prompt chat trước khi tokenize. Nó không phải một bộ xử lý chỉ viết thường câu trả lời. Khi kiểm tra tên riêng, mã nguồn hay dữ liệu phân biệt hoa/thường, đây là một biến cần biết và kiểm soát.

`stop_sequences` là danh sách các dãy **ID token**, không phải danh sách chuỗi văn bản. Generate luôn theo dõi `[EOS]` và `<|im_end|>`; danh sách này bổ sung điều kiện hậu tố. Các token stop ở cuối bị bỏ khi decode thành text, nhưng vẫn có thể được tính trong `token_ids` và `completion_tokens` của kết quả.

Web hiện ẩn field `stop_sequences`; không có editor stop text trong Control center. API/spec có cấu trúc field này, nhưng người dùng phải cung cấp ID theo đúng tokenizer khi có nhu cầu dùng qua cấu hình phù hợp.

Các default cụ thể được tập trung ở [Mốc SAI hiện tại](current-defaults.md), để tránh nhầm một hướng dẫn chỉnh tham số thành giá trị mặc định sống.
