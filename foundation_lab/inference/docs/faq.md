---
schema_version: 1
department: inference
page_id: faq
title: Câu hỏi thường gặp về suy luận
description: Xử lý những nhầm lẫn về mốc, sampling, prompt dài, checkpoint và kết quả đang chạy.
order: 70
kind: faq
updated: "2026-10-25"
tags: [faq, baseline, sampling, checkpoint]
sources:
  - SAI/generate.py
  - foundation_lab/inference/bench.py
  - foundation_lab/inference/loading.py
  - foundation_lab/state/checkpoint.py
  - web/site/src/lib/lab/views/infer.ts
  - web/site/src/data/lab_fields.json
---

## Vì sao phải giữ default dù chỉ muốn so sampling?

Lab yêu cầu có hành vi phát hành hiện tại làm mốc trong mọi phép so. Bạn có thể thêm hai ứng viên sampling và giữ mốc riêng. Tên `default` không đủ: decoder phải là `sai`, generation phải bằng mặc định hiện tại.

## Đổi temperature nhưng câu vẫn giống?

Temperature chỉ tác động sampling. Trong phòng infer, `auto` dùng beam, nên đổi temperature không điều khiển nhánh đó. Khi đã sample, cùng kết quả vẫn có thể xuất hiện nếu phân phối rất tập trung hoặc bộ lọc chỉ giữ ít ứng viên.

## Seed trống có bảo đảm mỗi lần khác nhau không?

Không. Nó chỉ không tạo generator riêng có seed cố định từ config; cùng câu vẫn có thể được chọn. Ngược lại, có seed giúp đối chiếu trong cùng môi trường nhưng không cam kết bitwise giống nhau qua thiết bị, thư viện hoặc checkpoint khác.

## Muốn greedy có nên đặt temperature bằng 0?

Không. Chọn `strategy: greedy`; validator yêu cầu temperature lớn hơn 0. Greedy dùng argmax và không đi qua các bộ lọc temperature/top-k/top-p của sampling.

## Có min-p, quantization hoặc tắt KV cache không?

Không có các field đó trong spec `infer`/`GenerationConfig` hiện tại. Decoder `sai` dùng API cache của model. Sổ tay không trình bày những lựa chọn chưa có như control đang dùng được.

## Prompt đang vừa nhưng tăng max_new_tokens lại lỗi?

Token sinh được đặt trước trong context budget, nên tăng nó làm giảm chỗ cho prompt. Chat formatting cũng chiếm token. Generate không cắt âm thầm câu user mới nhất để vừa; hãy xem [context và cache](cache-and-loading.md).

## Vì sao không thấy trọng số của mình trong model dropdown?

Danh sách ghép theo tên `model/<stage>_<cỡ>.pt` và `best.pt` trong run có config model khớp catalog. File khác mẫu dò hoặc model override có thể không được gán. Việc không hiện không chứng minh tensor hỏng; việc hiện cũng không chứng minh đã nạp thành công. Đọc [Sổ tay Lưu trữ](../../state/docs/index.md).

## Có thể so hai checkpoint trong cùng spec infer không?

Spec hiện có một model và một checkpoint chung, các variant thay decoder/config sinh. Muốn so checkpoint phải có các run riêng với điều kiện được kiểm soát hoặc tác vụ khác được thiết kế rõ; không đổi nghĩa tên variant thành tên checkpoint mà backend vẫn nạp cùng một file.

## Distinct cao và tokens/s cao có nghĩa model tốt hơn?

Chưa đủ. Distinct chỉ phản ánh bigram từ; tokens/s phụ thuộc độ dài và thời gian toàn lời gọi decoder. Đọc tính đúng, phù hợp yêu cầu, độ lặp, reason và độ dài bên cạnh metric. Xem [diễn giải kết quả](interpreting-results.md).

## Lưu benchmark có chạy model không? Hủy có mất hết kết quả không?

Lưu benchmark ghi lại danh sách prompt, tách khỏi thao tác Chạy. Khi hủy infer, record hoàn tất có thể còn trong event log; các lượt chưa chạy không có kết quả và có thể chưa có bảng tổng hợp. Mở/đóng Sổ tay không lưu benchmark, không hủy job và không thay cấu hình.
