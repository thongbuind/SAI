---
schema_version: 1
department: arch
page_id: faq
title: "Câu hỏi thường gặp về Kiến trúc"
description: "Giải đáp những nhầm lẫn giữa cấu hình model, kernel, checkpoint và kết quả so sánh."
order: 80
kind: faq
updated: "2026-10-25"
tags: [FAQ, lỗi cấu hình, baseline, checkpoint]
sources:
  - foundation_lab/eval/compare.py
  - foundation_lab/architecture/models/modular.py
  - foundation_lab/architecture/layers/attention.py
  - foundation_lab/architecture/layers/norm.py
  - foundation_lab/architecture/layers/ffn.py
  - SAI/modeling.py
  - config/models/SAI_35M.yaml
---

## Vì sao không chọn được SAI_35M để so kernel?

Họ `SAI` là model chính thức và không có trường `attention_kernel` trong config. Phép so cần biết kernel mốc từ model, nên yêu cầu họ có trường này, chẳng hạn `SAI_modular_35M`. Công cụ chỉ lấy shape và cấu hình kernel, không load model 35 triệu tham số để chạy toàn bộ forward.

## Vì sao 35M dùng lớp GQA nhưng có tám query head và tám KV head?

Lớp GQA bao quát trường hợp `Hkv = H`. Khi đó mỗi query head có KV riêng, tức MHA. GQA giữa hai đầu mút dùng ít KV head hơn query head; MQA dùng một KV head. Xem [Attention](attention.md).

## Tôi chỉ thấy một FFN và một norm; có bị thiếu menu không?

Snapshot ngày 25/10/2026 chỉ đăng ký `swiglu` và `rmsnorm`. Registry là điểm mở rộng cho lập trình viên, không có nghĩa mọi biến thể lý thuyết đã tồn tại. Giao diện hiện có chỉ so kernel; nó chưa là công cụ thay FFN/norm rồi tự huấn luyện.

## Candidate bằng mốc thì còn ý nghĩa gì?

Có thể dùng để kiểm tra luồng đo và mức nhiễu thời gian. Output/gradient nên khớp với điều kiện đo được kiểm soát; tốc độ vẫn có thể hơi khác do nhiễu. Một phép so chính nó không xác nhận biến thể khác đã đúng.

## Tôi có thể thay số head rồi dùng checkpoint cũ không?

Đổi số KV head thường đổi shape ma trận QKV. Đổi số query head có thể giữ tổng chiều Q nhưng thay cách tách head và vị trí RoPE. Ngay cả khi một số tensor vẫn nạp được, ý nghĩa phép tính có thể khác. Cần một thiết kế chuyển đổi/huấn luyện được duyệt riêng, không dựa vào việc file checkpoint mở thành công.

## Tăng context có làm model nhớ được nhiều hơn ngay không?

Tăng giới hạn thực thi chỉ cho phép nhiều vị trí hơn nếu implementation hỗ trợ. Chất lượng ở vị trí xa còn phụ thuộc dữ liệu train, mã hoá vị trí và ngân sách tính toán. Hiện SAI có giới hạn `max_seq_len` rõ ràng; không có điều khiển RoPE scaling trong phòng này.

## Kernel nhanh hơn có làm câu trả lời hay hơn không?

Không có kết luận đó từ phép so kernel. Nếu hai implementation tương đương số học, mục tiêu thường là cùng hàm với chi phí khác. Chất lượng phải được đo bằng checkpoint và dữ liệu/câu hỏi phù hợp ở [Đánh giá](../../eval/docs/index.md) và [Suy luận](../../inference/docs/index.md).

## Trạng thái Xong có nghĩa ứng viên đạt yêu cầu không?

Xong nghĩa tác vụ hoàn tất và ghi được kết quả. Kết quả có thể là “Lệch”, chậm hơn hoặc không có đo bộ nhớ trên thiết bị đó. Hãy đọc cả điều kiện và kết luận từng dòng. Nếu tác vụ bị hủy, các dòng đã hoàn tất chỉ là kết quả phần đã chạy, không đại diện cho tất cả model đã chọn.

## Lần chạy cũ và config hiện tại khác nhau thì tin bên nào?

Để hiểu run cũ, tin config đã resolve và bản ghi của run đó. Để tạo baseline mới, lấy mặc định từ code/config hiện tại. Sổ tay có ngày cập nhật để phân biệt hai nhu cầu này; không ghi đè mốc cũ bằng mặc định mới.
