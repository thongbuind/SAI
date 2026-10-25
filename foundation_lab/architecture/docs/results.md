---
schema_version: 1
department: arch
page_id: results
title: "Đọc kết quả so kernel"
description: "Diễn giải sai khác, dung sai, tốc độ và bộ nhớ mà không nhầm với chất lượng mô hình."
order: 70
kind: interpretation
updated: "2026-10-25"
tags: [kết quả, gradient, dung sai, benchmark, bộ nhớ]
sources:
  - foundation_lab/eval/compare.py
  - web/site/src/lib/lab/views/compareKernel.ts
  - tests/test_compare.py
---

## Đọc theo thứ tự: điều kiện, đúng số học, rồi chi phí

Mỗi dòng ứng với một model và ghi kernel mốc, ứng viên, dtype, shape. Kiểm tra những thông tin này trước khi nhìn tốc độ. Hai dòng khác shape không phải hai lần lặp của cùng workload.

| Chỉ số | Ý nghĩa | Cách đọc |
| --- | --- | --- |
| `max_abs_diff_out` | Sai khác tuyệt đối lớn nhất giữa hai output | Cần xét cùng dtype và dung sai; không phải trung bình |
| `max_abs_diff_grad_q/k/v` | Sai khác lớn nhất của gradient theo Q, K, V | Kiểm chứng hướng cập nhật khi dùng kernel trong train |
| `matches_reference` | Output và cả ba gradient đều qua `torch.testing.assert_close` | “Khớp” là khớp trong dung sai, không nhất thiết từng bit |
| `reference_ms`, `candidate_ms` | Trung vị thời gian một lượt forward + backward | Đơn vị ms; thấp hơn nhanh hơn trên workload này |
| `speedup` | `reference_ms / candidate_ms` | `2×` nghĩa ứng viên mất nửa thời gian; `0.5×` nghĩa gấp đôi thời gian |
| `*_peak_mib` | Bộ nhớ cấp phát tăng thêm ở đỉnh của lượt đo | Chỉ có trên CUDA; không phải tổng RAM/VRAM của toàn model |

Trên web, ô sai khác hiện output và gradient lớn nhất; rê chuột cho biết riêng q/k/v. Bảng chỉ số ở trên giải thích từng thành phần; bản ghi kết quả lưu các giá trị riêng này để đọc lại.

## “Khớp” đủ để làm gì?

Phép đo dùng cùng input và gradient đầu ra ngẫu nhiên cho cả hai bên, một seed cố định trong helper, causal bật/tắt theo spec và dropout mặc định bằng không. Nó là bằng chứng cho các trường hợp đã đo. Nó không bao phủ mọi độ dài, dtype, mask, cache, dropout hoặc thiết bị; cũng không tự chứng minh kết quả sinh văn bản của toàn model giữ nguyên trên mọi prompt.

Kernel không có trọng số nên về cấu trúc có thể dùng lại trọng số model khi thay kernel. Điều kiện thực tế là kernel tuân hợp đồng và được kiểm chứng trên đường dùng cần thiết. “Có thể dùng lại trọng số” khác với “đã kiểm chứng toàn bộ bản phát hành”.

## Khi thấy Lệch

Trước hết xác nhận hai bên có cùng shape, dtype, thiết bị và ý nghĩa causal. Ghi lại output/gradient nào lệch và độ lớn cụ thể. Lệch rất nhỏ có thể liên quan thứ tự làm tròn; lệch lớn hoặc chỉ một gradient lệch cần xem hợp đồng hay implementation. Không nâng dung sai chỉ để bảng chuyển sang xanh. Kết luận cần dựa trên mục đích sử dụng và phép kiểm chứng bổ sung phù hợp.

## Khi thấy nhanh hơn

Mỗi bên được khởi động một lần; các lượt đo chạy xen kẽ mốc rồi ứng viên và lấy trung vị. CUDA có đồng bộ khi đo; số đo CPU/MPS không cung cấp bộ đếm peak CUDA. Chênh lệch nhỏ có thể do tải máy hoặc nhiễu đo. Đo một shape ngắn không đại diện cho context dài, và forward + backward không phải tốc độ sinh token có cache.

Không có cột bộ nhớ trên CPU/MPS nghĩa là **không đo**, không phải 0 MiB. Việc đổi kernel cũng không thay đổi số tham số của model.

Đây là diễn giải implementation và test được đọc ngày 25/10/2026, không phải báo cáo benchmark mới. Muốn kết luận chất lượng, xem [Sổ tay Đánh giá](../../eval/docs/index.md).
