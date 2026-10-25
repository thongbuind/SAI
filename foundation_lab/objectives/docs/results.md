---
schema_version: 1
department: objectives
page_id: results
title: "Đọc kết quả so objective"
description: "Phân biệt sai khác implementation với thay đổi mục tiêu học và đọc đúng số đo hiệu năng."
order: 50
kind: interpretation
updated: "2026-10-25"
tags: [kết quả, gradient, loss, dung sai, benchmark]
sources:
  - foundation_lab/eval/compare.py
  - web/site/src/lib/lab/views/compareObjective.ts
  - tests/test_compare.py
  - tests/test_objectives.py
  - foundation_lab/objectives/registry.py
---

## Một dòng là một experiment tại một stage

Trước khi đọc kết luận, xem objective mốc, chunk mốc/ứng viên, số token `N`, `d_model`, vocabulary `V`, dtype và thiết bị. “Mốc” không chỉ là tên hàm; cùng hàm nhưng khác chunk có thể có chi phí khác.

| Chỉ số | Cách tính/ý nghĩa | Cần lưu ý |
| --- | --- | --- |
| `reference_loss`, `candidate_loss` | Tổng loss trên `N` token ngẫu nhiên | Không phải loss trung bình trên dữ liệu val |
| `loss_rel_diff` | `abs(candidate_loss-reference_loss)/abs(reference_loss)` | Sai khác tương đối; khác đơn vị với chênh lệch CE/token |
| `max_abs_diff_grad_hidden` | Sai khác tuyệt đối lớn nhất theo hidden | Kiểm chứng đạo hàm về đầu vào objective |
| `max_abs_diff_grad_weight` | Sai khác tuyệt đối lớn nhất theo head weight | Kiểm chứng đạo hàm về trọng số head |
| `matches_reference` | Loss và cả hai gradient qua dung sai `assert_close` | Không phải chứng nhận chất lượng mô hình |
| `speedup` | Thời gian mốc chia thời gian ứng viên | Lớn hơn 1 nhanh hơn trên phép đo này |
| `*_peak_mib` | Bộ nhớ cấp phát thêm ở đỉnh của forward+backward | Chỉ đo CUDA; CPU/MPS thiếu số đo |

Web hiện dòng chính là sai khác tương đối loss, dòng phụ là gradient lớn nhất trong hai loại; tooltip ghi riêng hidden/weight. Thời gian là trung vị các lần forward + backward, sau khởi động và đo xen kẽ hai bên.

## Khi cùng objective nhưng khác cách tính

Ví dụ `lm_ce` với hai chunk khác nhau: mong muốn loss và gradient khớp trong dung sai. Nếu loss gần nhau nhưng gradient lệch, implementation chưa được chứng minh tương đương để train. Nếu cả hai khớp, tiếp tục đọc thời gian và bộ nhớ; đừng ưu tiên tốc độ trước tính đúng.

Chunk nhỏ hơn không luôn nhanh hơn. Nó giảm kích thước logits tạm nhưng tăng số lần gọi phép tính và có thể tăng công tính lại. Dùng cùng workload để cân nhắc đánh đổi, giữ cả số tuyệt đối lẫn tỷ lệ.

## Khi objective khác bản chất

Một mục tiêu học mới có thể cố ý tạo loss/gradient khác CE. “Khác mốc” khi đó là kết quả mô tả, không tự là bug. Ngược lại, cùng giá trị loss trên một tensor không đủ chứng minh hai objective giống nhau.

Trước khi chạy hãy ghi kỳ vọng: cần tương đương số học, hay cần khác có chủ đích? Với mục tiêu khác, các bước tiếp theo là test công thức/hợp đồng riêng và thí nghiệm train có baseline. Val/test của pipeline hiện dùng CE làm thước đo chung; xem [Đánh giá](../../eval/docs/index.md).

## Phạm vi và nhiễu

Phép so dùng hidden/head/targets ngẫu nhiên, mọi token weight bằng 1. Nó không kiểm tra loss mask SFT trong lượt đo, không chạy toàn Transformer, không đo optimizer step và không tính chất lượng văn bản. Test mask riêng trong `tests/test_objectives.py` là một lớp bằng chứng khác.

Khi số ms rất nhỏ, overhead và tải máy có thể chi phối. Không diễn giải `1.02×` của một lượt đo thành cải thiện chắc chắn toàn pipeline. Nếu không có cột bộ nhớ, ghi “không đo trên thiết bị này”, không thay bằng 0.

Tài liệu mô tả code và test đã đọc ngày 25/10/2026; không báo cáo kết quả thực nghiệm mới.
