---
schema_version: 1
department: objectives
page_id: faq
title: "Câu hỏi thường gặp về Objective"
description: "Giải đáp các nhầm lẫn về loss tổng, chunk, mask, stage làm mốc và chất lượng mô hình."
order: 60
kind: faq
updated: "2026-10-25"
tags: [FAQ, chunk, mask, stage, objective]
sources:
  - foundation_lab/objectives/registry.py
  - foundation_lab/objectives/lm_ce.py
  - foundation_lab/eval/compare.py
  - foundation_lab/eval/lm_loss.py
  - foundation_lab/recipes/base.py
---

## Vì sao chỉ có lm_ce?

Đó là tên duy nhất được đăng ký ở snapshot 25/10/2026. Registry cho phép thêm implementation đúng hợp đồng về sau. Trong phạm vi hiện có, thử chunk khác nhau là một thí nghiệm có ích. Việc đổi chuỗi `candidate` thành một tên chưa đăng ký sẽ báo lỗi cấu hình.

## Loss tổng lớn hơn có nghĩa mô hình kém hơn không?

Tổng loss phụ thuộc số token và trọng số token. So hai tổng từ `N` khác nhau không trả lời được mô hình nào tốt hơn. Phòng này dùng dữ liệu ngẫu nhiên; kết luận chất lượng còn không áp dụng ngay cả khi `N` bằng nhau. Val/test dùng CE trung bình trên token chịu loss để đọc chất lượng dự đoán của checkpoint trên một tập cụ thể.

## Bỏ trống chunk khác điền 0 thế nào?

Bỏ trống có nghĩa lấy `loss_chunk_size` mốc của từng experiment. Điền 0 không có nghĩa tự động; schema yêu cầu số nguyên dương nếu cung cấp. Quy tắc tương tự áp dụng cho số token: trống dùng ngân sách mốc, 0 là sai.

## Experiment không ghi objective thì baseline là gì?

Resolver gộp `stage_defaults`, params của stage và mặc định dataclass của recipe. Do đó một trường không có trong YAML vẫn có giá trị sau resolve. Đọc mốc đã resolve của run để biết giá trị thực tế, thay vì suy luận từ một file YAML riêng lẻ.

## Objective khác mốc có bắt buộc dừng thí nghiệm không?

Nếu mục tiêu là tối ưu implementation của cùng công thức, lệch loss/gradient cần được giải thích trước khi dùng train. Nếu đang nghiên cứu objective mới, khác có thể đúng dự định; cần kiểm tra công thức và đo kết quả học bằng thí nghiệm riêng. Nhãn web “Khác mốc” giữ hai tình huống này tách khỏi một phán xét chất lượng.

## Chunk nhỏ có giống microbatch nhỏ không?

Không. Chunk chia các hàng hidden khi chiếu ra vocabulary. Microbatch chia forward của cả model. Chúng giảm bộ nhớ ở các phần khác nhau. Xem [Chunk và bộ nhớ](chunk-memory.md).

## Mask đúng rồi có cần kiểm tra gradient nữa không?

Có. Một implementation có thể trả đúng tổng nhưng truyền gradient sai. Với token weight 0, test kiểm tra hàng hidden tương ứng không có gradient trực tiếp từ objective; với weight khác không, cần kiểm tra hướng và độ lớn phù hợp. Phép so web hiện chỉ dùng weight 1 nên không thay test mask.

## Tôi thay objective train, vì sao val vẫn dùng CE?

Lab giữ `lm_ce` cho val/test để các thí nghiệm có một thước đo dự đoán token chung. Điều đó không bảo đảm CE là thước đo duy nhất cần dùng; chất lượng câu trả lời và mục tiêu ứng dụng còn cần đánh giá riêng.

## Có thể chạy lại kết quả cũ bằng cùng tên experiment không?

Tên experiment hiện tại có thể resolve ra cấu hình mới. Muốn giữ mốc cũ, dùng `resolved_config.yaml` được lưu của run và xem lineage/code version tương ứng. Giữ nguyên file run cũ; tạo run mới cho phép đo tiếp theo.
