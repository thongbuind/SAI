---
schema_version: 1
department: core
page_id: faq
title: Câu hỏi thường gặp về cấu hình Lab
description: Giải đáp các nhầm lẫn giữa model config, experiment, mặc định, lệnh kiểm tra và khả năng thực thi.
order: 70
kind: faq
updated: "2026-10-25"
tags: [FAQ, cấu hình, validate, lỗi, mặc định, stage]
sources:
  - foundation_lab/core/config.py
  - foundation_lab/core/experiment.py
  - foundation_lab/cli.py
  - foundation_lab/pipeline.py
  - foundation_lab/core/configs/base.yaml
  - web/site/src/lib/lab/workbench.ts
---

## Tôi sửa base nhưng stage vẫn dùng giá trị cũ?

Kiểm file con, override CLI và params riêng của stage. Mapping được gộp qua extends, sau đó override được áp; cuối cùng params riêng thắng stage_defaults. Dùng đầu ra validate để nhìn giá trị cuối thay vì chỉ đọc một file. Xem [Resolve config](config-resolution.md).

## Tôi thay danh sách freeze_blocks, vì sao block cũ biến mất?

Danh sách được thay toàn bộ, không nối. Viết rõ danh sách cuối bạn muốn, nhưng chỉ sau khi hiểu tác động của freeze ở [Phương pháp huấn luyện](../../recipes/docs/index.md).

## Validate thành công có nghĩa checkpoint chắc chắn nạp được?

Không. Validator experiment chủ yếu kiểm schema, registry, capability, data kind và quan hệ stage. Tồn tại file, shape trọng số và điều kiện runtime còn được xử lý ở những bước khác. Đọc [Lưu trữ](../../state/docs/index.md) trước khi chọn init/resume.

## Vì sao web Ban điều hành chưa có nút kiểm tra?

Bàn chạy đang được xây dựng trong working tree ngày đối chiếu. CLI `validate` có thật, còn biến thể `validate --json` là dự kiến. Sổ tay không làm lệnh dự kiến trở thành lệnh đã triển khai.

## Tôi có thể dùng config model làm file experiment?

Không trực tiếp. Experiment cần tên, model và pipeline, với runtime tùy chọn. Model config chỉ mô tả một mô hình. Chọn file experiment theo cỡ trong `foundation_lab/core/configs/`, rồi kiểm cấu hình đã resolve.

## Tôi bật continued, SFT1 có tự dùng kết quả của nó?

Không. Base hiện đặt init của sft1 là `pretrain.best`; bật stage không tự sửa các cạnh phụ thuộc. Bạn cần quyết định riêng luồng khởi tạo và cách đánh giá thay đổi đó.

## Tôi chỉ chạy SFT1 thì pretrain có tự chạy theo?

Không. `--stages` chọn những stage sẽ chạy, không tự dựng một scheduler thực thi mọi phụ thuộc. Nếu SFT1 cần pretrain.best mà chưa có file trong run, bạn phải có nguồn khởi tạo thích hợp hoặc thực hiện giai đoạn trước theo kế hoạch của mình.

## Tên model có hậu tố M có đủ để xác định checkpoint?

Không. Cần cấu hình cấu trúc, family và nguồn trọng số tương thích; cùng nhãn kích thước không chứng minh mọi tensor hoặc tokenizer giống nhau. Mốc và lineage giúp truy vết hơn là chỉ dựa tên.

## Default trong sổ tay khác form hoặc run cũ, tin cái nào?

Sổ tay có ngày đối chiếu để giải thích. Form lấy mô tả/config của phiên bản web–lab hiện tại. Run cũ có resolved config và lineage riêng. Trước hết xác định bạn đang hỏi về hiện tại hay một run cụ thể; không tự sửa run cho khớp bảng trong bài.

## Tôi thấy số phiên bản lab khác dev3, có lỗi không?

Không nhất thiết. Lab và bản phát hành SAI có dãy phiên bản độc lập. Thay tài liệu không tự đòi tăng một trong hai. Chính sách tương thích khi tăng phiên bản nằm ở [Lưu trữ](../../state/docs/index.md).

## Đọc sổ tay có chạy gì trên máy không?

Tính năng được thiết kế chỉ để đọc nội dung đã build. Các lệnh trong bài là ví dụ text; việc bấm Sổ tay, tìm kiếm hoặc sao chép không chạy train/eval/infer, không đọc tập dữ liệu thật và không sửa config.
