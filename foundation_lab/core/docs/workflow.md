---
schema_version: 1
department: core
page_id: workflow
title: Cách kiểm tra một experiment trước khi chạy
description: Một quy trình đọc cấu hình và dùng validate để phát hiện nhầm lẫn trước khi tiêu thụ tài nguyên.
order: 50
kind: guide
updated: "2026-10-25"
tags: [validate, hướng dẫn, workflow, CLI, config, lỗi]
sources:
  - foundation_lab/cli.py
  - foundation_lab/core/experiment.py
  - foundation_lab/core/config.py
  - web/site/src/lib/lab/workbench.ts
---

## Trên web hiện tại

Ban điều hành đang là trang chờ, chưa có form chọn experiment hoặc nút validate của riêng phòng. Chuỗi `validate --json` mô tả hướng dự kiến, không phải một lệnh đã dùng được. Các bước dưới là quy trình CLI hiện có; trong modal sau này, ví dụ vẫn chỉ là nội dung để đọc/sao chép.

## Bước 1 — Chọn đúng đơn vị cấu hình

Chọn file trong `foundation_lab/core/configs/` nếu muốn kiểm pipeline huấn luyện. File `config/models/SAI_35M.yaml` chỉ chứa cấu hình model, không phải experiment đầy đủ để truyền trực tiếp cho `validate`.

Nếu đang so kernel/objective hoặc suy luận, tác vụ có spec riêng; đọc sổ tay phòng đó. Không đưa spec infer vào validator experiment và kỳ vọng mọi field được hiểu.

## Bước 2 — Xem cấu hình mốc trước

Lệnh minh họa, chưa được thực thi bởi sổ tay:

```bash
python -m foundation_lab validate foundation_lab/core/configs/SAI_35M.yaml
```

Lệnh hiện tại in YAML đã resolve ra stdout và `# OK: config hợp lệ` ra stderr khi qua kiểm tra. Nó không có `--json`, không tạo run và không train model. Vẫn cần môi trường Python/dependency của Lab; không coi việc một lệnh không train là không cần môi trường.

Đọc ít nhất: family/shape model, runtime, thứ tự stage, enabled, recipe, nguồn data, init và params từng stage. Đặc biệt xem stage nào dùng giá trị riêng thắng stage_defaults.

## Bước 3 — Thay một yếu tố để hiểu override

Ví dụ minh họa kiểm một thay đổi ở số epoch SFT1:

```bash
python -m foundation_lab validate foundation_lab/core/configs/SAI_35M.yaml --set pipeline.sft1.params.epochs=1
```

Đối chiếu output với bước trước: chỉ trường mục tiêu phải đổi theo ý định. Lệnh này không đánh giá liệu một epoch đã đủ chất lượng. Cấu hình mặc định ở bước trước vẫn là mốc giải thích; đừng so ứng viên với một cấu hình ngẫu nhiên khác rồi gọi là baseline.

Không tự ghi đè file nguồn bằng redirect output nếu chưa muốn sửa file đó. Bạn có thể đọc output và quyết định riêng việc lưu cấu hình theo quy trình làm việc của mình.

## Bước 4 — Đọc lỗi đúng cấp

| Tín hiệu | Cách xử lý ở mức người dùng |
| --- | --- |
| Thiếu khóa bắt buộc | Kiểm đang dùng đúng loại spec, không thêm khóa đoán mò |
| Khóa không hợp lệ | So tên với danh sách hợp lệ trong thông báo |
| Recipe không nhận data kind | Chọn recipe/nguồn tương thích thực sự, không chỉ đổi nhãn |
| Thiếu capability | Kiểm model family hỗ trợ công việc đó |
| Init tham chiếu stage sau | Sắp thứ tự hoặc chọn nguồn khởi tạo đúng ý định |
| Không tìm thấy tên config | Kiểm tên file không đuôi và thư mục sở hữu |

Lỗi config có thể cho mã thoát 2; lỗi trong lúc chạy là nhóm khác. Một option CLI không tồn tại có thể được parser từ chối trước validator. Không diễn giải mọi lỗi mã 2 thành lỗi tensor/model.

## Bước 5 — Chuẩn bị trước khi thực thi

Khi validate xong, vẫn kiểm tồn tại dữ liệu/trọng số, khả năng tương thích, dự toán bộ nhớ, split đánh giá và tên run mới. Những bước này thuộc các phòng tương ứng; không bấm Chạy chỉ dựa vào chữ OK.

Nếu chỉ cần biết tên implementation, CLI `list` có sẵn. `describe` phục vụ công cụ web bằng JSON, nhưng cũng không thay kết quả validate một experiment cụ thể. Xem [Registry và capability](registry-capabilities.md).

## Khi bàn Ban điều hành được triển khai sau này

Hướng dự kiến là chọn experiment, xem cấu hình mốc lấy từ code/config, xem override và cấu hình cuối, rồi đọc lỗi trước khi chuyển sang phòng huấn luyện. Đó là thiết kế tương lai; sổ tay sẽ phải được cập nhật khi giao diện thực sự có các thao tác đó. Các mô tả dự kiến không phải nút đang có để bạn tìm trên trang hôm nay.
