---
schema_version: 1
department: core
page_id: registry-capabilities
title: Registry, model family và khả năng tương thích
description: Hiểu vì sao một option phải được đăng ký và vì sao một recipe chỉ dùng được với một số model và dữ liệu.
order: 30
kind: concept
updated: "2026-10-25"
tags: [registry, capability, model family, option, implementation, describe]
sources:
  - foundation_lab/core/registry.py
  - foundation_lab/core/tasks.py
  - foundation_lab/core/experiment.py
  - foundation_lab/catalog.py
  - foundation_lab/cli.py
---

## Registry là bảng tra tên

Registry ánh xạ tên trong config sang implementation. Khi chọn một attention kernel, bạn chọn một tên mà phòng Kiến trúc đã đăng ký; Lab dùng tên đó để lấy phần thực thi. Viết tên thuật toán mới vào YAML không tạo ra thuật toán.

Tên trùng bị từ chối khi đăng ký; tên không có bị báo lỗi khi lấy. Cách này giúp lỗi gõ sai hiện rõ, thay vì âm thầm dùng một mặc định khác và làm thí nghiệm sai ý định.

## Các phòng sở hữu lựa chọn gì?

| Phòng | Nhóm lựa chọn |
| --- | --- |
| Kiến trúc | Model family, attention, attention kernel, FFN, norm |
| Dữ liệu | Data source |
| Tokenizer | Encoder cho từng cách đóng gói dữ liệu |
| Objective | Hàm loss |
| Vận hành | Lịch learning rate |
| Phương pháp huấn luyện | Recipe |
| Suy luận | Decoder |

Ban điều hành, Lưu trữ và Đánh giá vẫn là phòng Lab dù không có một danh sách registry cấu hình riêng trong `DEPARTMENTS`. `TASKS` là danh mục lệnh tác vụ và không nằm trong danh mục implementation chọn qua config này.

## Model family khác cỡ model

Family định nghĩa cách dựng model và schema cấu hình của họ đó. Cỡ 35M/100M/375M là những config khác nhau trong một family. Chuyển từ 35M sang 375M không chỉ đổi nhãn hiển thị; shape trọng số có thể khác và cần checkpoint đúng cấu hình.

`SAI` là model chính thức. `SAI_modular` là nơi thử thành phần lắp ghép. Một thuật toán có trong bài kiến thức không đồng nghĩa đã có implementation tương ứng trong một trong hai family. Đọc [Kiến trúc](../../architecture/docs/index.md) để phân biệt họ model, lớp attention và kernel tính toán.

## Capability dùng để làm gì?

Recipe khai báo khả năng nó cần ở model family. Validator lấy tập yêu cầu đó so với tập capability của family. Nếu thiếu, nó dừng thay vì để một method không tồn tại gây lỗi sau khi đã nạp dữ liệu và tạo model.

Capability là hợp đồng ở mức tính năng, không phải chứng chỉ mọi shape, mask hoặc trường hợp góc đã được kiểm thử. Các giới hạn cụ thể của implementation vẫn cần đọc trong sổ tay từng phòng.

Recipe còn khai báo loại dữ liệu được nhận. `lm` và `sft` khác nhau về thông tin mask/giám sát. Không chuyển loại bằng cách đổi tên nguồn nếu dữ liệu thực không có cấu trúc tương ứng.

## `list` và `describe` khác nhau thế nào?

`list` in tên implementation theo phòng/registry. `describe` in JSON cho công cụ như web: phiên bản format, phiên bản lab, registry, schema task, tên model, experiment với stage bật, và danh sách trọng số phù hợp được tìm thấy.

Schema task gồm kiểu, default hoặc required khi được mô tả. Nó không phải toàn bộ experiment đã resolve, cũng không cung cấp mọi giá trị runtime của run hiện tại. Web cần dùng đúng field có trong output, không tự đoán API chứa dữ liệu chưa khai báo.

Danh sách trọng số trong describe dựa trên các đường dẫn/naming/config đã được hỗ trợ để tìm chúng; có tên trong danh sách không thay thế kiểm tra nạp trọng số. Describe cũng không đánh giá chất lượng model.

## Khi muốn thử một lựa chọn chưa có

Trước hết xác định phòng sở hữu biến thể: đổi phép tính attention thuộc Kiến trúc; đổi objective thuộc Objective; đổi cách sinh thuộc Suy luận. Thiết kế hợp đồng, kiểm tra nhỏ và đăng ký trong registry của phòng đó sau khi được duyệt code. Không đưa logic huấn luyện vào `SAI/` hoặc sửa phòng khác chỉ để bẻ đường chọn config.

Việc đọc sổ tay và chọn một option đã có là thao tác người dùng. Việc thêm implementation là thay đổi source, cần quy trình review riêng theo quy định repo.
