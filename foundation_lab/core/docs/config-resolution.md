---
schema_version: 1
department: core
page_id: config-resolution
title: Cấu hình được ghép và kiểm tra như thế nào?
description: Hiểu extends, tham chiếu tên, override và stage_defaults để biết giá trị cuối cùng thực sự được dùng.
order: 20
kind: reference
updated: "2026-10-25"
tags: [config, resolve, extends, override, stage_defaults, validate, đường dẫn]
sources:
  - foundation_lab/core/config.py
  - foundation_lab/core/experiment.py
  - foundation_lab/paths.py
  - foundation_lab/cli.py
---

## Bốn bước resolve experiment

1. Đọc YAML/JSON và gộp `extends` từ file cha.
2. Thay tham chiếu tên `model` và `pipeline.<stage>.data` bằng mapping config tương ứng.
3. Áp các `--set khoa.con=gia_tri` vào cấu hình đã có.
4. Gộp `stage_defaults` vào `params` từng stage; giá trị riêng của stage được ưu tiên.

Sau đó validate dựng các schema và kiểm tương thích. “Resolve” là làm rõ cấu hình; “validate” là kiểm nó có hợp lệ theo các ràng buộc đã triển khai hay không. Cả hai không phải chạy train.

## Kế thừa không phải nối mọi thứ

Khi hai giá trị cùng là mapping, Lab gộp đệ quy. Giá trị không phải mapping được thay toàn bộ. Vì vậy danh sách `freeze_blocks` ở file con thay danh sách ở file cha; không tự cộng thêm block.

Ví dụ khái niệm: cha đặt `runtime.device: auto` và `runtime.num_workers: 1`; con chỉ đặt `runtime.num_workers: 4`. Kết quả vẫn có device auto và worker 4. Nếu con thay một danh sách, các phần tử cũ không còn trừ khi con ghi lại. Những ví dụ này giải thích phép gộp, không phải đề xuất cấu hình hiệu quả nhất.

`extends` tương đối với file đang đọc. Tên model được tìm trong `config/models/` và `foundation_lab/architecture/configs/`; phải xuất hiện ở đúng một nơi. Tên data config được tìm trong `foundation_lab/data/configs/`. Các đường dẫn dữ liệu/trọng số tương đối trong config được hiểu theo gốc dự án khi đi qua `resolve_project_path`.

## Override đúng nơi

Sau bước thay tên model bằng nội dung, override nên nhắm vào field của mapping, ví dụ `runtime.device=cpu` hoặc `pipeline.sft1.params.epochs=1`. Không giả định mọi chỗ đều có thể thay một mapping đã resolve trở lại bằng tên chuỗi và được resolve lần nữa.

Giá trị bên phải dấu `=` được đọc bằng YAML. Dùng dấu nháy shell cho danh sách, chuỗi có khoảng trắng hoặc ký tự shell; `false` không giống chuỗi `"false"`. Trong trường số thực, loader có xử lý chuỗi số phù hợp, nhưng dùng ký hiệu rõ như `2.0e-4` dễ đọc hơn.

Nếu một stage đã ghi `learning_rate` riêng, đặt `stage_defaults.learning_rate` không thắng giá trị riêng đó. Muốn đổi đúng stage thì đặt `pipeline.<stage>.params.learning_rate`. Để biết chắc, đọc đầu ra `validate`, không suy luận chỉ từ file cuối của chuỗi extends.

## Schema kiểm những gì?

| Kiểm tra | Ví dụ lỗi bị phát hiện |
| --- | --- |
| Field được phép | Gõ nhầm `batch_szie` |
| Field bắt buộc | Experiment thiếu model hoặc pipeline cần thiết |
| Kiểu dữ liệu | Đưa chuỗi vào trường cần danh sách; dùng boolean như số nguyên |
| Lựa chọn cố định | Device không thuộc auto/cuda/mps/cpu |
| Registry | Recipe hoặc họ model chưa đăng ký |
| Khả năng model | Recipe đòi capability mà model family không có |
| Loại dữ liệu | Recipe nhận `sft` nhưng nguồn khai báo `lm` |
| Thứ tự init | Tham chiếu best của stage đứng sau |

Lỗi có đường dẫn field như `pipeline.sft1.params…`; dùng đường dẫn đó để tìm nơi sửa. Không xóa một field chưa hiểu chỉ để validation qua.

## Những gì validate chưa chứng minh

Không đảm bảo có đủ RAM/VRAM, thiết bị yêu cầu tồn tại, checkpoint tương thích mọi tensor, dữ liệu sạch hoặc dữ liệu không bị trùng giữa train/val/test. Không nhất thiết kiểm mọi miền số học mong muốn của từng tham số nếu implementation chưa đặt điều kiện đó. Tham chiếu stage trước hợp lệ về cấu trúc vẫn có thể thiếu file best khi thực thi một phần pipeline.

Schema thành công chỉ là cửa kiểm đầu tiên. Đọc các phòng [Dữ liệu](../../data/docs/index.md), [Vận hành](../../engine/docs/index.md) và [Lưu trữ](../../state/docs/index.md) trước những quyết định tương ứng.

## Resolved config và giá trị mặc định trong code

Resolved config đã bỏ kế thừa và thay các tham chiếu do resolver quản lý. Tuy vậy, những giá trị dataclass mặc định không được ghi tường minh trong mapping có thể vẫn được bổ sung lúc validate/dựng cấu hình. Vì thế file này dùng lại được trong môi trường code tương ứng, nhưng không thay thế việc lưu commit/version trong lineage.

Không đánh đồng “tự đủ về tham chiếu” với “độc lập mãi mãi với mọi phiên bản code”. Khi đọc run cũ, lấy resolved config cùng lineage của run đó.
