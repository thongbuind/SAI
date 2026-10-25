---
schema_version: 1
department: recipes
page_id: stages-init-resume
title: Stage, init, resume và đóng băng tham số
description: Chọn đúng cách bắt đầu hoặc tiếp tục huấn luyện và hiểu những gì được mang qua giữa các stage.
order: 20
kind: concept
updated: "2026-10-25"
tags: [stage, init, resume, checkpoint, freeze]
sources:
  - foundation_lab/pipeline.py
  - foundation_lab/core/experiment.py
  - foundation_lab/cli.py
  - foundation_lab/state/checkpoint.py
  - foundation_lab/recipes/sft.py
  - foundation_lab/engine/freeze.py
---

## Mỗi stage dựng model mới

Pipeline đi theo thứ tự khai báo. Với mỗi stage được chọn, nó đặt seed nếu có, dựng model mới từ config, rồi nạp trọng số khởi tạo hoặc checkpoint resume. Model của stage trước không được truyền trực tiếp làm một object đang sống sang stage sau.

| Cách bắt đầu | Model weights | Optimizer/scheduler | Vị trí train |
| --- | --- | --- | --- |
| `init: null` hoặc không có init | Khởi tạo mới | Mới | Đầu stage |
| `init: pretrain.best` | Best của stage được tham chiếu trong cùng run | Mới | Đầu stage |
| `init` là đường dẫn `.pt` | Trọng số từ file phù hợp | Mới | Đầu stage |
| `run --resume <run>` | Checkpoint của chính stage, nếu có | Khôi phục từ checkpoint | Tiếp tục vị trí đã lưu |

`--init stage=path` là override trọng số khởi tạo cho stage, không đổi thành resume. Nếu file là checkpoint đầy đủ nhưng dùng qua init, đường nạp chỉ lấy model weights.

## Init từ stage khác

Tham chiếu `<stage>.best` phải trỏ tới stage xuất hiện trước trong pipeline. File best cần tồn tại tại thời điểm chạy; chọn riêng `sft1` không tự chạy pretrain để tạo file còn thiếu. CLI có thể dùng `--init sft1=<path>` với trọng số đã có và tương thích.

`enabled: false` tắt stage khi chạy pipeline mặc định. Nếu dùng `--stages`, pipeline chọn theo danh sách được truyền; vì vậy đây là thao tác chọn stage rõ ràng, không chỉ là bộ lọc stage đang bật. Thứ tự thực thi vẫn theo pipeline, không theo thứ tự viết trong danh sách.

## Resume mang theo điều gì?

`last.ckpt.pt` lưu model, optimizer, scheduler, vị trí epoch/step, sampler state, scaler và RNG khi có. `best.pt` chỉ là state_dict trọng số. Resume dùng `resolved_config.yaml` của run; CLI từ chối truyền experiment mới hoặc `--set` cùng `--resume`. Stage đã có `done.json` được bỏ qua.

Nếu stage chưa xong nhưng không có checkpoint resume, pipeline đi lại đường khởi tạo của stage. Đừng coi bất kỳ thư mục run nào cũng bảo đảm tiếp tục đúng đến batch cuối cùng trước sự cố; chỉ trạng thái đã lưu mới phục hồi được. Tính tái lập còn cần dữ liệu/code/môi trường tương ứng; xem [Lưu trữ](../../state/docs/index.md).

## Đóng băng block trong SFT

`freeze_blocks` là danh sách chỉ số decoder block từ 0. Các tham số trong block đó được đặt `requires_grad=False`; optimizer chỉ nhận tham số còn trainable. Nó không xóa block khỏi forward và không có nghĩa toàn model ngừng học. Embedding/head hoặc block khác vẫn có thể cập nhật.

Đóng băng có thể giảm phần gradient/optimizer phải quản lý, nhưng hạn chế khả năng thích nghi; không mặc định bảo đảm giảm thời gian hoặc bộ nhớ theo đúng tỷ lệ tham số frozen. Quyết định phải dựa trên phép đo và đánh giá chất lượng.

Mỗi stage dựng model mới nên freeze của stage trước không tự tồn tại như trạng thái cấu hình stage sau. `freeze_blocks` của từng stage cần rõ ràng trong config sau resolve.

Tiếp theo: [Mốc hiện tại của SAI](baseline.md).
