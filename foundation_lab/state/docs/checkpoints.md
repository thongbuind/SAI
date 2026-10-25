---
schema_version: 1
department: state
page_id: checkpoints
title: Trọng số, best và checkpoint resume
description: Đọc đúng nội dung của best.pt, last.ckpt.pt và điều kiện chọn trọng số cho model.
order: 20
kind: concept
updated: "2026-10-25"
tags: [weights, checkpoint, best, resume]
sources:
  - foundation_lab/state/checkpoint.py
  - foundation_lab/engine/loop.py
  - foundation_lab/recipes/base.py
  - foundation_lab/cli.py
---

## Hai định dạng, hai mục đích

| Thuộc tính | `best.pt` | `last.ckpt.pt` |
| --- | --- | --- |
| Nội dung model | State dict thuần | `model_state_dict` trong một mapping |
| Optimizer/scheduler | Không | Có |
| Vị trí epoch/batch/step | Không | Có |
| Sampler/RNG/scaler | Không | Có thể có theo format |
| Dùng để sinh hoặc init | Có | Có thể trích phần model bằng loader |
| Dùng để tiếp tục đường học | Không đủ | Định dạng dành cho resume |

`best.pt` là trọng số có val loss thấp nhất đã quan sát trong stage; nó có thể đến từ epoch trước. `last.ckpt.pt` là điểm lưu gần nhất, không nhất thiết có val tốt nhất.

Huấn luyện ghi checkpoint định kỳ sau optimizer step theo `checkpoint_every_steps` và cuối mỗi epoch. Nếu tiến trình ngắt sau lần lưu, phần cập nhật chưa lưu phải chạy lại khi resume; không thể khôi phục việc chưa được ghi xuống đĩa.

## Checkpoint hiện giữ những gì?

Format đang dùng là `foundation_lab.train_checkpoint.v1` với các khóa:

- `epoch`, `global_step`, `best_val_loss`.
- `model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`.
- `batches_done`, `sampler_state`.
- `scaler_state_dict`, `rng_state`.

`batches_done=null` nghĩa là epoch đã hoàn tất; resume bắt đầu epoch kế tiếp. Có số batch nghĩa là tiếp tục cùng epoch và bỏ qua các batch đã được checkpoint xác nhận. `sampler_state` phục vụ thứ tự batch, RNG phục vụ các phép ngẫu nhiên như dropout.

Loader vẫn nhận checkpoint thiếu scaler/RNG cũ, nhưng nạp được không đồng nghĩa bảo đảm tái lập chính xác khi thiếu trạng thái. Ghi file dùng file tạm rồi đổi tên để tránh thấy checkpoint đích đang ghi dở; cơ chế này không tạo lịch sử nhiều phiên bản cho cùng `last.ckpt.pt`.

## Tương thích model là điều kiện bắt buộc

Trọng số cần đúng họ model, shape, config kiến trúc và tokenizer. Tên “35M” hay “375M” chỉ giúp phân loại, không thay việc kiểm tra tương thích. `load_model_weights` mặc định `strict=true`; nó nhận cả state dict thuần và checkpoint có `model_state_dict`.

`strict=false` tồn tại cho trường hợp cấy ghép module, có báo khóa thiếu/thừa. Đây không phải tuỳ chọn hợp thức hoá mọi checkpoint khác kiến trúc; mismatch kích thước tensor vẫn cần xử lý riêng và một thí nghiệm được duyệt.

## Vì sao một file không hiện trong danh sách web?

Danh sách checkpoint được tạo từ quy tắc nguồn hiện tại:

1. `model/*.pt` có tên dạng `<stage>_<cỡ>.pt` được gán cho model chính thức `SAI_<cỡ>` nếu model đó có trong catalog.
2. `runs/*/*/stages/*/best.pt` được ghép với model có mapping config bằng đúng `model` trong `resolved_config.yaml` của run.

Tên lạ, config model đã override không còn khớp catalog, hoặc `last.ckpt.pt` không thuộc danh sách best có thể không xuất hiện. Bộ dò chỉ đọc đường dẫn và config metadata, không nạp tensor để chứng minh từng file tương thích. CLI vẫn kiểm tra khi nạp thực tế.

Đọc [init và resume](init-and-resume.md) để chọn đúng hành động; đọc [phiên bản](versions.md) trước khi dùng checkpoint từ code cũ.
