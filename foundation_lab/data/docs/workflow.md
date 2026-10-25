---
schema_version: 1
department: data
page_id: workflow
title: Xem dữ liệu của một experiment
description: Quy trình đối chiếu nguồn dữ liệu và baseline khi phòng web chưa có inspect-data.
order: 50
kind: guide
updated: "2026-10-25"
tags: [hướng dẫn, config, baseline, planned]
sources:
  - foundation_lab/core/experiment.py
  - foundation_lab/data/sources.py
  - foundation_lab/core/configs/base.yaml
  - foundation_lab/cli.py
  - web/site/src/lib/lab/workbench.ts
---

## Điều hiện có và điều đang dự kiến

Phòng web hiện là trang chờ cho `inspect-data`. Chưa có chức năng giao diện để tính phân bố độ dài hay hiệu suất padding. Python đã có nguồn dữ liệu, chia split và loader dùng trong recipe. CLI `validate` đã có để resolve và kiểm tra experiment; nó không phải phép kiểm tra toàn bộ nội dung dataset.

## Quy trình kiểm tra trước khi train

1. Chọn đúng experiment và stage: pretrain, continued, sft1 hoặc sft2. Xác định stage có bật không; tên stage và recipe có thể khác nhau.
2. Xem cấu hình đã resolve, gồm chuỗi `extends`, data config và `stage_defaults`. Không đọc riêng `base.yaml` rồi cho rằng đã biết tất cả cấu hình của model đang chọn.
3. Ghi lại `kind`, `main`, `sub`, `mix_ratio`, `mix_seed`, `split_seed`, tỷ lệ chia. `causal_lm` cần `lm`; `sft` cần nguồn `sft` có mask.
4. Đối chiếu tokenizer và giới hạn context đã dùng khi encode với model của experiment. Đường dẫn đúng không bảo đảm tokenizer đúng.
5. Đọc phương pháp batching của recipe và giá trị `batch_size`, `max_microbatch_tokens`, `resample_per_epoch` đã resolve.
6. Kiểm tra giới hạn [rò rỉ split](split-mix.md) trước khi coi val/test là đánh giá độc lập.
7. Chỉ khi đã có phép đo được thực hiện và ghi nhận, mới điền các thống kê mẫu, token, padding vào báo cáo. Không điền 0 thay dữ liệu chưa đo.

## Lệnh tham khảo đã có

Ví dụ dưới đây là nội dung hướng dẫn; không được chạy khi mở sổ tay và chưa được thực hiện trong phiên biên soạn:

```bash
python -m foundation_lab validate foundation_lab/core/configs/SAI_35M.yaml
```

Lệnh xuất YAML đã resolve và kiểm schema/tính tương thích recipe–model–data. Nó không nạp dataset để chứng minh shard tồn tại, mask đúng, split độc lập hay RAM đủ. Trong quy tắc repo hiện tại, việc chạy CLI này vẫn cần được người dùng cho phép; các lệnh đọc file nhẹ có thể dùng để đối chiếu trước.

## Mốc so sánh hợp lệ

Nếu câu hỏi là “sampler mới giảm padding bao nhiêu?”, mốc phải là sampler và cấu hình data hiện hành của cùng stage. Giữ nguyên tập mẫu, tokenizer, seed, batch size và context; ghi rõ phần nào thay đổi. Nếu câu hỏi là “thêm nguồn sub có ích không?”, thay đổi data là chủ đích nhưng cần một tập đánh giá độc lập và baseline chạy lại phù hợp.

## Khi phòng web được triển khai

Luồng dự kiến là chọn experiment/stage → xem mốc mặc định lấy từ config → xem nguồn và giới hạn → chủ động yêu cầu thống kê → đọc kết quả. Sổ tay không được ngầm đọc dữ liệu lớn, encode, huấn luyện hoặc tạo run. Các biểu đồ minh hoạ trong tài liệu phải có nhãn ví dụ, không trộn với dữ liệu đo thật.

Tiếp theo: [Đọc kết quả và câu hỏi thường gặp](read-results.md).
