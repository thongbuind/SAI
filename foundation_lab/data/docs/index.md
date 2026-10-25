---
schema_version: 1
department: data
page_id: overview
title: Sổ tay phòng Dữ liệu
description: Hiểu dữ liệu đi từ token đã encode đến batch mà một stage thực sự học.
order: 0
kind: overview
updated: "2026-10-25"
tags: [dữ liệu, bắt đầu, batch]
sources:
  - foundation_lab/data/__init__.py
  - foundation_lab/data/sources.py
  - foundation_lab/data/loaders.py
  - web/site/src/lib/lab/workbench.ts
---

## Phòng này giúp bạn trả lời điều gì?

Một model không học trực tiếp từ file văn bản. Tokenizer chuyển văn bản thành token ID; phòng Dữ liệu đọc các ID, chọn nguồn, trộn, chia train/val/test rồi gom thành batch. Chất lượng kết luận thí nghiệm phụ thuộc cả nội dung dữ liệu lẫn cách tổ chức những bước này.

```text
Văn bản → Tokenizer → NPZ hoặc manifest → nguồn lm/sft
        → trộn nguồn → chia split → sampler → collator → LMBatch → recipe
```

Phòng này chịu trách nhiệm từ NPZ/manifest đến `LMBatch`. Cách tách văn bản thuộc [Tokenizer](../../tokenizer/docs/index.md); cách cập nhật trọng số thuộc [Phương pháp huấn luyện](../../recipes/docs/index.md).

## Bắt đầu theo nhu cầu

| Bạn muốn hiểu | Trang nên đọc |
| --- | --- |
| Model dự đoán token nào; vì sao prompt không chịu loss | [Token, nhãn và hai loại mask](tokens-masks.md) |
| Train/val/test, trộn dữ liệu và rò rỉ đánh giá | [Split và trộn nguồn](split-mix.md) |
| Batch, sampler, padding và bộ nhớ | [Gom batch và padding](batch-padding.md) |
| File encode chứa gì, manifest dùng để làm gì | [Định dạng lưu trữ](storage-manifest.md) |
| Xem cấu hình dữ liệu của experiment đang dùng | [Quy trình sử dụng](workflow.md) |
| Đọc số token, hiệu suất padding, những nhầm lẫn thường gặp | [Đọc kết quả và câu hỏi thường gặp](read-results.md) |

## Trạng thái hiện tại

Tại thời điểm đối chiếu working tree ngày 25/10/2026, nguồn `lm`, `sft`, sampler và collator đã có trong Python. Trang web phòng Dữ liệu đang ở trạng thái dự kiến; `inspect-data` là tên lệnh dự kiến trong cấu hình giao diện, chưa phải chức năng chạy được. Sổ tay có thể đọc độc lập khi phòng chưa có nút chạy.

Giới hạn cần biết ngay: nguồn phụ hiện được trộn trước khi split; SFT2 còn có thể trộn lại mỗi epoch. Các cách này có thể đưa mẫu thuộc tập đánh giá trước đó vào train. Đọc [phạm vi và hệ quả](split-mix.md) trước khi dùng loss để kết luận model tiến bộ.

## Mốc mặc định của SAI

Mốc phải là dữ liệu của **stage trong experiment đang chọn**, sau khi resolve config. Không có một dataset hoặc batch size chung cho mọi stage. Những bảng trong sổ tay mô tả snapshot; cấu hình đang chạy và `resolved_config.yaml` của run mới là căn cứ cho một kết quả cụ thể.
