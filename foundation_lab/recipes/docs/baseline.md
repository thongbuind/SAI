---
schema_version: 1
department: recipes
page_id: baseline
title: Pipeline mặc định của SAI và mốc so sánh
description: Tra cứu snapshot các stage hiện tại và cách lấy baseline từ experiment đã resolve.
order: 40
kind: reference
updated: "2026-10-25"
tags: [SAI, baseline, mặc định, pipeline, SFT2]
sources:
  - foundation_lab/core/configs/base.yaml
  - foundation_lab/core/configs/SAI_35M.yaml
  - foundation_lab/core/configs/SAI_100M.yaml
  - foundation_lab/core/configs/SAI_375M.yaml
  - foundation_lab/core/experiment.py
  - foundation_lab/recipes/base.py
  - foundation_lab/recipes/sft.py
---

## Pipeline đang khai báo

Snapshot từ config ngày 25/10/2026:

```text
pretrain.best ──→ sft1.best ──→ sft2
      └────────→ continued     (mặc định tắt)
```

SFT1 mặc định lấy best của pretrain, không lấy continued. Bật continued không tự đổi init SFT1. `base.yaml` giải thích continued đang tắt do dữ liệu chưa encode và stage sau chưa dùng kết quả đó; đây là mô tả config, không phải kết quả kiểm tra file dữ liệu trong phiên này.

| Stage | Recipe / data | Init | Epoch | LR | Weight decay | Test weights |
| --- | --- | --- | --- | --- | --- | --- |
| pretrain | causal_lm / pretrain | Mới | 1 | 2e-4 | 0,2 | last |
| continued, tắt | causal_lm / continued_mix | pretrain.best | 5 | 1,5e-5 | 0,1 | last |
| sft1 | sft / sft1 | pretrain.best | 4 | 2e-5 | 0,05 | best |
| sft2 | sft / sft2_mix | sft1.best | 5 | 1e-5 | 0,05 | best |

Các con số trên là ảnh chụp cấu hình ngày 25/10/2026. Khi chuẩn bị chạy hoặc so sánh, kiểm experiment đã resolve để lấy giá trị hiện hành; một run cũ cần đối chiếu config và lineage của chính run đó.

## Stage defaults và khác biệt theo cỡ model

Snapshot `stage_defaults` hiện đặt batch_size 64, accumulation 1, max_microbatch_tokens 16.384, max_grad_norm 1, loss_chunk_size 4.096, checkpoint_every_steps 5.000 và lịch `warmup_stable_decay`. Giá trị ở từng stage ưu tiên hơn stage_defaults; trường không ghi cần lấy default từ dataclass recipe. Đặc biệt default dataclass của một tham số có thể khác giá trị cấu hình experiment.

| Experiment | SFT1 và SFT2 freeze_blocks | num_workers |
| --- | --- | --- |
| SAI_35M | [0, 1] | 1 |
| SAI_100M | [0, 1, 2] | 4 |
| SAI_375M | [0, 1, 2, 3] | 1 |

Runtime base hiện để seed null và compile true. Null nghĩa không chủ động đặt seed cho run; không diễn giải là seed 0. Không bỏ qua khác biệt model, data hoặc frozen blocks khi so hai run từ hai experiment.

## Baseline phải được lấy từ đâu?

Với thí nghiệm mới: resolve file experiment, chuỗi extends, model/data refs, override và stage_defaults; sau đó áp schema/default của recipe. Đó là mốc cho model/stage đang chọn. Với run đã tồn tại: đọc `resolved_config.yaml` và lineage của run; config hiện tại có thể đã khác khi run được tạo.

Khi thử một thay đổi, ghi cả baseline và candidate, phần khác nhau, bộ dữ liệu đánh giá, checkpoint khởi tạo và ngân sách học. Nếu đổi số step hoặc lượng token huấn luyện, phải nói rõ; không gọi hai cấu hình công bằng chỉ vì cùng số epoch.

## Giới hạn baseline hiện tại

SFT2 mặc định `resample_per_epoch=true`. Implementation trộn rồi split lại theo epoch, trong khi val/test cố định từ lần đầu; docstring ghi nhận mẫu đánh giá có thể lọt train, làm loss và chọn best lạc quan. Ngoài ra, mix nguồn phụ lấy từ toàn bộ nguồn nên còn rủi ro xuyên stage. Xem [Dữ liệu](../../data/docs/index.md).

Mốc mặc định có ích để biết hệ thống đang làm gì; nó không tự là một thiết kế đánh giá không có vấn đề. Khi cần kết luận chất lượng, phải giải quyết tính độc lập của dữ liệu trong kế hoạch thí nghiệm được duyệt; sổ tay không tự thay config hoặc sửa run cũ.
