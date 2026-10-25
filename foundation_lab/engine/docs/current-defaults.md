---
schema_version: 1
department: engine
page_id: current-defaults
title: Mặc định vận hành hiện tại của SAI
description: Snapshot cấu hình pipeline chuẩn và cách phân biệt mặc định dataclass với experiment đã resolve.
order: 50
kind: reference
updated: "2026-10-25"
tags: [sai, mặc định, config, baseline]
sources:
  - foundation_lab/core/configs/base.yaml
  - foundation_lab/core/configs/SAI_375M.yaml
  - foundation_lab/core/experiment.py
  - foundation_lab/recipes/base.py
  - foundation_lab/engine/step.py
---

## Nguồn đúng của mốc so sánh

Snapshot dưới đây được đọc từ working tree ngày 25/10/2026. **Experiment đã resolve là mốc thực thi**: gộp `extends`, model/data, override và stage defaults. Giá trị default ở dataclass chỉ được dùng khi cấu hình không cung cấp field đó.

Một khác biệt cần nhớ: `LMTrainParams.max_microbatch_tokens` và hằng bộ chia có default 57.600, nhưng `core/configs/base.yaml` đặt 16.384 cho pipeline chuẩn. Giao diện phải lấy cấu hình đã resolve theo experiment/stage đang chọn, không lấy một con số duy nhất từ bài này.

## Runtime chuẩn

| Field | Snapshot | Ý nghĩa |
| --- | --- | --- |
| `device` | `auto` | CUDA → MPS → CPU |
| `seed` | `null` | Không đặt seed theo pipeline |
| `compile` | `true` | Compile `forward_features` |
| `limited_max_autotune` | `false` | Chưa bật nhánh autotune giới hạn |
| `dynamo_cache_size_limit` | `24` | Default từ `RuntimeSpec` |
| `profile_kernels`, `profile_vram` | `false` | Chưa bật profiling chi tiết |
| `num_workers` | `1` | Một worker DataLoader |

## Tham số chung cho stage chuẩn

| Field | Snapshot |
| --- | --- |
| `batch_size` | 64 |
| `accumulation_steps` | 1 |
| `max_microbatch_tokens` | 16.384 |
| `max_grad_norm` | 1,0 |
| `loss_chunk_size` | 4.096 |
| `checkpoint_every_steps` | 5.000 optimizer step |
| `lr_schedule` | `warmup_stable_decay` |

## Tham số tối ưu theo stage

| Stage | Epoch | LR | Weight decay | Warmup | Fused | Test dùng |
| --- | --- | --- | --- | --- | --- | --- |
| pretrain | 1 | `2e-4` | 0,2 | 0,1 | `true` | last |
| continued | 5 | `1.5e-5` | 0,1 | 0,1 | `true` | last |
| sft1 | 4 | `2e-5` | 0,05 | 0,15 | `null` | best |
| sft2 | 5 | `1e-5` | 0,05 | 0,15 | `null` | best |

`continued` mặc định tắt. `null` của fused nghĩa là bật theo CUDA trong recipe. Bảng là mô tả lựa chọn đang có, không phải khuyến nghị tất cả các thiết bị đều chạy được với chúng.

Base không freeze block; riêng experiment `SAI_375M` override `freeze_blocks: [0, 1, 2, 3]` cho SFT1 và SFT2. Muốn biết cỡ model khác, đọc experiment tương ứng đã resolve thay vì suy từ 375M.

## Giới hạn ảnh hưởng đọc kết quả

`final_eval:last` của pretrain/continued là chính sách lấy trọng số cuối để tính test loss; `best.pt` vẫn được chọn theo val. Không gán test loss này cho best khi chưa kiểm tra `test_weights`.

Base bật `resample_per_epoch` cho SFT2. Source hiện ghi nhận giới hạn đã biết: việc trộn và split lại có thể làm mẫu val/test cố định lọt vào train ở epoch sau. Không coi val/test của chế độ này là bằng chứng đánh giá độc lập hoàn toàn. Sổ tay chỉ ghi trạng thái nguồn, không thay đổi cách chạy.

Đọc [quy trình thử nghiệm](workflow.md) để ghi lại mốc và điều kiện đo, cùng [FAQ](faq.md) nếu số liệu không phù hợp kỳ vọng.
