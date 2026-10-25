---
schema_version: 1
department: core
page_id: current-defaults
title: Pipeline SAI hiện tại lấy mặc định từ đâu?
description: Đọc đúng cấu hình gốc, cấu hình theo cỡ model và giá trị riêng của từng stage.
order: 40
kind: reference
updated: "2026-10-25"
tags: [SAI, mặc định, baseline, pretrain, sft1, sft2, continued]
sources:
  - foundation_lab/core/configs/base.yaml
  - foundation_lab/core/configs/SAI_35M.yaml
  - foundation_lab/core/configs/SAI_100M.yaml
  - foundation_lab/core/configs/SAI_375M.yaml
  - foundation_lab/core/experiment.py
  - foundation_lab/__init__.py
  - SAI/version.py
---

## Đây là ảnh chụp để giải thích

Các bảng dưới được đọc từ code/config ngày 25/10/2026, không phải bộ giá trị để sao chép làm baseline mãi mãi. Khi chạy hoặc so sánh, chọn experiment/model hiện có và xem config đã resolve. Một run cũ giữ config riêng; sổ tay hôm nay không thay thế config đó.

Trong working tree lúc đọc, lab là `2.1.0`, model phát hành ghi `dev3.0.1`. Hai số phiên bản quản lý hai phạm vi độc lập, không phải một dãy tăng chung.

## Nguồn cấu hình

`foundation_lab/core/configs/base.yaml` đặt pipeline chung. File `SAI_35M.yaml`, `SAI_100M.yaml`, `SAI_375M.yaml` kế thừa base, chọn model và đổi một số thông số theo cỡ. Cấu trúc model thật đến từ `config/models/`, không được suy ra chỉ từ tên experiment.

```text
base.yaml
  ├─ SAI_35M.yaml  → model config SAI_35M
  ├─ SAI_100M.yaml → model config SAI_100M
  └─ SAI_375M.yaml → model config SAI_375M
```

## Pipeline chung

| Stage | Recipe | Mặc định bật? | Khởi tạo | Epoch | Learning rate | Final eval |
| --- | --- | --- | --- | --- | --- | --- |
| pretrain | causal_lm | Có | Mới | 1 | 2.0e-4 | last |
| continued | causal_lm | Không | pretrain.best | 5 | 1.5e-5 | last |
| sft1 | sft | Có | pretrain.best | 4 | 2.0e-5 | best |
| sft2 | sft | Có | sft1.best | 5 | 1.0e-5 | best |

Nhánh mặc định là pretrain → sft1 → sft2. Bật continued không tự thay init của sft1; sft1 vẫn lấy pretrain.best nếu bạn không thay cấu hình đó có chủ đích. `final_eval: last` nghĩa là đánh giá trọng số cuối vòng lặp; `best` chọn trọng số được coi tốt nhất theo quy tắc validation của recipe. Không gộp chúng thành cùng một loại số đo.

## Tham số chung và runtime

| Nguồn | Field | Giá trị tại lần đối chiếu | Cách hiểu |
| --- | --- | --- | --- |
| stage_defaults | batch_size | 64 | Cỡ batch mục tiêu; SFT dùng nó để đặt ngân sách token chịu loss nên số mẫu thực có thể thay đổi |
| stage_defaults | accumulation_steps | 1 | Số batch tích lũy theo cấu hình chung |
| stage_defaults | max_microbatch_tokens | 16384 | Ngân sách chia microbatch theo số ô token sau padding; không phải trần RAM/VRAM cứng |
| stage_defaults | loss_chunk_size | 4096 | Chia phần tính loss; không phải context length |
| stage_defaults | max_grad_norm | 1.0 | Ngưỡng clip gradient |
| stage_defaults | checkpoint_every_steps | 5000 | Nhịp checkpoint định kỳ |
| stage_defaults | lr_schedule | warmup_stable_decay | Tên lịch đã đăng ký |
| runtime | device | auto | Thiết bị do runtime chọn theo khả năng máy |
| runtime | seed | null | Không đặt seed chung từ cấu hình này |
| runtime | compile | true | Có yêu cầu dùng compile theo runtime |
| runtime | profile_kernels / profile_vram | false / false | Không bật profiling mặc định |

Giá trị lớn không phải đề xuất cho máy 8 GB. Trước khi chạy, đọc [Vận hành](../../engine/docs/index.md) để hiểu bộ nhớ, microbatch và compile. Sổ tay không tự thử các cấu hình này trên máy của bạn.

## Khác nhau theo cỡ

| Experiment | num_workers | freeze_blocks ở SFT1 và SFT2 |
| --- | --- | --- |
| SAI_35M | 1 | [0, 1] |
| SAI_100M | 4 | [0, 1, 2] |
| SAI_375M | 1 | [0, 1, 2, 3] |

Block đánh số từ 0. Đóng băng block không đồng nghĩa đóng băng toàn model. Số worker là một lựa chọn nguồn lực, không phải số GPU. Đọc [Phương pháp huấn luyện](../../recipes/docs/index.md) để biết tác động của freeze.

## Giới hạn dữ liệu cần biết trước khi đọc kết quả

Base hiện đặt `sft2.resample_per_epoch: true`. Source đã ghi nhận việc trộn/split lại có thể đưa val/test vào train qua các epoch. Ngoài ra việc trộn toàn nguồn phụ trước split có thể gây lẫn dữ liệu giữa các stage. Do đó validation loss thấp hơn không tự chứng minh khả năng tổng quát hóa tốt hơn. Xem [Dữ liệu](../../data/docs/index.md) và [Đánh giá](../../eval/docs/index.md) để diễn giải đúng.

Đây là mô tả chính sách và giới hạn đang có, không phải đề xuất giữ chúng vĩnh viễn. Thay cách split/loss có thể làm kết quả cũ mất khả năng so trực tiếp và cần quyết định phiên bản riêng.
