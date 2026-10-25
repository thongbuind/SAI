---
title: Experiments
description: Một experiment là một file YAML mô tả model, runtime và các stage huấn luyện. Lab đọc file, kiểm tra nghiêm ngặt, rồi chạy từng stage theo thứ tự.
group: Bắt đầu
order: 12
---

## File experiment

File experiment nằm trong `foundation_lab/core/configs/`. Pipeline chuẩn của SAI nằm trong `base.yaml`; mỗi cỡ model có một file kế thừa nó và chỉ khai báo phần khác:

```yaml
# foundation_lab/core/configs/SAI_375M.yaml
extends: base.yaml
model: SAI_375M
runtime:
  num_workers: 1
pipeline:
  sft1:
    params:
      freeze_blocks: [0, 1, 2, 3]
  sft2:
    params:
      freeze_blocks: [0, 1, 2, 3]
```

Các khoá ở cấp cao nhất:

| Khoá | Ý nghĩa |
| --- | --- |
| `model` | Tên config trong `config/models/` (vd. `SAI_375M`) hoặc `foundation_lab/architecture/configs/` (model thí nghiệm), hoặc mapping config model |
| `pipeline` | Các stage theo thứ tự chạy |
| `runtime` | Thiết bị, seed, `compile`, đo đạc, số worker; không làm đổi thuật toán |
| `stage_defaults` | Tham số gộp vào `params` của mọi stage |
| `name` | Tên experiment, cũng là tên thư mục trong `runs/`; mặc định là tên file (vd. `SAI_375M`) |
| `description` | Mô tả ngắn của experiment |
| `extends` | File cha để kế thừa, xem [Config](../config/#kế-thừa-bằng-extends) |

Cách lab tìm và gộp các config này được mô tả ở trang [Config](../config/).

## Pipeline nhiều stage

Mỗi khoá trong `pipeline` là một stage, chạy theo thứ tự khai báo.

| Khoá | Ví dụ | Ý nghĩa |
| --- | --- | --- |
| `recipe` | `sft` | Phương pháp huấn luyện, chọn trong registry `recipe` |
| `data` | `sft2_mix` | Tên file trong `foundation_lab/data/configs/` hoặc mapping cấu hình dữ liệu |
| `init` | `sft1.best` | Trọng số khởi tạo (xem bên dưới) |
| `enabled` | `false` | Tắt stage mà không xoá khỏi file |
| `params` | `epochs: 4` | Tham số của recipe |

### Khởi tạo giữa các stage

- Không có `init`: khởi tạo trọng số mới.
- `<tên stage>.best`: dùng checkpoint tốt nhất của một stage **đứng trước** trong pipeline.
- Giá trị khác: đường dẫn tới file `.pt`.

Khi chạy, có thể ghi đè `init` của một stage bằng `--init stage=duong_dan.pt`, ví dụ để bỏ qua pretrain:

```bash
python -m foundation_lab run foundation_lab/core/configs/SAI_375M.yaml \
    --stages sft1,sft2 --init sft1=model/pretrained_375M.pt
```

## Lab kiểm tra gì trước khi chạy

`validate` (và `run`) kiểm tra toàn bộ trước khi nạp dữ liệu hay dựng model:

- schema của experiment và config model theo họ mô hình;
- `params` của từng stage theo recipe;
- config dữ liệu theo loại nguồn, và recipe có nhận loại dữ liệu đó không;
- họ mô hình có đủ capability mà recipe cần không;
- `init` dạng `<stage>.best` có trỏ tới stage đứng trước không.

> `base.yaml` còn một số điểm chưa tốt đã biết, ví dụ stage sft2 trộn lại dữ liệu mỗi epoch nên mẫu val/test có thể lọt vào train. Các điểm này sẽ thành tuỳ chọn ở mốc sau.

Mỗi lần `run` tạo một thư mục run; xem [Thư mục run](../runs/) để biết bên trong có gì và cách chạy tiếp khi bị ngắt.
