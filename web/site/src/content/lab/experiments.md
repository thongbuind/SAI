---
title: Experiments
description: Một experiment là một file YAML mô tả model, runtime và các stage huấn luyện. Lab đọc file, kiểm tra nghiêm ngặt, rồi chạy từng stage theo thứ tự.
group: Kiến trúc lab
order: 2
---

## File experiment

File experiment nằm trong `foundation_lab/core/configs/`. Một file có thể kế thừa file khác bằng `extends`; khoá trùng tên thì giá trị của file con được dùng. Ví dụ cấu hình 35M chỉ khai báo phần khác với pipeline gốc `base.yaml`:

```yaml
# foundation_lab/core/configs/SAI_35M.yaml
extends: base.yaml
model: SAI_35M
runtime:
  num_workers: 1
pipeline:
  sft1:
    params:
      freeze_blocks: [0, 1]
  sft2:
    params:
      freeze_blocks: [0, 1]
```

Các khoá ở cấp cao nhất:

| Khoá | Ý nghĩa |
| --- | --- |
| `model` | Tên file trong `config/models/` (vd. `SAI_35M`), `foundation_lab/architecture/configs/` (model thí nghiệm) hoặc mapping config model |
| `runtime` | Thiết bị, seed, `compile`, profiling, số worker; không làm đổi thuật toán |
| `stage_defaults` | Tham số gộp vào `params` của mọi stage |
| `pipeline` | Các stage theo thứ tự chạy |
| `description` | Mô tả ngắn của experiment |

### Lab resolve một experiment thế nào

1. Đọc file và gộp chuỗi `extends`.
2. Thay tên config bằng nội dung: `model: SAI_35M` thành `config/models/SAI_35M.yaml`, `data: sft1` thành `foundation_lab/data/configs/sft1.yaml`.
3. Áp các override `--set khoa.con=gia_tri` (giá trị đọc như YAML).
4. Gộp `stage_defaults` vào `params` của từng stage; giá trị riêng của stage được ưu tiên.

Kết quả là một config tự đủ, không còn tham chiếu, được ghi thành `resolved_config.yaml` trong thư mục run. Xem trước bằng lệnh:

```bash
python -m foundation_lab validate foundation_lab/core/configs/SAI_35M.yaml --set pipeline.sft1.params.epochs=1
```

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
python -m foundation_lab run foundation_lab/core/configs/SAI_35M.yaml \
    --stages sft1,sft2 --init sft1=model/pretrained_35M.pt
```

### Lab kiểm tra gì trước khi chạy

`validate` (và `run`) kiểm tra toàn bộ trước khi nạp dữ liệu hay dựng model:

- schema của experiment và config model theo họ mô hình;
- `params` của từng stage theo recipe;
- config dữ liệu theo loại nguồn, và recipe có nhận loại dữ liệu đó không;
- họ mô hình có đủ capability mà recipe cần không;
- `init` dạng `<stage>.best` có trỏ tới stage đứng trước không.

> `base.yaml` còn một số điểm chưa tốt đã biết, ví dụ stage sft2 trộn lại dữ liệu mỗi epoch nên mẫu val/test có thể lọt vào train. Các điểm này sẽ thành tuỳ chọn ở mốc sau.

## Thư mục run

Mỗi lần `run` tạo một thư mục `runs/<tên experiment>/<thời gian>/`:

```text
runs/SAI_35M/20261023-091500/
├── resolved_config.yaml   # config đã resolve
├── lineage.json           # config_hash, commit git, file dữ liệu của từng stage
├── invocations.jsonl      # mỗi lần gọi run/resume: argv, stage, init, commit git
├── metrics.jsonl
└── stages/
    └── pretrain/
        ├── best.pt        # trọng số tốt nhất
        ├── last.ckpt.pt   # checkpoint để chạy tiếp
        └── done.json      # có file này nghĩa là stage đã xong
```

`lineage.json` chỉ ghi lúc tạo run; `invocations.jsonl` thêm một dòng cho mỗi lần gọi, nên biết được code nào đã chạy ở từng lần resume.

## Chạy tiếp với --resume

Khi bị ngắt, chạy lại với thư mục run. Lab đọc `resolved_config.yaml` của run đó và bỏ qua stage đã có `done.json`.

```bash
python -m foundation_lab run --resume runs/SAI_35M/20261021-120000
```

> Khi resume không truyền file experiment hay `--set`: run phải chạy tiếp đúng config đã ghi. Muốn đổi config thì tạo run mới.
