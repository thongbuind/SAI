---
title: Config
description: Lab đọc config YAML từ bốn thư mục, gộp kế thừa bằng extends, rồi kiểm tra nghiêm ngặt từng khoá trước khi chạy.
group: Bắt đầu
order: 11
---

## Config nằm ở đâu

| Thư mục | Chứa gì | Tham chiếu bằng |
| --- | --- | --- |
| `config/models/` | Config của model chính thức (`SAI_35M`, `SAI_100M`, `SAI_375M`) | `model: SAI_375M` |
| `foundation_lab/architecture/configs/` | Model thí nghiệm, thường kế thừa một model chính thức | `model: SAI_modular_375M` |
| `foundation_lab/data/configs/` | Nguồn dữ liệu của từng stage (`pretrain`, `sft1`, `sft2_mix`...) | `data: sft1` |
| `foundation_lab/core/configs/` | File experiment (`base.yaml`, `SAI_375M.yaml`...) | đường dẫn trên dòng lệnh |

Tên model được tìm trong cả hai thư mục model và phải có ở đúng một nơi; trùng tên thì lab báo lỗi thay vì tự chọn. Thay vì tên, có thể viết thẳng một mapping config.

## Kế thừa bằng extends

Một file khai báo `extends: <đường dẫn>` (tương đối theo chính file đó) để kế thừa file cha. Hai file được gộp đệ quy: mapping lồng nhau gộp từng khoá, giá trị khác thì của file con được dùng.

```yaml
# foundation_lab/architecture/configs/SAI_modular_375M.yaml
extends: ../../../config/models/SAI_375M.yaml
family: SAI_modular
attention: gqa
attention_kernel: sdpa
ffn: swiglu
norm: rmsnorm
```

## Kiểm tra nghiêm ngặt

Mỗi phần của config được đọc vào một schema có kiểu rõ ràng:

- Khoá lạ: báo lỗi, kèm danh sách khoá hợp lệ. Gõ sai `learning_rate` thành `learnig_rate` không thể trôi qua âm thầm.
- Thiếu khoá bắt buộc: báo lỗi.
- Sai kiểu: báo lỗi. Số nguyên được nhận cho trường số thực, nhưng `true`/`false` không được coi là số.
- Giá trị ngoài tập cho phép (vd. `runtime.device` khác `auto`, `cuda`, `mps`, `cpu`): báo lỗi.

## Override khi chạy

`--set khoa.con=gia_tri` ghi đè một giá trị mà không sửa file; giá trị được đọc như YAML, nên `--set runtime.seed=42`, `--set pipeline.sft1.params.freeze_blocks=[0, 1]` hay `--set runtime.seed=null` đều đúng kiểu.

## Thứ tự resolve

1. Đọc file và gộp chuỗi `extends`.
2. Thay tên config bằng nội dung: `model: SAI_375M` thành nội dung `config/models/SAI_375M.yaml`, `data: sft1` thành nội dung `foundation_lab/data/configs/sft1.yaml`.
3. Áp các override `--set`.
4. Gộp `stage_defaults` vào `params` của từng stage; giá trị riêng của stage được ưu tiên.

Vì `--set` áp sau bước 2, override nhắm vào nội dung đã resolve, ví dụ `--set model.dropout=0.1`, chứ không đổi được tên model.

Kết quả là một config tự đủ, không còn tham chiếu, được ghi thành `resolved_config.yaml` trong thư mục run. Xem trước bằng:

```bash
python -m foundation_lab validate foundation_lab/core/configs/SAI_375M.yaml --set pipeline.sft1.params.epochs=1
```
