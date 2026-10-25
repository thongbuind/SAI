---
schema_version: 1
department: data
page_id: split-mix
title: Split, trộn nguồn và rò rỉ đánh giá
description: Hiểu nguồn main/sub, mix_ratio, seed và điều kiện để so sánh loss có ý nghĩa.
order: 20
kind: concept
updated: "2026-10-25"
tags: [split, train, val, test, leakage, mix, seed]
sources:
  - foundation_lab/data/sources.py
  - foundation_lab/data/splits.py
  - foundation_lab/data/configs/pretrain.yaml
  - foundation_lab/data/configs/continued_mix.yaml
  - foundation_lab/data/configs/sft1.yaml
  - foundation_lab/data/configs/sft2_mix.yaml
  - foundation_lab/recipes/sft.py
  - foundation_lab/core/configs/base.yaml
---

## Ba tập có ba nhiệm vụ

Train tạo gradient để cập nhật trọng số. Validation, viết tắt val, theo dõi khả năng dự đoán ngoài tập train và chọn `best.pt`. Test phục vụ báo cáo cuối. Nếu dùng test để liên tục điều chỉnh thí nghiệm thì test không còn giữ vai trò kiểm tra độc lập của lựa chọn đó.

`split_train_val_test` hiện hoán vị **vị trí mẫu** rồi cắt theo `train_ratio`, `val_ratio`; phần còn lại là test. Mốc snapshot trong các data config là 95% train, 4% val và phần còn lại test. Số mẫu thực tế chịu phép lấy phần nguyên, vì vậy dataset nhỏ có thể có split rỗng. Tỷ lệ hiển thị trực tiếp trong web sau này phải đọc từ stage đã resolve.

## Trộn hoạt động thế nào?

`main` là nguồn chính; `sub` là nguồn bổ sung tùy chọn. Số mẫu lấy không hoàn lại từ sub là:

```text
min(floor(mix_ratio × số mẫu main), số mẫu sub)
```

Sau đó ghép với main, xáo trộn, rồi mới chia split. `mix_ratio` là tỷ lệ **số mẫu**, không phải tỷ lệ token. Với 100 mẫu main và `mix_ratio=3`, lấy tối đa 300 mẫu sub; nếu sub chỉ có 80 mẫu thì lấy 80. Hai nguồn có độ dài khác nhau sẽ cho tỷ lệ token khác tỷ lệ mẫu.

| Config snapshot | Main | Sub | Tỷ lệ lấy sub |
| --- | --- | --- | --- |
| `pretrain` | Manifest pretrain | Không có | Không trộn |
| `continued_mix` | Continued pretrain | Pretrain | 3 lần số mẫu main, bị chặn bởi sub |
| `sft1` | SFT1 | Không có | Không trộn |
| `sft2_mix` | SFT2 | SFT1 | 1 lần số mẫu main, bị chặn bởi sub |

`mix_seed` điều khiển chọn sub và xáo tập trộn; `split_seed` điều khiển hoán vị vị trí trước khi cắt. Giữ `split_seed` nhưng đổi dữ liệu hoặc `mix_seed` **không đảm bảo cùng một mẫu ở cùng split**.

## Giới hạn đã biết của implementation hiện tại

`data/sources.py` ghi nhận nguồn phụ được lấy từ toàn bộ nguồn, gồm cả mẫu từng thuộc val/test của stage trước. Do đó continued có thể học dữ liệu từng dùng đánh giá pretrain; SFT2 có thể học dữ liệu từng dùng đánh giá SFT1. Split rời nhau trong một lần gọi không bảo đảm độc lập xuyên stage.

`recipes/sft.py` còn ghi nhận: khi `resample_per_epoch=true`, mỗi epoch trộn với seed mới rồi split lại; val/test giữ từ lần trộn ban đầu. Mẫu trong val/test cố định có thể đi vào train ở epoch khác. `base.yaml` hiện bật cơ chế này cho SFT2. Hệ quả là loss đánh giá có thể lạc quan và chọn best bị lệch; không dùng những số này như bằng chứng đánh giá độc lập.

Về nguyên tắc, các mẫu có quan hệ gần nhau như nhiều cửa sổ của một tài liệu cũng cần được xem xét cùng nhóm trước khi đánh giá tính độc lập. Bộ chia hiện tại chỉ biết vị trí mẫu, không chứng minh sự độc lập nội dung. Đây là điều cần kiểm tra khi thiết kế thí nghiệm, không phải cam kết rằng Lab đã có group split hay deduplication.

## Khi chuẩn bị so sánh

Ghi lại nguồn, thứ tự shard, tokenizer, cách chunk, seed và cấu hình resample. Đối chiếu cùng một baseline lấy từ config hiện hành. Nếu đổi tập trộn, đừng quy toàn bộ chênh lệch loss cho recipe hoặc model. Xem [quy trình sử dụng](workflow.md) và [cách đọc kết quả](read-results.md).
