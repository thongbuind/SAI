---
schema_version: 1
department: data
page_id: storage-manifest
title: NPZ, shard và manifest
description: Đọc đúng cấu trúc dữ liệu đã encode và giới hạn bộ nhớ của đường nạp hiện tại.
order: 40
kind: reference
updated: "2026-10-25"
tags: [NPZ, manifest, shard, offsets, lineage]
sources:
  - foundation_lab/data/io.py
  - foundation_lab/data/sources.py
  - foundation_lab/tokenizer/encode.py
---

## Một dataset có thể gồm nhiều file

Một file `.npz` chứa nhiều mẫu dài ngắn khác nhau. Manifest `.json` liệt kê các shard `.npz`; khi đọc, các shard được ghép thành một dataset logic. Shard là đơn vị lưu trữ, không đồng nghĩa một split train/val/test.

| Trường NPZ | Ý nghĩa |
| --- | --- |
| `X_flat`, `X_offsets` | Các input token nối thành mảng phẳng và ranh giới từng mẫu |
| `Y_flat`, `Y_offsets` | Các target đã shift và ranh giới từng mẫu |
| `lengths` | Độ dài X của từng mẫu |
| `M_flat`, `M_offsets` | Mask target cho SFT; chỉ có khi encoder ghi mask |

Ví dụ X gồm `[1,5,6]` và `[1,7]` thì `X_flat=[1,5,6,1,7]`, `X_offsets=[0,3,5]`. Mẫu thứ i là đoạn từ offsets[i] đến trước offsets[i+1]. X và Y có offsets riêng vì SFT lưu Y ngắn hơn X một vị trí.

## Hợp đồng manifest hiện tại

Mỗi phần tử trong `shards` có `name`, `file`, `n_samples`, `n_tokens`. Trường `file` được giải theo **thư mục chứa manifest**, không theo thư mục gốc repo. `total_samples` và `total_tokens` là tổng metadata của các shard.

`n_tokens` hiện đếm tổng độ dài X đã encode. Nó không phải số token văn bản gốc duy nhất, cũng không phải tổng mask SFT: X có token cấu trúc, pretrain dài có overlap, SFT có prompt. Xem [cách đọc số liệu](read-results.md).

## Những gì manifest không bảo đảm

Manifest hiện không phải hồ sơ đầy đủ của tokenizer, fingerprint nội dung, quyền sử dụng dữ liệu, hay danh sách split cố định. Không suy những thuộc tính đó chỉ từ tên file. Run thu thập lineage các file thực sự đọc; đối với manifest, nguồn dữ liệu liệt kê các shard để lineage biết đầu vào vật lý.

## Ghi theo hai lượt, đọc vẫn dùng RAM

`write_npz` duyệt dữ liệu hai lần: lượt đầu đếm độ dài để cấp phát, lượt sau điền mảng. Reader đưa vào writer phải trả iterator mới mỗi lượt và cho cùng chuỗi mẫu. Cách này giảm nhu cầu giữ mọi mẫu dưới dạng danh sách khi encode, nhưng vẫn cấp phát các mảng đầu ra trong RAM.

`load_npz` đọc shard và tái dựng các mẫu; với manifest, implementation đọc tất cả shard rồi ghép mảng. Vì vậy chia shard **không đồng nghĩa streaming hay giới hạn RAM khi train**. Trên máy ít RAM, không ước lượng khả năng chạy chỉ từ kích thước nén của NPZ.

## Tính bất biến của dữ liệu encode

Writer từ chối ghi đè NPZ và manifest đã tồn tại. Khi đổi tokenizer, cách chunk hoặc mask, nên tạo bộ đầu ra mới được duyệt và giữ thông tin provenance. Không đổi nội dung file cũ rồi dùng cùng tên để so sánh run; đường dẫn giống nhau không chứng minh dữ liệu giống nhau.

Tiếp theo: [Quy trình sử dụng](workflow.md).
