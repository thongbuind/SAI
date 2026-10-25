---
schema_version: 1
department: state
page_id: versions
title: Phiên bản và điều kiện dùng lại kết quả
description: Phân biệt phiên bản lab với bản phát hành SAI và áp đúng chính sách resume, init, so sánh.
order: 60
kind: reference
updated: "2026-10-25"
tags: [version, tương thích, baseline, checkpoint]
sources:
  - AGENTS.md
  - foundation_lab/__init__.py
  - SAI/version.py
  - web/site/src/content/lab/versioning.md
  - foundation_lab/state/lineage.py
---

## Hai dãy phiên bản độc lập

Snapshot working tree ngày 25/10/2026: `foundation_lab.__version__ = "2.1.0"`, thường gọi là **lab2.1.0**; bản phát hành `SAI.MODEL_VERSION = "dev3.0.1"`. Một số thành phần có thể giữ version cũ nếu không thay đổi trong lần phát hành đó.

| Dãy | Đối tượng | Nơi truy nguồn |
| --- | --- | --- |
| `labx.y.z` | Hành vi thí nghiệm, spec và hợp đồng lab | Phiên bản lab và git trong lineage |
| `dev3.y.z` | Trọng số, package SAI và adapter phát hành | `SAI/version.py`, metadata bản phát hành |

Không so số 2.1.0 với 3.0.1 để kết luận mới/cũ. Tăng một dãy không kéo theo dãy còn lại. Tài liệu không là nguồn version động cho giao diện.

## Chính sách của lab

| Loại thay đổi | Ý nghĩa | Run và checkpoint cũ |
| --- | --- | --- |
| Tăng z | Thêm khả năng tương thích, không đổi kết quả cũ | Vẫn resume, init và so sánh như bình thường |
| Tăng y | Spec hoặc kết quả cũ không còn tái lập theo cùng nghĩa | Chỉ tham khảo; không resume hay so trực tiếp; chạy lại baseline |
| Tăng x | Tổ chức lại lab hoặc hợp đồng ở quy mô lớn | Kết quả trước đó chỉ tham khảo; baseline chạy lại |

Khi lab tăng y mà muốn init bằng checkpoint cũ, phải chứng minh tương thích qua test hoặc model tí hon trước. Nếu cần chuyển đổi, đó là một script riêng cần được duyệt. Khi tăng x, không tự viết tương thích ngược trừ khi người dùng yêu cầu.

Đây là chính sách sử dụng kết quả, không phải khẳng định mọi lệnh CLI tự chặn từng trường hợp. Người đọc phải kiểm tra version/commit và phạm vi thay đổi trước khi gọi resume hoặc so sánh.

## Chính sách bản phát hành SAI

- Tăng `z`: checkpoint cũ còn dùng, không train lại. Nếu thay generate, đánh giá câu trả lời cũ phải chạy lại vì cùng input/config có thể cho hành vi khác.
- Tăng `y`: kiến trúc/phép tính/tokenizer đổi theo tiêu chí phát hành; checkpoint cũ không dùng với code mới, cần train lại. Không ghi checkpoint mới đè file cũ.
- Refactor không đổi hành vi, comment, test và tài liệu không tự làm tăng phiên bản phát hành.

Tag cũ trên Hub được giữ nguyên. Không xóa hoặc gắn lại tag để làm lịch sử cũ có vẻ giống hiện tại.

## Quy trình quyết định dùng run cũ

1. Xác định mục đích: tham khảo, so trực tiếp, init hay resume.
2. Đọc `lab_version`, git và dirty trong lineage; đọc invocation nếu có resume.
3. Kiểm tra config, tokenizer, dữ liệu và kiến trúc checkpoint.
4. Áp chính sách tăng x/y/z; nếu cần baseline mới, giữ run cũ nguyên trạng và tạo run khác.
5. Ghi rõ phiên bản/commit trong kết luận so sánh.

Nếu cần tái hiện đúng code cũ, người dùng tự checkout commit thích hợp và giữ cả các thay đổi chưa commit đã từng được dùng nếu có. Agent không đổi branch, commit hoặc push thay người dùng trong workspace này.

Không dọn, sửa, di chuyển hay ghi đè run/checkpoint cũ để “chuẩn hoá” dữ liệu. Việc dọn dẹp do người dùng quyết định. Đọc [lineage](lineage.md) để hiểu giới hạn bằng chứng đang được lưu.
