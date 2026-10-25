---
schema_version: 1
department: tokenizer
page_id: overview
title: Sổ tay phòng Tokenizer
description: Hiểu tokenizer của SAI, cách encode pretrain và hội thoại, cùng giới hạn so sánh hiện tại.
order: 0
kind: overview
updated: "2026-10-25"
tags: [tokenizer, bắt đầu, encode]
sources:
  - foundation_lab/tokenizer/__init__.py
  - foundation_lab/tokenizer/registry.py
  - foundation_lab/tokenizer/encode.py
  - foundation_lab/tokenizer/train.py
  - foundation_lab/cli.py
  - web/site/src/lib/lab/workbench.ts
---

## Tokenizer là cửa vào của model

Tokenizer biến văn bản thành token ID để model đọc. Detokenizer đổi ID thành văn bản để người dùng đọc. Cách chia token ảnh hưởng độ dài ngữ cảnh, vocabulary, chi phí tính toán và tính tương thích giữa trọng số với đầu vào.

Trong Lab có hai công việc khác nhau: **train tokenizer** tạo vocabulary và quy tắc phân đoạn mới; **encode dữ liệu** dùng tokenizer đã có để tạo X/Y/mask phục vụ huấn luyện model. Encode không phải train tokenizer, cũng không cập nhật trọng số SAI.

## Đọc theo nhu cầu

| Câu hỏi | Trang |
| --- | --- |
| BPE, Unigram và SentencePiece khác nhau ra sao? | [Thuật toán và đánh đổi](algorithms.md) |
| SAI đang dùng gì; BOS/EOS và byte fallback là gì? | [Tokenizer chính thức của SAI](official-tokenizer.md) |
| Dữ liệu hội thoại dùng format nào; chấm loss ở đâu? | [Chat template và mask SFT](chat-masking.md) |
| Bài dài bị cắt thế nào, overlap có ích và tốn gì? | [Chia cửa sổ pretrain](chunking.md) |
| Các lệnh encode và train-tokenizer dùng thế nào? | [Quy trình sử dụng](workflow.md) |
| Đọc số token và lựa chọn tokenizer ra sao? | [Đánh giá và câu hỏi thường gặp](read-results.md) |

## Trạng thái hiện tại

Snapshot ngày 25/10/2026: CLI đã có `encode` và `train-tokenizer`. Registry `ENCODERS` đăng ký `pretrain` và `sft`. Đây là **kiểu xử lý dữ liệu**, không phải tên thuật toán tokenizer. Trainer hiện khai báo SentencePiece Unigram; chưa có registry trong phòng này cho phép chọn BPE/Unigram bằng config.

Trang web Tokenizer hiện ở trạng thái dự kiến với tên `compare-tokenizer`. Không có nút chạy so sánh thật hoặc báo cáo benchmark tokenizer trong giao diện hiện tại. Tài liệu giải thích cách đánh giá là hướng dẫn phương pháp, không phải kết quả đo đã có.

## Giữ đúng đường ranh với các phòng khác

Phòng Tokenizer đọc văn bản thô và tạo mẫu encode; [Dữ liệu](../../data/docs/index.md) quản lý file encode, nguồn, split và batch. [Phương pháp huấn luyện](../../recipes/docs/index.md) dùng batch đó để cập nhật model. Khi thay tokenizer, vocabulary và ý nghĩa ID có thể đổi; không gắn tokenizer mới vào checkpoint cũ chỉ vì chúng có cùng số token.
