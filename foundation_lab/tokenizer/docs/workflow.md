---
schema_version: 1
department: tokenizer
page_id: workflow
title: Quy trình encode và train tokenizer
description: Sử dụng đúng CLI hiện có, chọn đầu ra mới và kiểm tra số liệu trước khi giao dữ liệu cho recipe.
order: 50
kind: guide
updated: "2026-10-25"
tags: [hướng dẫn, encode, train-tokenizer, CLI, an toàn dữ liệu]
sources:
  - foundation_lab/cli.py
  - foundation_lab/tokenizer/train.py
  - foundation_lab/tokenizer/encode.py
  - foundation_lab/data/io.py
  - web/site/src/lib/lab/workbench.ts
---

## Chọn đúng công việc

Nếu cần biến dữ liệu thành X/Y/mask dùng với SAI hiện tại, chọn **encode bằng tokenizer hiện tại**. Nếu nghiên cứu vocabulary hoặc thuật toán mới, đó là **train tokenizer mới** và cần một kế hoạch model tương thích riêng. Phòng web đang dự kiến `compare-tokenizer`; các thao tác dưới đây là CLI hiện có, không phải luồng đã chạy được trên web.

## Encode một bộ dữ liệu

1. Xác định schema thô: JSONL có trường `text` cho pretrain; JSON hội thoại theo `instruction/history` cho SFT.
2. Chọn tokenizer và model config. `--model` dùng config model để lấy `max_seq_len`, không nạp trọng số model để encode.
3. Chọn đầu ra mới. Một input có thể ghi một NPZ; nhiều input dùng output directory và manifest. Không ghi đè bộ dữ liệu cũ.
4. Sau khi được phép chạy, đọc số tài liệu giữ lại, số mẫu encode, số token và từng lý do bỏ mẫu.
5. Giữ thông tin nguồn/tokenizer/config encode cùng thí nghiệm. Chỉ sau đó mới tham chiếu đầu ra trong config dữ liệu được duyệt.

Ví dụ dưới đây minh hoạ cú pháp với đường dẫn dự án; đây là **lệnh tham khảo, chưa thực hiện**. Đầu ra `handbook_example` phải được chọn mới và được người dùng duyệt trước khi chạy thực tế.

```bash
python -m foundation_lab encode pretrain --input data/raw/pretrain_data_001.jsonl --output data/processed/handbook_example_pretrain_ids.npz --model SAI_35M
python -m foundation_lab encode sft --input data/raw/SFT_1.json --output data/processed/handbook_example_sft_ids.npz --model SAI_35M
```

Để ghi nhiều shard, dùng `--input` với nhiều file, `--output-dir` và `--manifest`. `--output` chỉ nhận đúng một input. `--tokenizer` tùy chọn thay đường dẫn tokenizer; nếu bỏ qua, dùng `SAI/tokenizer.model`.

## Train tokenizer thử nghiệm

Trainer hiện đọc JSONL `text`, chuyển chữ thường, ghi dữ liệu tạm rồi gọi SentencePiece với `TRAINER_ARGS`. Thư mục output phải chưa tồn tại; tokenizer chính thức không bị writer ghi đè. Ví dụ cú pháp, chưa chạy:

```bash
python -m foundation_lab train-tokenizer --input data/raw/pretrain_data_001.jsonl --output-dir runs/tokenizers/handbook_example
```

Có thể truyền `--vocab-size`, nhưng không có cờ CLI đổi `model_type`. Output là `tokenizer.model` và `tokenizer.vocab` trong thư mục mới. Việc tạo được tokenizer không chứng minh nó tốt hơn baseline hoặc tương thích checkpoint cũ.

## Giới hạn tài nguyên và quyền chạy

Encode duyệt input hai lần để ghi NPZ. Train tokenizer và encode đều có thể đọc dữ liệu lớn, cấp phát nhiều RAM và tạo file. Theo quy tắc workspace hiện tại, phải xin phép trước khi chạy, đặc biệt trên máy 8GB RAM; mở Sổ tay hoặc bấm copy lệnh không phải sự chấp thuận chạy. Phiên viết tài liệu này không chạy các lệnh trên.

## Sau khi có kết quả

So trên cùng bộ văn bản với tokenizer chính thức, cùng tiền xử lý và cách tính metric. Đọc [đánh giá tokenizer](read-results.md). Nếu đổi tokenizer dùng huấn luyện SAI, cần xử lý tương thích vocabulary, dữ liệu encode, trọng số và phiên bản phát hành trong một thay đổi riêng được duyệt.
