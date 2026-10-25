---
schema_version: 1
department: tokenizer
page_id: algorithms
title: SentencePiece, BPE, Unigram và đánh đổi
description: Phân biệt thư viện tokenizer, thuật toán phân đoạn và encoder đã đăng ký trong Lab.
order: 10
kind: concept
updated: "2026-10-25"
tags: [SentencePiece, BPE, unigram, vocabulary, lựa chọn]
sources:
  - foundation_lab/tokenizer/train.py
  - foundation_lab/tokenizer/registry.py
  - foundation_lab/tokenizer/encode.py
  - foundation_lab/cli.py
---

## Ba tầng dễ nhầm

**SentencePiece** là bộ công cụ train và dùng tokenizer; **BPE/Unigram** là thuật toán tạo và dùng các đơn vị con của từ; **pretrain/sft** trong registry Lab là quy tắc biến mẫu thô thành X/Y/mask. Một encoder SFT có thể dùng vocabulary từ một thuật toán tokenizer, nhưng hai khái niệm này không thay thế nhau. SentencePiece hỗ trợ BPE và Unigram, có thể học từ văn bản thô. [Nguồn dự án SentencePiece](https://github.com/google/sentencepiece).

## Các cách chia văn bản

| Cách chia | Ý tưởng | Điểm mạnh thường nhắm tới | Đánh đổi cần đo |
| --- | --- | --- | --- |
| Theo ký tự | Mỗi ký tự là một đơn vị | Bộ đơn vị nhỏ, dễ giải thích | Chuỗi dài; một ký tự Unicode không đồng nghĩa một byte |
| Theo từ | Dùng từ làm đơn vị | Từ phổ biến có thể rất gọn | Vocabulary lớn, khó xử lý từ hiếm/mới |
| BPE | Ghép các cặp đơn vị thường gặp để tạo subword | Quy tắc ghép rõ, xử lý từ hiếm qua các mảnh nhỏ | Phân đoạn phụ thuộc corpus và quy tắc ghép |
| Unigram | Chọn phân đoạn theo mô hình xác suất trên subword | Có cách diễn đạt nhiều phân đoạn có thể có | Độ nén và chất lượng phụ thuộc corpus, vocab và chế độ encode |

BPE được áp dụng để biểu diễn từ hiếm bằng chuỗi subword trong [Sennrich và cộng sự, 2016](https://aclanthology.org/P16-1162/). Unigram và việc lấy mẫu nhiều phân đoạn được trình bày trong [Kudo, 2018](https://aclanthology.org/P18-1007/). Các lợi ích nghiên cứu đó không chứng minh một thuật toán sẽ tốt hơn trên bộ tiếng Việt của SAI; cần đo trên cùng dữ liệu và quy trình.

## Vì sao vocabulary lớn chưa chắc tốt hơn?

Vocabulary lớn có thể biểu diễn cụm phổ biến bằng ít token hơn, nhưng embedding và LM head cần quản lý nhiều hàng hơn; token hiếm nhận ít lần cập nhật. Vocabulary nhỏ tiết kiệm số hàng, đổi lại có thể làm chuỗi dài hơn. So sánh chỉ số token mà bỏ chi phí model sẽ thiếu một phần đánh đổi.

Không chọn tokenizer chỉ từ một câu mẫu. Cần có tập đại diện gồm tiếng Việt, dấu câu, số, xuống dòng, thuật ngữ, mã và ký tự hiếm; giữ quy tắc chuẩn hóa giống nhau. Khi đo theo từ tiếng Việt, phải định nghĩa “từ” rõ vì dấu cách không luôn phân cách một từ hoàn chỉnh.

## Những lựa chọn thực sự có trong Lab

`TRAINER_ARGS` hiện cố định `model_type=unigram`. CLI `train-tokenizer` nhận input, output directory và vocab size; không có cờ chọn BPE. `ENCODERS` hiện có `pretrain`, `sft`, và `encode` chọn encoder bằng tên đó. Không thử truyền `bpe` vào vị trí encoder và không mô tả nó như một option UI đang hoạt động.

Unigram có khả năng lấy mẫu phân đoạn trong thuật toán nền; đường encode hiện gọi `tokenizer.encode(..., out_type=int)` và không bật sampling/regularization. Một khả năng thư viện hỗ trợ chưa đồng nghĩa Lab đang dùng nó.

Tiếp theo: [Tokenizer chính thức](official-tokenizer.md).
