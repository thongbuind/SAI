---
schema_version: 1
department: tokenizer
page_id: official-tokenizer
title: Tokenizer chính thức của SAI
description: Tra cứu cấu hình trainer khai báo, token đặc biệt, chuẩn hóa và yêu cầu tương thích.
order: 20
kind: reference
updated: "2026-10-25"
tags: [SAI, mặc định, special tokens, byte fallback, lowercase]
sources:
  - foundation_lab/tokenizer/train.py
  - foundation_lab/tokenizer/encode.py
  - SAI/generate.py
  - tests/test_encode.py
---

## Nguồn mặc định và phạm vi snapshot

Tokenizer phát hành của SAI là `SAI/tokenizer.model`. Bảng dưới lấy từ khai báo `TRAINER_ARGS`, `OFFICIAL_VOCAB_SIZE`, mã encode/generate và test nguồn ngày 25/10/2026; phiên biên soạn không nạp file model tokenizer hay chạy test để xác minh lại binary. Khi làm một phép so sánh thật, baseline phải là tokenizer phát hành đang chọn và có định danh artifact rõ ràng.

| Khai báo trong code | Snapshot | Tác dụng cần hiểu |
| --- | --- | --- |
| `model_type` | `unigram` | Thuật toán tokenizer |
| `OFFICIAL_VOCAB_SIZE` | 10.000 | Kích thước vocab mặc định khi train mới |
| `normalization_rule_name` | `identity` | Không dùng quy tắc chuẩn hóa mặc định kiểu NFKC trong trainer này |
| Chữ hoa/thường ở bước train/encode | Chuyển thành chữ thường | Khớp `GenerationConfig.lowercase=True` mặc định |
| `byte_fallback` | `true` | Có biểu diễn byte cho ký tự ngoài mảnh đã học |
| `character_coverage` | 0,9999 | Tham số bao phủ ký tự của trainer |
| `split_digits` | `true` | Quy tắc tách chữ số của tokenizer |
| `remove_extra_whitespaces` | `false` | Không chủ ý gộp khoảng trắng dư bằng tùy chọn này |
| `max_sentencepiece_length` | 8 | Giới hạn độ dài piece trong trainer |
| `input_sentence_size` | 1.000.000 | Giới hạn tập câu lấy vào trainer; không phải context model |

Các con số là ảnh chụp khai báo ngày 25/10/2026. Khi đánh giá hoặc encode, kiểm code và artifact tokenizer thực sự được chọn cho lần chạy; không coi bảng này là bằng chứng mọi tokenizer mang tên SAI đều có cấu hình giống nhau.

## Token đặc biệt

| Piece | ID được source/test khai báo | Vai trò |
| --- | --- | --- |
| `[UNK]` | 0 | Token không biết; trùng giá trị dùng lấp padding trong collator |
| `[BOS]` | 1 | Bắt đầu chuỗi |
| `[EOS]` | 2 | Kết thúc chuỗi |
| `<|im_start|>` | 3 | Mở segment hội thoại |
| `<|im_end|>` | 4 | Đóng segment hội thoại |

Trainer đặt tên `pad_piece` nhưng không bật `pad_id`; theo comment nguồn, tokenizer chính thức không có `[PAD]` riêng. Đừng đồng nhất ID 0 với padding: loader dựa độ dài và mask. `SpecialTokens.from_tokenizer` tra piece rồi đối chiếu ngược để phát hiện token đặc biệt bị thiếu; không chỉ kiểm ID khác âm.

## Byte fallback giữ khả năng biểu diễn, không giữ độ ngắn

Khi một ký tự không được vocabulary biểu diễn trực tiếp, byte fallback dùng chuỗi token byte UTF-8. Ký tự hiếm có thể cần nhiều token, làm tăng độ dài dù không bị mất hoàn toàn thành UNK. [Tài liệu special symbols của SentencePiece](https://google.github.io/sentencepiece/doc/special_symbols/).

Các piece byte nhìn như `<0xAB>` là cách ghi trong vocabulary. Trong chunking, byte ngoài ASCII được biểu diễn bằng ký tự thay thế chỉ để xét ranh giới; đó không phải phép decode văn bản cho người dùng.

## “Identity” không có nghĩa văn bản không đổi

Wrapper của SAI vẫn `.lower()` trước train và encode; reader còn strip đầu/cuối mẫu. Vì thế chữ hoa và một số khoảng trắng biên không giữ nguyên qua pipeline. Đánh giá round-trip phải so với văn bản sau tiền xử lý đã công bố, đồng thời báo rõ những phần thông tin bị chuẩn hóa.

Thay tokenizer hoặc ý nghĩa token ID ảnh hưởng embedding và đầu ra model. Cùng vocab size không chứng minh tương thích trọng số; tokenizer mới cần một quyết định huấn luyện/phát hành riêng. Xem [quy trình sử dụng](workflow.md).
