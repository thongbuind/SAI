---
schema_version: 1
department: tokenizer
page_id: read-results
title: Đánh giá tokenizer và câu hỏi thường gặp
description: Đọc số token, mức phân mảnh, byte fallback và log encode mà không nhầm với chất lượng model.
order: 60
kind: interpretation
updated: "2026-10-25"
tags: [đánh giá, FAQ, compression, byte fallback, skipped]
sources:
  - foundation_lab/tokenizer/encode.py
  - foundation_lab/tokenizer/train.py
  - foundation_lab/tokenizer/chunking.py
  - foundation_lab/data/io.py
  - web/site/src/lib/lab/workbench.ts
---

## Hiện có log encode, chưa có benchmark web

`encode_file` trả `n_documents`, `n_samples`, `n_tokens`, `skipped`. `encode_shards` ghi metadata từng shard và tổng vào manifest. Phòng web `compare-tokenizer` chưa triển khai; các metric so sánh dưới đây là hướng dẫn thiết kế phép đo, không phải kết quả đo hay field API hiện có.

| Số liệu | Cách hiểu đúng |
| --- | --- |
| `n_documents` | Mẫu thô giữ lại, mỗi mẫu sinh ít nhất một mẫu encode |
| `n_samples` | Tổng cửa sổ/mẫu sinh ra; pretrain dài có thể tăng số này |
| `n_tokens` | Tổng chiều dài X, có token cấu trúc và phần overlap |
| `skipped` | Bộ đếm theo lý do reader/encoder bỏ mẫu |

Số token ở manifest không phải tổng token chịu loss SFT. Không cộng số prompt vào lượng giám sát rồi so với pretrain như hai đơn vị giống hệt nhau.

## Một phép so tokenizer cần kiểm gì?

| Góc nhìn | Cách đo cần công bố | Ý nghĩa và giới hạn |
| --- | --- | --- |
| Độ dài | Tổng token cùng tập văn bản sau cùng tiền xử lý | Ít token có thể giảm độ dài chuỗi; chưa nói chất lượng model |
| Độ nén | Byte UTF-8 hoặc ký tự chuẩn hóa chia token | Phải ghi đơn vị, có/không special tokens |
| Phân mảnh | Token trên đơn vị văn bản đã định nghĩa | Tiếng Việt cần định nghĩa từ/từ cách bằng khoảng trắng rõ |
| Byte fallback | Tỷ lệ token byte trên token nội dung | Cao có thể báo vocabulary chưa bao phủ tốt nhóm văn bản đó |
| Mất thông tin | So văn bản decode với bản sau tiền xử lý và đối chiếu bản gốc | Phân biệt tác động tokenizer với lowercase/strip |
| Hiệu năng | Thời gian cùng phần cứng, phiên bản thư viện, số lần lặp | Không lấy một lần chạy làm kết luận ổn định |

Không lấy văn bản train tokenizer làm tập duy nhất để lựa chọn. Cần xem từng miền dữ liệu, câu dài và đuôi phân bố; giá trị trung bình có thể che một nhóm ký tự bị phân mảnh mạnh.

## Câu hỏi thường gặp

### Vì sao encode lại cho số mẫu khác?

Kiểm model context, tokenizer, thứ tự/file input, chunking và số mẫu bỏ. Pretrain dài hiện được chia cửa sổ; SFT quá dài bị bỏ. Cùng số tài liệu không bảo đảm cùng số mẫu hoặc token.

### Byte fallback có phải lỗi không?

Không tự động là lỗi: đó là cơ chế biểu diễn ký tự chưa có piece phù hợp. Tuy nhiên nhiều byte token làm chuỗi dài hơn. Đọc cùng ví dụ văn bản và tỷ lệ theo miền; không xóa mọi mẫu có fallback mà chưa hiểu dữ liệu.

### Vì sao chữ hoa biến mất dù normalization là identity?

SAI gọi `.lower()` trước encode. Quy tắc wrapper khác với quy tắc normalizer trong SentencePiece. Nếu muốn nghiên cứu giữ chữ hoa, đó là thay đổi pipeline và phân phối đầu vào cần được thiết kế, không chỉ thay một nhãn UI.

### Cùng vocabulary size có dùng checkpoint cũ được không?

Không đủ. ID 123 có thể mang piece khác, khiến embedding/head đọc sai ý nghĩa. Cần tokenizer tương thích cụ thể, không chỉ cùng kích thước ma trận.

### Có thể dùng “sft” thay cho “unigram” không?

Không: `sft` là encoder dữ liệu, `unigram` là thuật toán tokenizer. Xem [bảng phân biệt](algorithms.md).

Quay lại [Tổng quan](index.md) hoặc xem [Dữ liệu](../../data/docs/index.md).
