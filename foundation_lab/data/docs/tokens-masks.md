---
schema_version: 1
department: data
page_id: tokens-masks
title: Token, nhãn và hai loại mask
description: Phân biệt token đầu vào, nhãn đã shift, attention mask và trọng số loss.
order: 10
kind: concept
updated: "2026-10-25"
tags: [token, labels, mask, shift, SFT]
sources:
  - foundation_lab/data/batch.py
  - foundation_lab/data/loaders.py
  - foundation_lab/tokenizer/encode.py
  - foundation_lab/objectives/lm_ce.py
  - SAI/modeling.py
  - tests/test_data.py
---

## Causal LM học dự đoán token tiếp theo

Token ID là chỉ số trong vocabulary, không phải mã ký tự hay số biểu diễn ý nghĩa. Một từ có thể gồm nhiều token. Với chuỗi minh hoạ `BOS A B EOS`, mẫu pretrain có:

| Vị trí | 0 | 1 | 2 |
| --- | --- | --- | --- |
| `input_ids` | BOS | A | B |
| `labels` | A | B | EOS |
| `loss_weight` | 1 | 1 | 1 |

Tại vị trí 1, model đã thấy `BOS A` và được chấm điểm khi dự đoán B. Nhãn trong Lab **đã shift một lần tại bước encode**. Không dịch thêm nhãn trong collator, model hay objective.

## Hợp đồng `LMBatch`

Sau collate, các trường tensor có cùng shape `(B, T)`: B là số mẫu trong batch, T là độ dài đã padding. `input_ids` và `labels` dùng số nguyên long; hai mask dùng số thực.

| Trường | Câu hỏi nó trả lời | Giá trị |
| --- | --- | --- |
| `attention_mask` | Vị trí đầu vào có phải token thật? | 1 thật, 0 padding |
| `loss_weight` | Dự đoán ở vị trí này có đóng góp vào loss? | Hiện tại 1 hoặc 0 |
| `has_padding` | Batch có ít nhất một vị trí padding? | Boolean |
| `padding_side` | Phần đệm đặt ở đâu? | Collator hiện dùng `right` |

Prompt trong SFT là token thật: `attention_mask=1`, nhưng `loss_weight=0`. Model vẫn dùng prompt làm ngữ cảnh để trả lời. Mask loss bằng 0 không có nghĩa là xóa token khỏi phép attention.

## SFT khác pretrain ở đâu?

Encoder SFT giữ `X = BOS + hội thoại + EOS`, còn `Y = X[1:]`, nên Y và mask ngắn hơn X một vị trí. Collator đệm Y và mask tới T; vị trí cuối cùng của X có trọng số loss bằng 0. Cách lưu này khác mẫu pretrain nhưng batch đầu ra vẫn có shape đồng nhất.

Mask SFT phủ cả segment của role `model`, gồm dấu mở/đóng segment và phần role, cùng EOS cuối. Segment `system` và `user` không chịu loss. Xem [chat template](../../tokenizer/docs/index.md) để hiểu ranh giới token.

## Không suy ra padding từ ID 0

Trong cấu hình hiện tại, giá trị dùng lấp padding là 0, cũng là ID `[UNK]` của tokenizer chính thức. Vì vậy `token_id == 0` không đủ xác định padding. Collator nhận diện token thật theo **độ dài mẫu**, và `lm_ce` nhân loss từng token với `loss_weight`.

Implementation `SAIModel` hiện tôn trọng `attention_mask`, kết hợp nó với causal mask và dùng mask để suy ra vị trí. Pipeline dữ liệu của Lab vẫn tạo padding phải; không đổi chiều padding chỉ dựa trên khả năng riêng của một model.

## Ví dụ đọc loss

Nếu có 5 dự đoán nhưng mask chỉ bật 2 vị trí, số token chịu loss là 2. Tổng CE có trọng số là tổng lỗi tại hai vị trí đó; loss báo cáo theo token lấy tổng này chia tổng trọng số. CE đo mức độ model gán xác suất cho nhãn đúng; giá trị thấp hơn có ý nghĩa khi cùng dữ liệu, tokenizer và mask. [Tài liệu CrossEntropyLoss của PyTorch](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) trình bày CE và các cách reduction.

Tiếp theo: [Gom batch và padding](batch-padding.md).
