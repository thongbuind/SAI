---
schema_version: 1
department: objectives
page_id: mask-normalization
title: "Token chịu loss, target và mẫu số"
description: "Phân biệt attention mask với loss weight và cách chuẩn hoá ở pretrain, SFT, train và eval."
order: 20
kind: concept
updated: "2026-10-25"
tags: [mask, SFT, pretrain, chuẩn hoá, microbatch]
sources:
  - foundation_lab/objectives/lm_ce.py
  - foundation_lab/data/batch.py
  - foundation_lab/recipes/causal_lm.py
  - foundation_lab/recipes/sft.py
  - foundation_lab/engine/step.py
  - foundation_lab/eval/lm_loss.py
  - tests/test_objectives.py
---

## Input và target lệch nhau một token

Dữ liệu đã chuẩn bị `labels[b,t]` là token cần dự đoán sau khi model đọc `input_ids[b,:t+1]`. Objective dùng target này trực tiếp. Shift thêm một lần trong objective sẽ khiến model học sai vị trí.

Ví dụ chuỗi token minh hoạ `[A, B, C, EOS]` có thể được ghép thành input `[A, B, C]`, target `[B, C, EOS]`. Đây chỉ là cách nhìn quan hệ vị trí; định dạng hội thoại và cách encode thực tế thuộc [Dữ liệu](../../data/docs/index.md).

## Hai loại mask không thay thế nhau

| Trường | Câu hỏi nó trả lời | Ví dụ |
| --- | --- | --- |
| `attention_mask` | Vị trí nào là token thật để model xử lý attention? | Prompt và câu trả lời đều có thể là 1 |
| `loss_weight` | Vị trí target nào đóng góp bao nhiêu vào loss? | Prompt SFT có thể là 0, câu trả lời/EOS là 1 |

Token prompt vẫn là ngữ cảnh hữu ích dù target tại một số vị trí không chịu loss. Loại prompt khỏi loss không đồng nghĩa xóa nó khỏi attention. Padding có trọng số loss bằng 0.

Với `lm_ce`, một hàng hidden có `token_weight=0` không nhận gradient trực tiếp qua loss của chính hàng đó; test hiện có kiểm tra điều này. Nhưng ở toàn model, token prompt vẫn có thể ảnh hưởng gradient vì token trả lời attend vào ngữ cảnh. Không diễn giải “mask loss” thành “model không học gì từ prompt”.

## Train chuẩn hoá theo mẫu số cố định

Engine hiện tính mỗi microbatch:

```text
scaled_loss = loss_sum / denominator / accumulation_steps
```

`denominator` do recipe quyết định, không nhất thiết là số token thật của microbatch hiện tại:

- `causal_lm`: số token train trung bình mỗi mẫu nhân `batch_size`.
- `sft`: `target_loss_tokens` do sampler cân bằng số token chịu loss quyết định.

Mẫu số cố định giữ một quy tắc tỷ trọng thống nhất khi logical batch có lượng token biến thiên. `accumulation_steps` là số logical batch tích luỹ cho cập nhật optimizer; nó khác `chunk_size` của phép tạo logits và khác cách engine chia microbatch theo ngân sách token.

## Eval chuẩn hoá trên toàn loader

Val/test cộng toàn bộ tổng CE có trọng số rồi chia tổng trọng số token trên toàn loader. Ví dụ hai nhóm có 2 token với tổng loss 2 và 8 token với tổng loss 16: loss chung là `(2+16)/(2+8)=1.8`; trung bình hai loss nhóm `(1+2)/2=1.5` không đúng trọng số theo token.

Train có objective chọn qua config, nhưng val/test hiện luôn đo `lm_ce`. Vì thế khi thử objective khác bản chất, không đồng nhất train loss với val loss chỉ vì cùng tên cột `loss`.

## Đối chiếu trong phòng bao phủ gì?

`compare-objective` hiện đặt tất cả `token_weight=1`; mask có kiểm chứng riêng trong test objective. Một kết quả Khớp trên web không chứng minh implementation mới xử lý đúng weight 0, weight phân số, padding hay dữ liệu SFT. Phạm vi test cần được xác định từ hợp đồng mới trước khi train.

Những quy tắc trên được đọc từ implementation ngày 25/10/2026; khi recipe hoặc batch contract thay đổi, phải cập nhật nội dung cùng lúc.
