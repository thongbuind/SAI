---
schema_version: 1
department: arch
page_id: transformer
title: "Từ token đến logits"
description: "Theo dõi luồng dữ liệu qua embedding, decoder block, residual và đầu ra ngôn ngữ."
order: 10
kind: concept
updated: "2026-10-25"
tags: [transformer, embedding, residual, lm_head, weight tying]
sources:
  - SAI/modeling.py
  - SAI/block.py
  - SAI/attention.py
  - SAI/ffn.py
---

## Bức tranh tổng thể

SAI là mô hình ngôn ngữ decoder-only có causal attention: ở vị trí hiện tại, model dự đoán token tiếp theo từ các token đã thấy. Token là đơn vị của tokenizer, có thể là một phần của từ; nó không đồng nghĩa với một từ tiếng Việt.

Quy ước: `B` là số chuỗi trong batch, `T` là số vị trí, `d_model` là chiều biểu diễn, `V` là kích thước từ vựng. Luồng forward của SAI hiện tại:

```text
input_ids (B,T), số nguyên
  → embedding (B,T,d_model)
  → nhiều decoder block (giữ nguyên shape)
  → RMSNorm cuối (B,T,d_model)
  → lm_head (B,T,V), logits
```

Model nhận ID token, không nhận trực tiếp chuỗi ký tự. Việc chuẩn bị nhãn, mask và chọn token chịu loss thuộc [Dữ liệu](../../data/docs/index.md) và [Objective](../../objectives/docs/index.md).

## Embedding: từ một ID sang một vector

Embedding là bảng tham số có `V` hàng, mỗi hàng dài `d_model`. Tra ID 42 lấy hàng 42; khoảng cách giữa hai ID không diễn tả khoảng cách ngữ nghĩa. Trong quá trình train, các hàng thay đổi nhờ gradient. Output embedding ở đây chưa biết toàn bộ ngữ cảnh; các block phía sau mới kết hợp thông tin giữa vị trí.

Tăng `V` làm bảng lớn hơn, đồng thời ảnh hưởng kích thước logits. Tăng `d_model` làm vector ở mỗi vị trí rộng hơn và làm nhiều ma trận trong model lớn hơn. Hai thay đổi này có hệ quả khác nhau; không thể chỉ so số tham số rồi coi hai model tương đương.

## Decoder block và đường residual

Mỗi block của SAI có hai bước, viết gọn:

```text
u = x + dropout(attention(norm1(x)))
y = u + dropout(ffn(norm2(u)))
```

Attention trao đổi thông tin giữa các token được phép nhìn. FFN xử lý vector ở từng vị trí. Dấu cộng là residual: đầu vào có đường đi trực tiếp qua block, còn nhánh con học phần bổ sung. Các tensor cộng với nhau cùng shape `(B,T,d_model)`. Norm đặt trước mỗi nhánh nên cấu trúc này gọi là pre-norm. Đây là cấu trúc đang dùng, không phải một nút chọn pre/post-norm trên web.

Dropout làm ngẫu nhiên một phần nhánh khi train; trong chế độ eval, dropout bị tắt. Vì vậy so hai lần forward trong chế độ train có dropout không phải phép kiểm chứng số học công bằng nếu chưa kiểm soát ngẫu nhiên.

## Logits và weight tying

`lm_head` chiếu vector cuối sang `V` điểm số. Logit lớn hơn biểu thị ưu tiên tương đối cao hơn trước softmax; logits không phải xác suất, cũng không tự quyết định câu trả lời cuối. Bộ giải mã còn chọn token bằng greedy, beam hoặc sampling.

SAI gán `lm_head.weight = embed.weight`: đầu vào và đầu ra dùng chung **cùng tham số**, không chỉ khởi tạo giống nhau. Weight tying giảm số tham số độc lập và gắn việc đọc token với việc dự đoán token. Đổi sang hai bảng riêng làm thay đổi cấu trúc học và yêu cầu kiểm chứng/checkpoint tương ứng; hiện không có option untie trong config.

Nguồn nền: [Transformer](https://arxiv.org/abs/1706.03762) giới thiệu kiến trúc attention; [nghiên cứu weight tying](https://arxiv.org/abs/1608.05859) phân tích việc dùng chung embedding đầu vào và đầu ra. Chi tiết luồng SAI ở trên được đối chiếu trực tiếp với code ngày 25/10/2026.
