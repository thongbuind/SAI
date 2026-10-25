---
schema_version: 1
department: arch
page_id: position-cache
title: "Vị trí, RoPE và KV cache"
description: "Hiểu mã hoá vị trí, giới hạn context và cách tái sử dụng key/value khi sinh văn bản."
order: 30
kind: concept
updated: "2026-10-25"
tags: [RoPE, vị trí, context, KV cache, padding]
sources:
  - SAI/rope.py
  - SAI/modeling.py
  - SAI/attention.py
  - foundation_lab/architecture/layers/block.py
  - foundation_lab/architecture/models/modular.py
---

## Vì sao cần thông tin vị trí?

Một tập vector token chưa đủ để phân biệt các thứ tự như “mèo đuổi chó” và “chó đuổi mèo”. Attention cần thông tin về vị trí để xử lý chuỗi. RoPE đưa thông tin này vào query và key bằng phép xoay phụ thuộc chỉ số vị trí; không cộng thêm một embedding vị trí vào input trong SAI hiện tại.

SAI tạo tần số từ `head_dim` và `base`, dựng cache cosine/sine đến `max_seq_len`, rồi xoay Q và K. Value không được áp RoPE. `head_dim` phải chẵn. Trong code ngày 25/10/2026, `base` của lớp RoPE có mặc định `10000`, nhưng không có trường chọn RoPE base/scaling trong `SAIConfig` hay control center của phòng Kiến trúc.

RoPE có cấu trúc để attention phụ thuộc vị trí tương đối; điều đó không bảo đảm chất lượng ngoài độ dài đã train. Model hiện kiểm tra tổng độ dài không vượt `max_seq_len`. Tăng riêng một con số không phải bằng chứng model đã học context dài hơn. [Bài báo RoFormer](https://arxiv.org/abs/2104.09864).

## Những lựa chọn vị trí để hiểu thêm

| Cách tiếp cận | Ý tưởng | Trạng thái trong SAI |
| --- | --- | --- |
| Embedding vị trí học được | Một vector học được cho từng vị trí | Kiến thức tham khảo; chưa là lựa chọn registry |
| Sinusoidal cộng vào embedding | Hàm sin/cos cố định của vị trí | Kiến thức tham khảo từ Transformer gốc |
| RoPE | Xoay query/key theo vị trí | Đang dùng |

Mỗi cách làm thay đổi phép tính của model. Các lựa chọn tham khảo trong bảng chưa có implementation để chọn trong Lab. Nguồn đối chiếu cho cách vị trí của Transformer gốc: [Attention Is All You Need](https://arxiv.org/abs/1706.03762).

## Prefill khác decode thế nào?

```text
Prompt nhiều token → prefill → cache K,V cho từng tầng
                                     ↓
Token mới → tính Q,K,V mới → nối/ghi K,V → attend vào phần cache hợp lệ
                                     ↓
                              logits token kế tiếp
```

KV cache tránh tính lại K và V của mọi token cũ ở mỗi bước sinh. Nó không lưu toàn bộ kiến thức của model và không thay thế trọng số. Cache một tầng có shape `(B,Hkv,T,D)` cho mỗi tensor K và V. Tổng lượng phần tử KV tỷ lệ với `2 × số tầng × B × Hkv × T × D`; dtype quyết định số byte. Công thức này chỉ tính KV, chưa gồm trọng số, workspace và các tensor khác.

Trong API chính thức, `attention_mask` dùng `1` cho token thật, `0` cho padding; khi có mask, vị trí mặc định đếm token thật. Khi không truyền mask, `forward_features` giữ giả định cũ là không padding hoặc padding phải và loss bỏ PAD. Không suy ra rằng mọi API nhận padding trái mà không cần mask.

## Giới hạn cần biết của bản lắp ghép

Snapshot hiện tại: `SAI_modular` kế thừa cached API của `SAIModel`, nhưng block modular chưa nhận các keyword `attn_mask`/`past_kv` ở `prefill`, và chưa nhận `attn_mask` ở `forward_with_cache`. Vì vậy đường cached có mask hoặc truyền past cache có thể gọi sai chữ ký và báo `TypeError`. Không dùng mô tả “cùng phép tính mặc định” để suy ra mọi đường API mới đã tương đương.

Hạn chế này không nằm trên phép đo `compare-kernel`, vốn gọi trực tiếp kernel với tensor ngẫu nhiên. Trước khi thí nghiệm cached API của modular, cần kiểm chứng riêng đường gọi dự định dùng. Sổ tay chỉ ghi nhận hiện trạng, không sửa hoặc mở rộng API.
