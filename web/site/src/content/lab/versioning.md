---
title: Versioning
description: Mỗi thành phần của SAI có phiên bản riêng dạng devX.Y.Z; phiên bản mô hình là phiên bản lớn nhất trong các thành phần.
group: Kiến trúc lab
order: 4
---

## Quy ước devX.Y.Z

Mỗi file thành phần trong `SAI/` (`rope.py`, `attention.py`, `ffn.py`, `block.py`, `modeling.py`, `generate.py`) có hằng `VERSION`. Tokenizer không chứa được hằng số nên phiên bản của nó ghi trong `SAI/version.py`.

| Phần | Khi nào tăng |
| --- | --- |
| X | Giai đoạn phát triển (hiện là dev3) |
| Y | Thay đổi buộc phải train lại mô hình: đổi tham số hay `state_dict`, hoặc đổi phép tính làm output khác. Khi Y tăng thì Z về 0 |
| Z | Thay đổi không cần train lại: ví dụ kernel nhanh hơn cho cùng output, hay sửa `generate` |

Đổi tokenizer luôn phải train lại mô hình, nên luôn tăng Y.

## Phiên bản của thành phần vừa sửa

Phiên bản mới của thành phần được tính từ **phiên bản mô hình hiện tại** cộng bậc tương ứng, không tính từ phiên bản cũ của riêng thành phần đó. Nhờ vậy thành phần vừa sửa luôn có phiên bản lớn nhất, và phiên bản mô hình bằng phiên bản của nó.

Ví dụ, bắt đầu từ dev3.0.0:

1. Sửa kernel attention, cùng output: attention thành **dev3.0.1**, mô hình dev3.0.1.
2. Đổi FFN, phải train lại: ffn thành **dev3.1.0**, mô hình dev3.1.0.
3. Sửa generate: generate thành **dev3.1.1**, mô hình dev3.1.1.

## Đọc phiên bản trong code

```python
from SAI import MODEL_VERSION, COMPONENT_VERSIONS

print(MODEL_VERSION)        # phiên bản mô hình
print(COMPONENT_VERSIONS)   # {"rope": ..., "attention": ..., ..., "tokenizer": ...}
```

Chạy đoạn code trên từ thư mục gốc của repo để xem phiên bản hiện tại của mô hình và từng thành phần.
