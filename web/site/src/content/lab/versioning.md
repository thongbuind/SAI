---
title: Phiên bản
description: SAI có hai dãy phiên bản độc lập. devX.Y.Z là phiên bản phát hành của mô hình; labx.y.z là phiên bản nội bộ của lab, ghi vào mỗi run.
group: Bắt đầu
order: 16
---

## Hai dãy phiên bản

| Dãy | Ở đâu | Dùng cho |
| --- | --- | --- |
| `devX.Y.Z` | `SAI/version.py` | Bản phát hành: trọng số, package `SAI` và adapter Hugging Face. Gắn tag trên Hub. |
| `labx.y.z` | `foundation_lab.__version__` | Lịch sử của lab trên git. Ghi vào `lineage.json` của mỗi run, không phát hành. |

Hai dãy không so sánh với nhau: đổi bên này không kéo theo bên kia. Mốc chung duy nhất là lab1.0.0 ứng với dev3.0.0.

## Phiên bản phát hành devX.Y.Z

Mỗi file thành phần trong `SAI/` (`rope.py`, `attention.py`, `ffn.py`, `block.py`, `modeling.py`, `generate.py`) có hằng `VERSION`. Tokenizer không chứa được hằng số nên phiên bản của nó ghi trong `SAI/version.py`. Phiên bản phát hành `MODEL_VERSION` cũng ghi ở đó.

Phiên bản phát hành chỉ tăng khi người dùng bản phát hành nhận được hành vi khác:

| Phần | Khi nào tăng |
| --- | --- |
| X | Giai đoạn phát triển (hiện là dev3) |
| Y | Phải train lại: đổi tham số hay `state_dict`, đổi phép tính làm output forward khác, đổi tokenizer. Khi Y tăng thì Z về 0 |
| Z | Không cần train lại nhưng hành vi đổi: output của generate khác với cùng input và config, đổi giao diện dùng (tham số, field config, cách nạp từ Hub), kernel nhanh hơn hoặc ít bộ nhớ hơn cho cùng output |

Không tăng phiên bản khi chỉ refactor, sửa comment, docstring, test, lab, web, README hay model card.

### Phiên bản của thành phần

`MODEL_VERSION` không phải là phiên bản lớn nhất của các thành phần mà được đặt riêng. Khi phát hành, thành phần được sửa nhận đúng phiên bản phát hành mới; thành phần không đổi giữ phiên bản cũ. Có những lần phát hành không sửa thành phần nào, ví dụ chỉ đổi cách nạp model từ Hub.

Hai ràng buộc được kiểm tra trước khi export: mọi thành phần cùng giai đoạn X với bản phát hành, và không thành phần nào có phiên bản lớn hơn `MODEL_VERSION`.

Ví dụ, bắt đầu từ dev3.0.0:

1. Viết kernel attention nhanh hơn, cùng output: attention thành **dev3.0.1**, phát hành dev3.0.1.
2. Đổi FFN, phải train lại: ffn thành **dev3.1.0**, phát hành dev3.1.0.
3. Chỉ đổi cách nạp model từ Hub: phát hành **dev3.1.1**, mọi thành phần giữ nguyên (ffn vẫn dev3.1.0).

Mỗi lần phát hành được gắn tag phiên bản trên Hugging Face; tag cũ được giữ nguyên, nên luôn tải lại được bản cũ.

## Phiên bản lab

Lab có phiên bản nội bộ `labx.y.z`, quản lý bằng `foundation_lab.__version__`:

| Phần | Khi nào tăng |
| --- | --- |
| x | Tổ chức lại lab quy mô lớn: thêm, bỏ hay gộp phòng, đổi hợp đồng giữa các phòng, đổi cách khai báo thí nghiệm hoặc cấu trúc CLI |
| y | Thí nghiệm cũ không còn tái lập được: spec cũ chạy lỗi hoặc cho kết quả khác, đổi định dạng thư mục run, checkpoint hay dữ liệu encode, đổi cách tính loss hay metric |
| z | Thêm khả năng mới mà thí nghiệm cũ vẫn chạy y như trước: biến thể mới trong registry, lệnh CLI mới, sửa lỗi không đổi kết quả |

Mỗi run ghi `lab_version` vào `lineage.json`, nên biết ngay một kết quả cũ còn so sánh được với run mới hay không:

- Cùng x và y: run cũ vẫn resume, làm `init` và so sánh trực tiếp với run mới.
- Khác y: run cũ chỉ để tham khảo; muốn so sánh thì chạy lại baseline trên phiên bản mới.
- Khác x: mọi kết quả trước đó chỉ để tham khảo.

Muốn chạy lại y như cũ, checkout commit git ghi trong `lineage.json` của run đó.

## Đọc phiên bản trong code

```python
import foundation_lab
from SAI import MODEL_VERSION, COMPONENT_VERSIONS

print(MODEL_VERSION)              # phiên bản phát hành
print(COMPONENT_VERSIONS)         # {"rope": ..., "attention": ..., ..., "tokenizer": ...}
print(foundation_lab.__version__) # phiên bản lab
```

Chạy đoạn code trên từ thư mục gốc của repo.
