---
schema_version: 1
department: arch
page_id: attention
title: "Attention, MHA, GQA và kernel"
description: "Phân biệt cách chia head với cách thực hiện phép tính attention và hiểu các đánh đổi."
order: 20
kind: concept
updated: "2026-10-25"
tags: [attention, GQA, MHA, MQA, SDPA, causal]
sources:
  - SAI/attention.py
  - foundation_lab/architecture/layers/attention.py
  - foundation_lab/architecture/kernels/attention.py
  - foundation_lab/architecture/registry.py
---

## Query, key và value

Với mỗi token, attention layer chiếu hidden state thành query `Q`, key `K` và value `V`. Có thể hiểu query là thông tin vị trí này muốn tìm, key là thông tin dùng để so mức liên quan, value là nội dung được tổng hợp. Đây là cách hình dung; chúng đều là vector học được, không có nhãn ý nghĩa cố định.

Một head thực hiện `softmax(QKᵀ / sqrt(D) + mask) V`. `D` là chiều mỗi head; chia cho căn `D` kiểm soát độ lớn score. Softmax tạo trọng số cộng lại bằng một trên các key hợp lệ. Attention output là tổng có trọng số của các value, sau đó các query head được ghép và chiếu về `d_model`.

Trong SAI, `D = d_model / H`, với `H = num_heads`. QKV dùng một phép chiếu gộp không bias. `Hkv = num_kv_heads` có thể nhỏ hơn `H`.

## Ba cách phân bổ head

| Cấu hình | Quan hệ | Lợi ích và đánh đổi |
| --- | --- | --- |
| MHA | `Hkv = H` | Mỗi query head có KV riêng; nhiều tham số KV và cache hơn |
| GQA giữa hai đầu mút | `1 < Hkv < H`, `H` chia hết `Hkv` | Nhiều query head dùng chung một KV head; giảm chi phí KV, cần đánh giá chất lượng sau train |
| MQA | `Hkv = 1` | Cache KV nhỏ nhất trong họ này; chia sẻ mạnh hơn có thể ảnh hưởng năng lực biểu diễn |

Cả ba trường hợp được biểu diễn bằng lớp `gqa` hiện có thông qua số head; không có ba tên registry riêng `mha`, `gqa`, `mqa`. Đổi `Hkv` làm đổi kích thước phép chiếu QKV. Không thể coi đó là đổi kernel không tham số hay dùng nguyên checkpoint mà không xử lý tương thích.

GQA được đề xuất như một đánh đổi giữa MHA và MQA; kết quả trong bài báo không bảo đảm thứ hạng trên dữ liệu SAI. [Nghiên cứu GQA](https://arxiv.org/abs/2305.13245).

## Causal mask và padding mask

Causal mask cấm nhìn token tương lai; nếu không có nó trong train next-token thông thường, model có thể nhìn đáp án. Padding mask cấm sử dụng các vị trí đệm làm key. Hai mask có mục đích khác nhau.

Hợp đồng kernel của Lab dùng mask boolean với `True = được nhìn`; mask số thực được cộng vào score. Kernel không nhận đồng thời `attn_mask` và `is_causal=True`. Causal mask của kernel căn góc trên-trái. Trong phép đo web hiện có, query và key dài bằng nhau và chỉ chọn causal bật/tắt; chưa có form khảo sát mask tùy ý hay cache dài lệch nhau.

## Kernel hiện có

| Registry | Cách làm | Khi hữu ích |
| --- | --- | --- |
| `sdpa` | Gọi SDPA của PyTorch với GQA; danh sách backend Flash, efficient, math | Đường mặc định; backend thực tế phụ thuộc thiết bị, dtype và shape |
| `eager` | Tạo score, mask, softmax và nhân V bằng phép toán PyTorch; lặp KV head khi cần | Đọc thuật toán và đối chiếu; score lớn có thể tốn bộ nhớ |

Tên `sdpa` không cam kết luôn chạy Flash Attention. Thay kernel có thể đổi thứ tự tính số thực nên sai khác nhỏ cần được xét theo dung sai. Xem [hợp đồng SDPA của PyTorch](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) và [cách đọc kết quả](results.md).

Các lựa chọn trên là snapshot ngày 25/10/2026; xem danh sách trong Lab để biết những kernel đã đăng ký ở phiên bản bạn đang dùng.
