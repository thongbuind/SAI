---
schema_version: 1
department: arch
page_id: norm-ffn
title: "Residual, chuẩn hoá và FFN"
description: "Hiểu vai trò của RMSNorm và SwiGLU trong decoder block cùng các lựa chọn tham khảo."
order: 40
kind: concept
updated: "2026-10-25"
tags: [RMSNorm, LayerNorm, SwiGLU, FFN, residual]
sources:
  - SAI/block.py
  - SAI/ffn.py
  - foundation_lab/architecture/layers/norm.py
  - foundation_lab/architecture/layers/ffn.py
  - foundation_lab/architecture/layers/block.py
---

## Ba việc khác nhau trong một block

Attention kết hợp thông tin giữa vị trí; FFN biến đổi đặc trưng tại từng vị trí; normalization điều chỉnh thang độ của vector trước nhánh tính toán. Residual đưa đầu vào cộng trở lại đầu ra nhánh. Cả bốn cùng làm việc, không có một thành phần đơn lẻ đại diện cho toàn bộ khả năng học của Transformer.

Trong SAI, norm đi trước attention và trước FFN. Nếu nhánh attention/FFN trả toàn số không thì residual giữ nguyên vector đầu vào; đây là cách dễ hình dung đường truyền trực tiếp qua block. Việc chuyển norm ra sau phép cộng làm đổi hàm mô hình, không phải chỉnh trình bày code.

## RMSNorm và LayerNorm

RMSNorm chia vector cho căn trung bình bình phương của chính vector, có epsilon để ổn định số học, rồi dùng hệ số học được. Nó không thực hiện bước trừ trung bình như LayerNorm. Mỗi vị trí được chuẩn hoá trên chiều biểu diễn; không trộn các mẫu trong batch.

| Lựa chọn | Điểm chính | Trạng thái |
| --- | --- | --- |
| RMSNorm | Không trừ trung bình; cấu trúc phép tính gọn hơn | `rmsnorm` là norm duy nhất đã đăng ký |
| LayerNorm | Trừ trung bình và chuẩn hoá phương sai | Khái niệm tham khảo; chưa đăng ký trong Lab |
| Pre-norm / post-norm | Vị trí đặt norm so với nhánh residual | SAI đang pre-norm; chưa có trường chọn trên web |

SAI dùng `nn.RMSNorm(..., eps=1e-6)` cho hai norm trong block và norm cuối model. Đừng chép epsilon từ thư viện khác rồi xem là cùng implementation. Kết quả tốc độ trong một nghiên cứu không bảo đảm trên phần cứng của bạn. [Bài báo RMSNorm](https://arxiv.org/abs/1910.07467).

## FFN và SwiGLU

FFN là mạng truyền thẳng áp riêng cho từng token, với các trọng số dùng chung qua vị trí. Trong SAI:

```text
x (B,T,d_model)
  → gate_up → hai nhánh gate và up, mỗi nhánh (B,T,ff_dim)
  → up × SiLU(gate)
  → down → (B,T,d_model)
```

Phép nhân theo phần tử tạo một cổng mềm phụ thuộc input. `ff_dim` là chiều trung gian của mỗi nhánh, không phải số tầng. `gate_up` gộp hai phép chiếu, còn `down` đưa kết quả về đúng chiều để cộng residual. Các Linear của SwiGLU SAI không có bias.

| Lựa chọn tham khảo | Đánh đổi cần kiểm tra |
| --- | --- |
| FFN ReLU hoặc GELU không cổng | Hai phép chiếu; cùng `ff_dim` không có cùng số tham số với FFN có cổng |
| SwiGLU | Có nhánh cổng; hiện đã đăng ký `swiglu` |
| FFN có cổng dùng activation khác | Có thể đổi tính chất tối ưu; chưa phải option của Lab |

Với SAI hiện tại, phần trọng số FFN có khoảng `3 × d_model × ff_dim` phần tử. Khi so với FFN hai phép chiếu, phải quyết định giữ cùng tham số, FLOPs hay chiều trung gian; đó là ba thiết kế thí nghiệm khác nhau. Không kết luận activation tốt hơn nếu một bên được cấp ngân sách lớn hơn.

[Nghiên cứu các biến thể GLU](https://arxiv.org/abs/2002.05202) là nguồn nền cho FFN có cổng. Chỉ những biến thể đã đăng ký mới dùng được trong config; snapshot ngày 25/10/2026 có `swiglu` và `rmsnorm`.
