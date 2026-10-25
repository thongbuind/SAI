---
schema_version: 1
department: arch
page_id: sai-current
title: "Kiến trúc hiện tại và các lựa chọn trong Lab"
description: "Đối chiếu các cỡ SAI chính thức với bản lắp ghép và phân biệt lựa chọn hiện có với ý tưởng thí nghiệm."
order: 50
kind: reference
updated: "2026-10-25"
tags: [SAI, cấu hình, registry, mặc định, tương thích]
sources:
  - SAI/modeling.py
  - SAI/attention.py
  - foundation_lab/architecture/models/__init__.py
  - foundation_lab/architecture/models/modular.py
  - foundation_lab/architecture/layers/norm.py
  - foundation_lab/architecture/layers/ffn.py
  - foundation_lab/architecture/layers/attention.py
  - foundation_lab/architecture/kernels/attention.py
  - foundation_lab/architecture/configs/SAI_modular_35M.yaml
  - config/models/SAI_35M.yaml
  - config/models/SAI_100M.yaml
  - config/models/SAI_375M.yaml
---

## Snapshot cấu hình ngày 25/10/2026

Các số sau được đọc từ ba file model chính thức, không phải số đo tốc độ hay số tham số đã tính chính xác. Tên 35M/100M/375M là tên cấu hình.

| Trường | SAI_35M | SAI_100M | SAI_375M |
| --- | ---: | ---: | ---: |
| `d_model` | 512 | 768 | 1024 |
| `num_layers` | 8 | 12 | 24 |
| `num_heads` | 8 | 12 | 16 |
| `num_kv_heads` | 8 | 6 | 4 |
| `head_dim = d_model / num_heads` | 64 | 64 | 64 |
| `ff_dim` | 2048 | 3072 | 4096 |
| `vocab_size` | 10000 | 10000 | 10000 |
| `max_seq_len` | 10240 | 10240 | 10240 |
| `dropout` | 0.05 | 0.05 | 0.05 |

Cả ba dùng decoder pre-norm, RoPE trên Q/K, RMSNorm, SwiGLU và weight tying. Lớp attention chung có tên GQA; cấu hình 35M có `Hkv = H` nên là trường hợp MHA. 100M và 375M có mức chia sẻ KV khác nhau.

**Khi xem một run, hãy dùng config đã resolve và bản ghi của chính run đó để biết mốc thực tế.** Nếu config thay đổi sau ngày này, bảng tĩnh không còn đại diện mặc định mới. Danh sách lựa chọn trong Lab phản ánh các implementation đã đăng ký ở phiên bản bạn đang dùng.

## Chính thức và lắp ghép

| Họ | Vai trò | Chọn thành phần qua config |
| --- | --- | --- |
| `SAI` | Model chính thức trong `SAI/` | Không có các trường `attention_kernel`, `ffn`, `norm` của bản modular |
| `SAI_modular` | Thí nghiệm kiến trúc trong Lab | Có các tên registry tương ứng |

Config `SAI_modular_35M` kế thừa `config/models/SAI_35M.yaml` rồi chọn họ modular và các thành phần. Ý định thiết kế là cấu hình modular mặc định giữ tên tham số và phép tính của model chính thức; không mở rộng kết luận này sang mọi API cached mới. Xem [giới hạn cached API](position-cache.md).

## Những tên đã có trong registry

| Registry | Tên trong snapshot | Mặc định modular |
| --- | --- | --- |
| `MODELS` | `SAI`, `SAI_modular` | Do config model chọn |
| `ATTENTIONS` | `gqa` | `gqa` |
| `ATTENTION_KERNELS` | `sdpa`, `eager` | `sdpa` |
| `FFNS` | `swiglu` | `swiglu` |
| `NORMS` | `rmsnorm` | `rmsnorm` |

Một registry có một tên vẫn hữu ích: nó xác định chỗ thêm biến thể sau này. LayerNorm, GELU và RoPE scaling được nhắc để hiểu kiến trúc; chúng chưa là những lựa chọn bạn có thể chọn trong Lab hiện tại.

## Đổi gì thì cần bằng chứng gì?

Đổi kernel không đổi trọng số, nhưng phải kiểm chứng output, gradient và đường gọi cần dùng. Đổi `Hkv`, `ff_dim`, số tầng hoặc thành phần thường làm đổi state_dict/phép tính và cần thí nghiệm train phù hợp. Đổi `max_seq_len` cần kiểm chứng cả giới hạn thực thi lẫn chất lượng context dài. Đổi tokenizer làm đổi nghĩa các ID và đòi hỏi xem lại toàn bộ dữ liệu, embedding và checkpoint.

Đừng dùng phép so kernel để phê duyệt thay kiến trúc. Hãy xác định trước giả thuyết, baseline lấy từ config hiện tại và ngân sách so sánh, rồi ghi lại điều kiện thí nghiệm.
