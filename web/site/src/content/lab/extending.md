---
title: Mở rộng lab
description: Thêm biến thể bằng cách đăng ký implementation vào registry của phòng tương ứng rồi chọn nó qua config, không sửa code của phòng khác.
group: Bắt đầu
order: 15
---

## Registry

Mỗi phòng ban sở hữu một hoặc vài registry: bảng tra từ tên sang implementation. Implementation mới chỉ cần `@REGISTRY.register("ten")`, rồi chọn bằng tên đó trong config. Tên trùng hoặc tên không tồn tại đều báo lỗi ngay, không âm thầm thay thế.

| Registry | Phòng | Đăng ký cái gì |
| --- | --- | --- |
| `ATTENTION_KERNELS` | Kiến trúc | Phép tính attention không tham số |
| `ATTENTIONS`, `FFNS`, `NORMS` | Kiến trúc | Module attention, feed-forward, normalization |
| `MODELS` | Kiến trúc | Họ mô hình (`ModelFamily`) |
| `DATA_SOURCES` | Dữ liệu | Nguồn dữ liệu đã tokenize |
| `ENCODERS` | Tokenizer | Cách encode văn bản thô |
| `OBJECTIVES` | Objective | Hàm loss |
| `LR_SCHEDULES` | Vận hành | Lịch learning rate |
| `RECIPES` | Phương pháp huấn luyện | Phương pháp huấn luyện một stage |
| `DECODERS` | Suy luận | Cách giải mã cho lệnh `infer` (`sai` gọi `model.generate_response` chính thức) |

## Bốn cấp độ thay đổi kiến trúc

| Cấp độ | Ví dụ | Cách làm |
| --- | --- | --- |
| Giá trị cấu hình | `num_layers`, `num_kv_heads` | Thêm file vào `foundation_lab/architecture/configs/` |
| Phép tính | Attention kernel tự viết | Đăng ký vào `ATTENTION_KERNELS` (không đổi tham số) |
| Thành phần | Attention, FFN, norm khác | Đăng ký vào `ATTENTIONS`, `FFNS`, `NORMS` |
| Họ mô hình | Encoder–decoder | `ModelFamily` mới đăng ký vào `MODELS` |

Hai cấp "phép tính" và "thành phần" dùng họ `SAI_modular`: bản lắp ghép của SAI mà mỗi mảnh chọn được qua config. Với lựa chọn mặc định, `SAI_modular` cho ra đúng model chính thức (cùng phép tính, cùng `state_dict`). Model chính thức nằm trong `SAI/`; biến thể chỉ được đưa vào đó sau khi đã có thí nghiệm.

## Ví dụ: thêm một attention kernel

### 1. Viết kernel theo hợp đồng chung

Kernel nhận `q` có shape `(B, H, Tq, D)`, `k` và `v` có shape `(B, Hkv, Tk, D)` với `H` chia hết cho `Hkv` (Grouped Query Attention), và trả về `(B, H, Tq, D)`. `is_causal=True` dùng causal mask căn góc trên-trái như SDPA; `attn_mask` là mask bool (True là được nhìn) hoặc float (cộng vào score), không dùng cùng `is_causal`.

Thêm hàm vào `foundation_lab/architecture/kernels/attention.py`, nơi hai kernel `sdpa` và `eager` đã được đăng ký:

```python
@ATTENTION_KERNELS.register("my_kernel")
def my_attention(q, k, v, *, attn_mask=None, is_causal=False, dropout_p=0.0):
    """Mô tả ngắn: kernel này khác sdpa ở điểm nào."""
    ...
```

Nếu đặt kernel trong một file mới, import file đó trong `foundation_lab/architecture/kernels/__init__.py` để việc đăng ký chạy khi lab được nạp.

### 2. So sánh với kernel tham chiếu

Kernel không có tham số, nên nếu output và gradient khớp kernel mặc định của model thì có thể thay kernel mà vẫn dùng nguyên trọng số đã train.

```bash
python -m foundation_lab compare-kernel --set candidate=my_kernel
```

Kernel tham chiếu luôn là `attention_kernel` mặc định hiện tại của từng model (hiện là `sdpa`), lấy từ config của model chứ không gõ tay. Lệnh chạy trên shape attention của cả ba cỡ model (`SAI_modular_35M`, `SAI_modular_100M`, `SAI_modular_375M`); mỗi cỡ cho sai khác lớn nhất của output (`max_abs_diff_out`) và của gradient theo q, k, v, kết luận `matches_reference`, thời gian forward cộng backward (`reference_ms`, `candidate_ms`, `speedup`) và bộ nhớ đỉnh (`reference_peak_mib`, `candidate_peak_mib`, chỉ đo được trên CUDA) của hai kernel. Kết quả ghi vào `runs/compare-kernel/<thời gian>/`; trên web, mở Phòng làm việc › Kiến trúc để chạy và xem dạng bảng.

### 3. Chọn kernel trong config

Sao chép `foundation_lab/architecture/configs/SAI_modular_375M.yaml` thành file mới và đổi một dòng:

```yaml
extends: ../../../config/models/SAI_375M.yaml
family: SAI_modular
attention: gqa
attention_kernel: my_kernel
ffn: swiglu
norm: rmsnorm
```

Rồi trỏ `model` của experiment tới file đó và chạy như bình thường.

## Các registry khác

Cách làm giống hệt: viết implementation theo hợp đồng ghi trong docstring của registry, đăng ký bằng tên, chọn qua config. Chạy `python -m foundation_lab list` để kiểm tra implementation mới đã xuất hiện. Mỗi registry nằm trong file `registry.py` hoặc module chính của phòng tương ứng trong `foundation_lab/`.
