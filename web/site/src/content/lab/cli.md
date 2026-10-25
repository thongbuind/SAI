---
title: Dòng lệnh
description: Mọi việc trong lab đi qua một lệnh python -m foundation_lab với 8 lệnh con.
group: Bắt đầu
order: 14
---

## Danh sách lệnh

| Lệnh | Việc làm |
| --- | --- |
| `run` | Chạy pipeline của một experiment, hoặc chạy tiếp một run bằng `--resume` |
| `validate` | Kiểm tra và in config đã resolve, không chạy gì |
| `list` | Liệt kê implementation đã đăng ký, nhóm theo phòng ban |
| `encode` | Encode văn bản thô thành `.npz` bằng tokenizer chính thức |
| `train-tokenizer` | Train tokenizer SentencePiece mới vào một thư mục mới |
| `compare-kernel` | So một attention kernel với kernel mặc định của model: output, gradient, tốc độ, bộ nhớ |
| `infer` | So sánh các cách giải mã trên cùng một checkpoint và bộ câu hỏi |
| `describe` | In JSON mô tả lab (phòng ban, registry, schema lệnh) cho công cụ bên ngoài như Phòng làm việc |

Mọi lệnh chạy từ thư mục gốc của repo. Lỗi config thoát với mã 2 và in rõ khoá nào sai.

`compare-kernel` và `infer` là tác vụ: nhận spec là file YAML/JSON, `-` (đọc từ stdin) hoặc bỏ trống rồi chỉ dùng `--set`, và mỗi lần chạy ghi vào `runs/<lệnh>/<thời gian>/`. Đưa `resolved_config.yaml` của một run làm spec là chạy lại đúng như cũ.

## run và validate

```bash
python -m foundation_lab validate foundation_lab/core/configs/SAI_375M.yaml
python -m foundation_lab run foundation_lab/core/configs/SAI_375M.yaml
python -m foundation_lab run foundation_lab/core/configs/SAI_375M.yaml --stages sft1,sft2 --init sft1=model/pretrained_375M.pt
python -m foundation_lab run --resume runs/SAI_375M/20261021-120000
```

| Tuỳ chọn | Ý nghĩa |
| --- | --- |
| `--set khoa.con=gia_tri` | Ghi đè config, lặp lại được (cả `validate`) |
| `--stages a,b` | Chỉ chạy các stage này |
| `--init stage=file.pt` | Ghi đè trọng số khởi tạo của một stage, lặp lại được |
| `--resume <thư mục run>` | Chạy tiếp run; không đi kèm file experiment hay `--set` |
| `--runs-root`, `--run-id` | Thư mục chứa run (mặc định `runs/`) và tên thư mục run (mặc định theo thời gian) |

Xem [Experiments](../experiments/) và [Thư mục run](../runs/).

## encode

```bash
python -m foundation_lab encode pretrain --input data/raw/pretrain_data_*.jsonl \
    --output-dir data/processed --manifest pretrain_manifest.json --model SAI_375M
python -m foundation_lab encode sft --input data/raw/SFT_1.json \
    --output data/processed/SFT1_data_ids.npz --model SAI_375M
```

- Kiểu encode (`pretrain`, `sft`) là tên trong registry `ENCODERS`.
- `--output` ghi một file; `--output-dir` ghi mỗi file đầu vào thành một shard `<tên>_ids.npz`, kèm `--manifest` để ghép chúng thành một dataset.
- `--model` chỉ dùng để lấy `max_seq_len` khi cắt văn bản dài; `--tokenizer` đổi file tokenizer (mặc định `SAI/tokenizer.model`).

## train-tokenizer

```bash
python -m foundation_lab train-tokenizer --input data/raw/pretrain_data.jsonl --output-dir runs/tokenizers/new
```

Thư mục đích phải chưa tồn tại, nên lệnh này không thể ghi đè tokenizer chính thức. `--vocab-size` mặc định bằng tokenizer chính thức (10 000).

## compare-kernel

```bash
python -m foundation_lab compare-kernel --set candidate=eager
python -m foundation_lab compare-kernel --set candidate=eager --set "models=[SAI_modular_35M]" --set seq_len=512
```

Spec chỉ bắt buộc `candidate`, kernel cần so (tên trong registry `ATTENTION_KERNELS`). Kernel mốc của mỗi model luôn là `attention_kernel` trong config hiện tại của model đó, không gõ tay. `models` mặc định là cả ba cỡ (`SAI_modular_35M`, `SAI_modular_100M`, `SAI_modular_375M`); model không chọn kernel qua config (họ `SAI`) bị từ chối, dùng bản `SAI_modular_*` tương ứng. Muốn đo một shape khác thì thêm file config vào `foundation_lab/architecture/configs/`, hoặc ghi thẳng config trong spec dưới một nhãn tự đặt:

```yaml
candidate: eager
models:
  mqa:                 # nhãn trong kết quả
    family: SAI_modular
    vocab_size: 10000
    max_seq_len: 10240
    d_model: 512
    num_heads: 8
    num_kv_heads: 1
    num_layers: 8
    ff_dim: 2048
    dropout: 0.0
```

Mỗi model cho một bản ghi: sai khác lớn nhất của output và của gradient theo q, k, v; `matches_reference` (khớp theo dung sai mặc định của `torch.testing.assert_close` cho dtype); thời gian forward + backward của hai kernel và tỉ lệ `speedup`; bộ nhớ đỉnh (chỉ đo được trên CUDA). Các khoá `batch_size`, `seq_len`, `dtype`, `device`, `causal`, `repeats` đổi phép thử. Kết quả ghi vào `runs/compare-kernel/<thời gian>/`. Trên web, Phòng làm việc › Kiến trúc chạy đúng lệnh này. Xem [Mở rộng lab](../extending/#2-so-sánh-với-kernel-tham-chiếu).

## infer

```bash
python -m foundation_lab infer foundation_lab/inference/configs/compare_375M.yaml --set device=cpu
```

Spec của `infer` khai báo model, checkpoint, bộ câu hỏi và các biến thể giải mã:

```yaml
# foundation_lab/inference/configs/compare_375M.yaml (rút gọn)
extends: benchmark.yaml      # bộ câu hỏi mặc định
model: SAI_375M
checkpoint: model/sft1_375M.pt
device: auto
variants:
  - name: default            # bắt buộc: cấu hình sinh mặc định của bản phát hành
  - name: greedy
    generation: {strategy: greedy}
```

Mỗi biến thể gồm `name`, `decoder` (registry `DECODERS`, mặc định `sai` gọi `model.generate_response` chính thức) và `generation` (tham số sinh). Spec phải có ít nhất một biến thể dùng decoder `sai` với `generation` để mặc định, để mọi biến thể khác luôn được so với cách sinh người dùng bản phát hành đang nhận; thiếu thì lệnh báo lỗi config trước khi nạp model. Truyền `-` thay cho đường dẫn để đọc spec từ stdin. Kết quả ghi vào `runs/infer/<thời gian>/`, kèm thước đo cho từng biến thể: độ trễ trung bình, token/giây, số token sinh trung bình, tỉ lệ bị cắt, `distinct_2`.

Trên web, Phòng làm việc › Suy luận chạy đúng lệnh này bằng nút bấm.
