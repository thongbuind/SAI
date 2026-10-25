---
title: Quickstart
description: Cài lab, xem các phòng ban, chuẩn bị dữ liệu và chạy experiment đầu tiên.
group: Bắt đầu
order: 17
---

## Cài đặt

Lab cần Python 3.10 trở lên và bốn thư viện: PyTorch, NumPy, PyYAML, SentencePiece.

```bash
git clone https://github.com/thongbuind/SAI.git
cd SAI
pip install torch numpy pyyaml sentencepiece
```

Mọi lệnh bên dưới chạy từ thư mục gốc của repo.

## Xem các phòng ban

Lệnh `list` in ra mọi implementation đã đăng ký, nhóm theo phòng ban. Đây là các lựa chọn bạn có thể dùng trong config.

```bash
python -m foundation_lab list
```

Ví dụ phòng Kiến trúc có hai attention kernel (`eager`, `sdpa`) và hai họ mô hình (`SAI`, `SAI_modular`). Xem [Các registry](../extending/#registry) để biết mỗi phòng đăng ký những thành phần nào.

## Chuẩn bị dữ liệu

Dữ liệu không nằm trong repo: thư mục `data/` không được commit. Bạn tự chuẩn bị văn bản thô rồi encode bằng tokenizer chính thức (`SAI/tokenizer.model`).

- **Pretrain**: file JSONL, mỗi dòng là `{"text": "..."}`.
- **SFT**: file JSON là một mảng hội thoại `{"instruction": ..., "history": [{"user": ..., "model": ...}, ...]}`.

Văn bản được chuyển về chữ thường trước khi encode, khớp với cách tokenizer được huấn luyện.

```bash
python -m foundation_lab encode pretrain --input data/raw/pretrain_data_*.jsonl \
    --output-dir data/processed --manifest pretrain_manifest.json --model SAI_375M
python -m foundation_lab encode sft --input data/raw/SFT_1.json \
    --output data/processed/SFT1_data_ids.npz --model SAI_375M
```

`--model` chỉ dùng để lấy độ dài context tối đa (`max_seq_len`) khi cắt văn bản dài. Đường dẫn dữ liệu mà mỗi stage đọc nằm trong `foundation_lab/data/configs/*.yaml`; đặt file đúng chỗ đó hoặc sửa config.

## Kiểm tra rồi chạy

Luôn chạy `validate` trước: lệnh này resolve toàn bộ config và báo lỗi trước khi nạp dữ liệu hay dựng model.

```bash
python -m foundation_lab validate foundation_lab/core/configs/SAI_375M.yaml
python -m foundation_lab run foundation_lab/core/configs/SAI_375M.yaml
```

Lab tự chọn thiết bị theo thứ tự CUDA, Apple MPS rồi CPU (`runtime.device: auto`). Nếu `torch.compile` gặp lỗi trên máy của bạn, tắt nó khi chạy:

```bash
python -m foundation_lab run foundation_lab/core/configs/SAI_375M.yaml --set runtime.compile=false
```

> Để thử nhanh toàn bộ pipeline, dùng một tập dữ liệu nhỏ và giảm số epoch của các stage SFT, ví dụ `--set pipeline.sft1.params.epochs=1 --set pipeline.sft2.params.epochs=1`.

## Tiếp theo

- [Thư mục run](../runs/): xem run vừa chạy ghi lại những gì, và chạy tiếp khi bị ngắt.
- [Dòng lệnh](../cli/): các lệnh khác như `infer`, `compare-kernel`, `train-tokenizer`.
- [Mở rộng lab](../extending/): thêm kernel, layer, loss hay recipe mới.
