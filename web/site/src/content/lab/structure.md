---
title: Cấu tạo
description: Lab gồm 10 phòng ban, mỗi phòng sở hữu một phần việc. Trang này cho thấy các phòng là gì, phụ thuộc nhau ra sao và một experiment đi qua chúng thế nào.
group: Tổng quan
order: 3
---

## Các package trong repo

```text
SAI/              mô hình chính thức: model, generate, tokenizer, phiên bản
  ▲
foundation_lab/   lab huấn luyện và thí nghiệm, dùng SAI/
  ▲
hf_release/       đóng gói và phát hành lên Hugging Face, đọc thư mục run của lab

web/              web chat và tài liệu; khi chạy chỉ dùng SAI/ và phần nạp checkpoint
                  của hf_release/, Phòng làm việc gọi lab qua dòng lệnh thay vì import
```

Mũi tên chỉ chiều phụ thuộc: `SAI/` không import gì từ lab, lab không import gì từ `hf_release/` hay `web/`.

## Các phòng ban

Mỗi phòng là một package con trong `foundation_lab/`, sở hữu một phần việc và các registry của nó.

| Phòng | Package | Trách nhiệm | Registry |
| --- | --- | --- | --- |
| Ban điều hành | `core` | Registry, config nghiêm ngặt, experiment và pipeline spec | |
| Kiến trúc | `architecture` | Bản lắp ghép `SAI_modular`, attention kernel, layer, họ mô hình | `ATTENTION_KERNELS`, `ATTENTIONS`, `FFNS`, `NORMS`, `MODELS` |
| Dữ liệu | `data` | Nguồn dữ liệu, split, trộn, sampler, collate, `LMBatch` | `DATA_SOURCES` |
| Tokenizer | `tokenizer` | Train SentencePiece, encode văn bản thô thành `.npz` | `ENCODERS` |
| Objective | `objectives` | Các hàm loss thuần | `OBJECTIVES` |
| Vận hành | `engine` | Thiết bị, mixed precision, compile, microbatch, optimizer, vòng lặp, đo đạc | `LR_SCHEDULES` |
| Lưu trữ | `state` | Thư mục run, checkpoint, lineage, metrics | |
| Phương pháp huấn luyện | `recipes` | Pretrain, SFT (sau này distill, DPO, RLHF...) | `RECIPES` |
| Đánh giá | `eval` | Loss val/test, công cụ so sánh, thước đo văn bản sinh | |
| Suy luận | `inference` | Nạp checkpoint để sinh câu trả lời, so sánh các cách giải mã | `DECODERS` |

Một phòng chỉ gọi phòng khác qua hợp đồng đã ghi trong docstring, không với tay vào chi tiết bên trong. Vì vậy thay một kernel attention không cần đụng tới vòng lặp huấn luyện, và thêm một recipe mới không cần sửa phòng Dữ liệu.

## Một experiment đi qua các phòng thế nào

Khi chạy `python -m foundation_lab run foundation_lab/core/configs/SAI_375M.yaml`:

1. **Ban điều hành** đọc file, gộp `extends`, thay tên config bằng nội dung, áp `--set`, rồi kiểm tra toàn bộ.
2. **Lưu trữ** tạo thư mục run, ghi `resolved_config.yaml` và `lineage.json`.
3. Với mỗi stage:
   1. **Kiến trúc** dựng model từ registry `MODELS`; **Lưu trữ** nạp trọng số khởi tạo theo `init`.
   2. **Dữ liệu** đọc nguồn từ `DATA_SOURCES`, trộn và split, dựng DataLoader.
   3. **Phương pháp huấn luyện** chọn recipe từ `RECIPES`; recipe lấy hàm loss từ **Objective**, optimizer và lịch learning rate từ **Vận hành**.
   4. **Vận hành** chạy vòng lặp: microbatch, mixed precision, clip gradient, checkpoint định kỳ.
   5. **Đánh giá** tính loss val mỗi epoch và test loss cuối stage.
   6. **Lưu trữ** ghi `best.pt`, `last.ckpt.pt`, metrics, và `done.json` khi stage xong.

Lệnh `infer` đi theo đường ngắn hơn: **Suy luận** nạp checkpoint, chạy từng biến thể giải mã, **Đánh giá** tính thước đo văn bản sinh, **Lưu trữ** ghi kết quả vào `runs/infer/`.

## Ranh giới với model chính thức

`SAI_modular` trong phòng Kiến trúc là nơi thử biến thể. Với lựa chọn mặc định, nó cho ra đúng model trong `SAI/` (cùng phép tính, cùng `state_dict`), nên checkpoint train trong lab dùng thẳng được cho bản phát hành. Một biến thể chỉ được chuyển vào `SAI/` sau khi đã có thí nghiệm chứng minh.
