---
title: Chức năng
description: Những gì Foundation Lab làm được, từ chuẩn bị dữ liệu, huấn luyện nhiều stage, thí nghiệm kiến trúc, so sánh cách giải mã tới phát hành lên Hugging Face.
group: Tổng quan
order: 2
---

## Huấn luyện nhiều stage

- Một experiment là một chuỗi stage chạy theo thứ tự: pipeline chuẩn của SAI là pretrain → SFT1 → SFT2, có sẵn stage continued pretrain (mặc định tắt).
- Hai recipe: `causal_lm` cho pretrain/continued pretrain, `sft` cho hội thoại có loss mask (chỉ câu trả lời của model chịu loss).
- Stage sau khởi tạo từ checkpoint tốt nhất của stage trước (`init: sft1.best`) hoặc từ một file `.pt` bất kỳ; đóng băng được từng decoder block (`freeze_blocks`).
- Chạy lại một phần pipeline: `--stages sft1,sft2 --init sft1=model/pretrained_375M.pt`.

## Chạy nhanh trên phần cứng nhỏ

- Tự chọn thiết bị CUDA → Apple MPS → CPU.
- Trên CUDA: autocast BF16 (FP16 + GradScaler nếu GPU không hỗ trợ BF16), AdamW fused.
- `torch.compile` cho phần forward.
- Batch tự chia thành microbatch theo `max_microbatch_tokens`; loss tính theo từng khúc (`loss_chunk_size`) để không phải giữ cả ma trận logits.
- Sampler sortish gom mẫu cùng độ dài để bớt padding; với SFT, batch cân bằng theo số token chịu loss.

## Config nghiêm ngặt

- Khoá lạ, thiếu khoá bắt buộc hay sai kiểu đều báo lỗi ngay, kèm danh sách khoá hợp lệ.
- `validate` kiểm tra cả tương thích giữa các phần: recipe có nhận loại dữ liệu đó không, họ mô hình có đủ capability mà recipe cần không, `init` có trỏ tới stage đứng trước không.
- Mọi việc này xong trước khi nạp dữ liệu hay dựng model.

## Truy vết và tái lập

- Config đã resolve được ghi thành `resolved_config.yaml` kèm `config_hash`: chạy lại một run chỉ cần file này.
- `lineage.json` ghi commit git (kèm cờ working tree có thay đổi chưa commit), môi trường (Python, PyTorch, CUDA, GPU), dấu vân tay tokenizer và từng file dữ liệu, phiên bản lab.
- `invocations.jsonl` ghi mỗi lần gọi trên run: lệnh, stage, init override, commit git.

## Resume chính xác

- Checkpoint định kỳ theo số bước (`checkpoint_every_steps`), kể cả giữa epoch.
- Checkpoint lưu cả trạng thái RNG, sampler và GradScaler, nên chạy tiếp cho kết quả như chưa từng bị ngắt; test kiểm tra điều này bằng cách cố ý dừng run giữa chừng.
- Stage đã xong được đánh dấu và bỏ qua khi resume.

## Thí nghiệm kiến trúc

- Họ `SAI_modular` là bản lắp ghép của SAI: attention, attention kernel, FFN và norm đều chọn được qua config. Với lựa chọn mặc định, nó cho ra đúng model chính thức (cùng phép tính, cùng `state_dict`).
- `compare-kernel` so một attention kernel với kernel mặc định của model về sai khác output, gradient theo q/k/v, thời gian và bộ nhớ đỉnh của forward + backward, trên shape attention của cả ba cỡ model.

## Dữ liệu và tokenizer

- `train-tokenizer` train tokenizer SentencePiece mới vào thư mục riêng, không đụng tokenizer chính thức.
- `encode pretrain`: văn bản dài hơn context được chia thành nhiều mẫu chồng lấn, cắt ở ranh giới tự nhiên; nhiều file đầu vào ghép thành các shard kèm manifest.
- `encode sft`: hội thoại được dựng theo chat template của SAI, kèm loss mask.
- Nguồn dữ liệu trộn được với nguồn phụ theo tỉ lệ (`mix_ratio`), ví dụ SFT2 trộn thêm dữ liệu SFT1.

## Suy luận và so sánh cách giải mã

- `infer` chạy một bộ câu hỏi trên nhiều biến thể, mỗi biến thể là một cách giải mã (`decoder`) cùng một bộ tham số sinh.
- Mỗi biến thể có thước đo: độ trễ trung bình, token/giây, số token sinh trung bình, tỉ lệ bị cắt vì hết độ dài, `distinct_2` (độ đa dạng, càng gần 1 càng ít lặp).

## Đo đạc

- Đo đỉnh VRAM theo từng pha của bước train, ước tính FLOPs, đo TFLOPS và log kernel; bật bằng `runtime.profile_vram`, `runtime.profile_kernels`.

## Phát hành lên Hugging Face

- `hf_release` dựng repo Hugging Face từ một stage của thư mục run: trọng số, config, tokenizer, model card và adapter để tải bằng Transformers.
- Bản export được tải lại như người dùng và so với `SAI.modeling` trước khi push; push chính thức yêu cầu export từ commit sạch và gắn tag phiên bản.

## Phòng làm việc trên web

- Chạy `python -m web --lab` để có thêm Phòng làm việc ở `/lab/workbench/`: nút bấm trên web gọi lệnh của lab, theo dõi tiến trình trực tiếp, huỷ được giữa chừng và đọc lại run cũ.
- Chỉ mở được từ máy của bạn (localhost), vì nút bấm chạy code thật trên máy đó.
- Hiện có hai phòng: Kiến trúc (so một attention kernel với kernel mặc định của từng model) và Suy luận (so sánh các cách giải mã trên cùng một checkpoint và cùng bộ câu hỏi).
- Mọi so sánh có cấu hình mặc định hiện tại làm mốc, mang nhãn "mốc" trên trang; mỗi run có dòng lệnh để chạy lại đúng như cũ từ dòng lệnh.

## Kiểm thử

- Bộ test chạy hoàn toàn trên CPU với model và dữ liệu tí hon tự tạo, xong trong chưa tới một phút trên laptop.
- Test đối chiếu `legacy` lấy code huấn luyện cũ từ lịch sử git, chạy cả hai pipeline và so từng trọng số.
