---
schema_version: 1
department: engine
page_id: batches-and-loss
title: Batch, microbatch và chuẩn hoá loss
description: Phân biệt số mẫu, token-slot, token chịu loss và lượng gradient trong một bước tối ưu.
order: 10
kind: concept
updated: "2026-10-25"
tags: [batch, microbatch, gradient, loss]
sources:
  - foundation_lab/engine/step.py
  - foundation_lab/engine/loop.py
  - foundation_lab/recipes/causal_lm.py
  - foundation_lab/recipes/sft.py
---

## Bốn đơn vị đếm

| Tên | Nghĩa trong lab | Dùng để làm gì? |
| --- | --- | --- |
| Logical batch | Batch mà DataLoader đưa cho engine | Đơn vị đọc dữ liệu và tích luỹ |
| Microbatch | Một lát theo chiều mẫu của logical batch | Giảm bộ nhớ cho forward/backward |
| Token-slot | Một ô trong tensor sau padding; số ô bằng `B × T` | Ước lượng tải và chia microbatch |
| Token chịu loss | Vị trí nhãn thật được objective tính trọng số | Tính loss và chuẩn hoá gradient |

Với SFT, nhiều token đầu vào là phần user/system nên vẫn tham gia attention nhưng không chịu loss. Với padding, ô PAD có trong tensor nhưng không phải nội dung cần học. Vì vậy `B × T` không phải số token hữu ích và cũng không phải số token chịu loss.

## Microbatch không phải một optimizer step

Mỗi microbatch forward, tính loss và backward riêng. Gradient của các microbatch cộng vào cùng tham số. Engine chỉ cập nhật optimizer khi đủ `accumulation_steps` logical batch hoặc đến cuối epoch.

Ví dụ minh hoạ: một logical batch có 64 mẫu, chiều dài sau padding 512 và budget 16.384 token-slot. Bộ chia hiện tại cho hai microbatch, mỗi microbatch tối đa 32 mẫu. Nếu tích luỹ 4 logical batch thì một lần cập nhật thường bao phủ khoảng 256 mẫu, nhưng sampler hoặc batch cuối có thể làm số thực tế khác đi.

Đây là effective batch theo số mẫu gần đúng: `batch_size × accumulation_steps` cho các batch đầy. Cấu hình lab hiện tại không có quy tắc nhân thêm số GPU qua distributed training. Về nguyên lý, tích luỹ cộng gradient của nhiều lượt trước khi update; AMP cần giữ cách scale nhất quán trong cửa sổ tích luỹ. [Tài liệu AMP của PyTorch](https://docs.pytorch.org/docs/stable/notes/amp_examples.html).

## Budget là mục tiêu chia, không phải bảo đảm VRAM

Engine ước tính số microbatch từ `B × T / max_microbatch_tokens`, sau đó cân bằng số mẫu bằng phép chia làm tròn. Vì phải giữ nguyên từng mẫu, budget không phải một giới hạn cứng tuyệt đối cho mọi shape. Một mẫu dài hơn budget vẫn phải chạy nguyên mẫu; chi phí attention và kích thước model còn ảnh hưởng bộ nhớ.

Giảm budget thường giảm activation đỉnh nhưng làm tăng số lượt forward/backward. Đổi budget không chủ động đổi số mẫu trong logical batch. Đổi `batch_size` hay `accumulation_steps` lại đổi cách gom gradient và số optimizer step.

## Engine chuẩn hoá gradient như thế nào?

Mỗi microbatch đóng góp:

```text
scaled_loss = loss_sum / target_tokens_per_logical_batch / accumulation_steps
```

`target_tokens_per_logical_batch` là mẫu số cố định do recipe cung cấp, **không phải** số token thật trong microbatch vừa chạy:

- `causal_lm`: độ dài train trung bình nhân `batch_size`.
- `sft`: `target_loss_tokens` của sampler, dựa trên token chịu loss.

Loss được báo trong log train lại là tổng loss đã quan sát chia tổng token chịu loss đã quan sát trong epoch. Nó là số để theo dõi, không phải toàn bộ biểu thức scalar được dùng ở mỗi lần backward. Khi resume giữa epoch, phần cộng dồn log trong bộ nhớ bắt đầu lại từ các batch còn lại; không coi một điểm log này là trung bình của cả epoch trước và sau ngắt.

Ở cửa sổ tích luỹ cuối epoch chưa đầy, engine vẫn chia cho `accumulation_steps` đã cấu hình. Không mặc định diễn giải nó thành trung bình riêng trên số batch còn lại.

## Loss chunk là lớp chia thứ hai

`loss_chunk_size` chia việc tính objective trên hidden state đã làm phẳng theo token. Mục tiêu là tránh giữ toàn bộ logits kích thước token × vocab cùng lúc; nó không rút ngắn chuỗi attention và không thay batch sampler. Hiệu quả thực tế còn phụ thuộc implementation objective đã chọn.

Đọc tiếp [profiling](profiling.md) để phân biệt áp lực bộ nhớ ở forward, loss và optimizer; đọc [optimizer](optimizer-and-schedule.md) để hiểu khi nào trọng số thực sự thay đổi.
