---
schema_version: 1
department: recipes
page_id: optimization
title: Learning rate, optimizer và chuẩn hóa loss
description: Hiểu các tham số huấn luyện quan trọng và phân biệt loss để backward với loss báo cáo.
order: 30
kind: concept
updated: "2026-10-25"
tags: [loss, learning rate, AdamW, accumulation, microbatch, warmup]
sources:
  - foundation_lab/recipes/base.py
  - foundation_lab/recipes/causal_lm.py
  - foundation_lab/recipes/sft.py
  - foundation_lab/engine/optim.py
  - foundation_lab/engine/step.py
  - foundation_lab/engine/loop.py
  - foundation_lab/objectives/lm_ce.py
  - foundation_lab/eval/lm_loss.py
---

## Các núm điều khiển trả lời câu hỏi khác nhau

| Tham số | Ý nghĩa | Đánh đổi cần theo dõi |
| --- | --- | --- |
| `epochs` | Số lượt qua train loader | Nhiều lần tiếp xúc dữ liệu hơn, tốn thêm thời gian và có thể overfit |
| `learning_rate` | Cỡ bước cập nhật danh nghĩa | Quá lớn có thể bất ổn; quá nhỏ có thể học chậm |
| `weight_decay` | Mức decay trong AdamW | Tác động lên trọng số, không thay thế chất lượng dữ liệu |
| `batch_size` | Số mẫu pretrain hoặc mốc mục tiêu token SFT | Đổi lượng gradient mỗi batch và số step |
| `accumulation_steps` | Số batch logic tích lũy trước update | Tăng lượng dữ liệu/update, đổi lịch step |
| `max_microbatch_tokens` | Ngân sách dự kiến theo ô token đã pad | Chi phối chia microbatch; không phải giới hạn byte RAM cứng |
| `max_grad_norm` | Ngưỡng clip norm gradient | Chặn update quá lớn; không chữa nguyên nhân dữ liệu/config sai |
| `loss_chunk_size` | Số hàng hidden được chiếu logits một lần | Đánh đổi bộ nhớ và overhead của phép tính loss |

Không tăng nhiều tham số cùng lúc nếu mục tiêu là xác định nguyên nhân cải thiện. Bắt đầu từ baseline đã resolve và nêu giả thuyết rõ.

## AdamW hiện được dùng thế nào?

AdamW tách weight decay khỏi bước cập nhật theo gradient của loss; cơ sở thuật toán nằm trong [Loshchilov và Hutter](https://arxiv.org/abs/1711.05101). `build_adamw` hiện áp decay lên mọi tham số được truyền vào, kể cả norm và embedding; code không chia nhóm no-decay riêng.

`adamw_fused` điều khiển implementation optimizer. Mặc định của từng recipe/config có thể khác nhau; đó không phải một tùy chọn làm thay đổi mục tiêu học. Không đánh giá chất lượng recipe chỉ từ việc fused chạy nhanh hơn.

## Lịch learning rate hiện tại

Registry có lịch `warmup_stable_decay`: warmup tuyến tính, giữ hệ số 1 tới 40% tổng optimizer steps, giảm tuyến tính tới 0,1 ở 70%, rồi giữ 0,1. `warmup_ratio` quyết định số warmup steps. Đây là snapshot implementation, không phải tên chung đảm bảo mọi thư viện dùng cùng đường cong.

Scheduler bước theo **optimizer step**, không theo microbatch. Tổng step dự kiến được recipe tính từ số batch train, accumulation và epoch. Đọc log `warmup`/`total` của run để biết các con số thực tế.

## Tổng loss, loss backward và loss báo cáo

Objective `lm_ce` trả tổng cross-entropy nhân `loss_weight`, chưa chia. Engine dùng:

```text
loss_backward = loss_sum / mẫu_số_mục_tiêu / accumulation_steps
```

Mẫu số mục tiêu của causal LM là độ dài train trung bình nhân batch_size; của SFT là `target_loss_tokens` sampler. Nó **không phải** số token chịu loss thực tế của từng microbatch. Chia microbatch không tạo một phép lấy trung bình riêng cho từng mảnh.

Loss train ghi log lấy tổng objective đã detach chia tổng token chịu loss đã đi qua. Val/test luôn dùng CE `chunked_lm_loss` trung bình theo token chịu loss trên loader. Nếu sau này chọn objective train khác CE, phải đọc nhãn metric cẩn thận; không mặc định train và val cùng đại lượng.

## Giới hạn giảm bộ nhớ

Microbatching giảm số mẫu xử lý đồng thời; chunked loss giảm kích thước logits tạo cùng lúc. Cả hai vẫn phải xử lý toàn bộ lượng dữ liệu được yêu cầu và vẫn cần memory cho trọng số/optimizer. Chúng không cho phép suy rằng cấu hình lớn phù hợp máy 8GB chỉ từ một ngưỡng token.

Xem [Vận hành](../../engine/docs/index.md) và [Objective](../../objectives/docs/index.md) để tra sâu.
