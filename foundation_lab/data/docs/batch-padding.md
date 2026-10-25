---
schema_version: 1
department: data
page_id: batch-padding
title: Gom batch, sampler và padding
description: Phân biệt batch theo số mẫu, batch theo token chịu loss và chi phí padding.
order: 30
kind: concept
updated: "2026-10-25"
tags: [batch, sampler, padding, sortish, bộ nhớ]
sources:
  - foundation_lab/data/loaders.py
  - foundation_lab/data/samplers.py
  - foundation_lab/data/batch.py
  - tests/test_data.py
---

## Vì sao các mẫu phải có cùng độ dài trong batch?

Mỗi mẫu có thể dài khác nhau, nhưng phép tính tensor cần hình chữ nhật `(B, T)`. Collator thêm các vị trí đệm bên phải. Những vị trí này vẫn chiếm ô tensor và tài nguyên tính toán, dù trọng số loss bằng 0.

Ví dụ có mẫu dài 90 và 110 token. Với bucket hiện tại là bội số 128, batch có 256 ô token, trong đó 200 ô là token thật. Hiệu suất padding là `200 / 256 ≈ 78,1%`. Đây chỉ là ví dụ số học; không phải kết quả benchmark.

## Sampler quyết định mẫu nào đi cùng nhau

| Cách dùng hiện tại | Sampler | Lợi ích | Đánh đổi |
| --- | --- | --- | --- |
| Pretrain, có shuffle | `RandomizedSortishBatchSampler` | Số mẫu gần cố định; nhóm độ dài tương tự giảm padding | Số token thật mỗi batch vẫn khác nhau |
| SFT, có shuffle | `TokenBalancedSortishBatchSampler` | Gom lượng token chịu loss gần mục tiêu; vẫn ưu tiên độ dài gần nhau | Số mẫu mỗi batch thay đổi; mục tiêu token là ngưỡng, không phải trần cứng |
| Val/test, không shuffle | Sắp theo độ dài, cắt nhóm `batch_size` | Cách duyệt rõ ràng; giảm padding | Không dùng thứ tự ngẫu nhiên như train |

Sortish xáo thứ tự mẫu rồi sắp theo độ dài trong từng pool; batch vẫn được xáo trước khi trả ra. Nó cân bằng giữa giảm padding và giữ biến thiên thứ tự, không phải sắp cả epoch từ ngắn đến dài.

SFT đặt `target_loss_tokens` bằng trung bình số token chịu loss mỗi mẫu nhân `batch_size`, làm tròn và ít nhất là 1. Sampler ngừng gom khi đã đạt mục tiêu hoặc đạt giới hạn số mẫu. Một mẫu được thêm vào có thể khiến tổng vượt mục tiêu. Giới hạn mẫu hiện do loader đặt bằng bốn lần `batch_size`.

## Collator làm gì sau sampler?

Collator chọn bucket đủ chứa mẫu dài nhất, làm tròn theo bội số 128 và không vượt `max_seq_len`. Nếu bucket gần cuối không còn phù hợp, nó dùng đúng `max_seq_len`. Sau đó đệm X, Y, mask và tạo `LMBatch`. Đây là padding theo batch, không phải cắt dữ liệu dài tùy ý; encoder phải bảo đảm mẫu phù hợp context từ trước.

## Batch logic và microbatch không phải một thứ

Sampler tạo batch logic. Engine có thể chia batch đó thành nhiều microbatch để giảm bộ nhớ tại một thời điểm; gradient accumulation có thể tích lũy nhiều batch logic trước một lần optimizer step. Vì vậy không suy số optimizer step trực tiếp từ số mẫu hoặc số microbatch. Chi tiết thuộc [phòng Vận hành](../../engine/docs/index.md).

## Tái lập thứ tự

Sampler lưu `seed` và `epoch` trong trạng thái. Khi resume giữa epoch, `skip_next_batches` bỏ qua phần đã xử lý mà vẫn giữ đúng thứ tự phần còn lại. Điều này chỉ giải quyết thứ tự sampler; tái lập toàn run còn phụ thuộc config, RNG, checkpoint và dữ liệu giữ nguyên.

Tiếp theo: [Token và mask](tokens-masks.md) hoặc [đọc hiệu suất padding](read-results.md).
