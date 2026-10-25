---
schema_version: 1
department: eval
page_id: lm-loss
title: "CE loss trên validation và test"
description: "Hiểu loss trung bình theo token chịu loss, teacher forcing và điều kiện để so perplexity."
order: 10
kind: concept
updated: "2026-10-25"
tags: [CE, perplexity, teacher forcing, validation, test]
sources:
  - foundation_lab/eval/lm_loss.py
  - foundation_lab/objectives/lm_ce.py
  - foundation_lab/recipes/base.py
  - foundation_lab/engine/loop.py
  - foundation_lab/data/batch.py
---

## Model đang được yêu cầu làm gì?

Đánh giá CE đưa chuỗi đúng từ bộ dữ liệu vào model và hỏi xác suất model dành cho token đích ở từng vị trí. Đây là teacher forcing: ngữ cảnh trước đó lấy từ dữ liệu, không phải chuỗi do model tự sinh rồi tiếp tục đọc lại.

Vì vậy CE tốt có nghĩa model dự đoán các token được chấm tốt hơn trong điều kiện này. Nó không đồng nghĩa câu trả lời tự sinh luôn đúng, vì khi sinh model còn chịu ảnh hưởng lựa chọn decoder và những token nó đã chọn trước đó.

## Công thức đang dùng trong Lab

`evaluate_lm_loss` đặt model ở chế độ eval, dùng `torch.no_grad()`, chia microbatch để xử lý, tính `lm_ce` theo chunk và cộng hai tổng:

```text
total_loss   = tổng mọi token (loss_weight × CE)
total_weight = tổng mọi token loss_weight
reported_loss = total_loss / (total_weight + 1e-8)
```

Đây là trung bình trên token chịu loss của toàn loader, không phải trung bình không trọng số các batch. Padding không làm tăng mẫu số nếu weight bằng 0. Với SFT, phần prompt được mask không bị chấm trực tiếp, trong khi token trả lời/EOS chịu loss theo dữ liệu đã encode.

Ví dụ minh hoạ: batch A có 10 token chịu loss, CE trung bình 1; batch B có 90 token, CE trung bình 3. Loss chung là `(10×1 + 90×3)/100 = 2.8`, không phải `(1+3)/2 = 2`. Luôn giữ đúng mẫu số khi tự tổng hợp báo cáo.

## Validation và test có vai trò khác nhau

Validation được dùng trong quá trình phát triển để theo dõi và chọn checkpoint best theo val loss. Test dành cho đánh giá cuối sau khi đã quyết định cấu hình. Nếu liên tục nhìn test rồi sửa hyperparameter để tối ưu nó, tập đó dần trở thành dữ liệu chọn model, không còn phép kiểm tra độc lập như ban đầu.

Trong code hiện tại, val được gọi cuối epoch. `best.pt` lưu khi val loss giảm. Cuối stage, `final_eval` quyết định test dùng trọng số `last` hay nạp `best`. Vì vậy phải đọc cột/field `weights`, không mặc định test luôn của best.

## Perplexity là gì?

Với CE trung bình dùng log tự nhiên và cùng quy tắc chấm token, perplexity được tính bằng `exp(CE)`. Loss 2 tương ứng perplexity khoảng 7.39; đây là ví dụ toán học. Lab hiện trả CE scalar, chưa có cột PPL trong helper hoặc một dashboard eval độc lập.

PPL nhạy với tokenizer, lượng context và cách chia chuỗi; không xếp hạng trực tiếp hai tokenizer khác nhau chỉ từ PPL/token. Khi chỉ chấm một phần token của SFT, phải ghi rõ phạm vi đó. [Hướng dẫn perplexity của Hugging Face](https://huggingface.co/docs/transformers/perplexity).

## Điều kiện tối thiểu để đọc một CE

Cần biết checkpoint nào, stage/split nào, tokenizer nào, loss mask nào, độ dài/cách cắt nào, có bao nhiêu token được chấm và code/config nào tạo ra kết quả. Một loader không có token chịu loss không phải một phép đánh giá có ý nghĩa, dù công thức có epsilon để tránh chia 0. Epsilon là chi tiết số học, không thay thế kiểm tra dữ liệu.

Snapshot implementation: 25/10/2026. Cách chuẩn hoá train khác phép mean eval này; xem [Objective](../../objectives/docs/index.md).
