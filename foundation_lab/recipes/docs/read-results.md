---
schema_version: 1
department: recipes
page_id: read-results
title: Đọc loss, best và kết quả huấn luyện
description: Diễn giải metric recipe trong đúng phạm vi dữ liệu, objective và trọng số được đánh giá.
order: 60
kind: interpretation
updated: "2026-10-25"
tags: [kết quả, loss, best, last, FAQ, đánh giá]
sources:
  - foundation_lab/recipes/base.py
  - foundation_lab/recipes/causal_lm.py
  - foundation_lab/recipes/sft.py
  - foundation_lab/engine/loop.py
  - foundation_lab/eval/lm_loss.py
  - foundation_lab/state/checkpoint.py
---

## Những kết quả recipe trả về

| Field | Cách đọc |
| --- | --- |
| `best_val_loss` | Val CE thấp nhất được ghi nhận để chọn best trong quá trình train |
| `test_loss` | CE trung bình theo token chịu loss trên test loader |
| `test_weights` | `last` hoặc `best`: bộ trọng số thực sự dùng tính test_loss |
| `optimizer_steps` | Tổng step optimizer được ghi ở cuối stage, không phải số microbatch |

Loop ghi train/val metrics theo stage; test metric có thêm nhãn weights. `best.pt` là trọng số tại thời điểm val cải thiện. `last.ckpt.pt` là trạng thái huấn luyện đầy đủ để resume, không đồng nghĩa checkpoint chất lượng tốt nhất.

## Loss thấp hơn có nghĩa gì?

Với cùng tokenizer, dữ liệu, mask và cách tính, CE thấp hơn nghĩa model gán xác suất cao hơn cho các token đích được chấm. Nó không trực tiếp đo tính đúng sự thật, khả năng làm theo yêu cầu hoặc chất lượng đối thoại toàn bài. Hai câu trả lời hợp lệ khác nhau vẫn có thể cho CE khác nhau nếu chỉ một câu được dùng làm nhãn.

Train loss là mức trung bình tích lũy trên phần train đã đi qua trong epoch, với model đang thay đổi; val/test được tính khi model ở chế độ eval. Không coi khoảng cách train–val là một phép so cùng thời điểm và cùng phân phối hoàn hảo. Nếu objective train được thay đổi, val/test hiện vẫn là CE, nên phải ghi rõ đơn vị metric.

## So sánh cần khớp những điều kiện nào?

Giữ rõ experiment/model, tokenizer, split và source data, mask, init, ngân sách train, frozen blocks, seed, objective và `test_weights`. Cùng số epoch chưa đảm bảo cùng số token học hoặc optimizer step, nhất là khi sampler SFT/resample thay đổi.

Với baseline SFT2 hiện tại, phải kèm giới hạn rò rỉ val/test do trộn lại giữa epoch. Đừng kết luận “SFT2 tổng quát hóa tốt” chỉ từ đường loss giảm trên một tập có thể đã đi vào train. Muốn so độc lập cần thiết kế dữ liệu tương ứng được duyệt.

## Câu hỏi thường gặp

### Vì sao best khác last?

Best được chọn bằng val; last là trọng số cuối vòng lặp. Học thêm có thể làm val tăng dù train giảm. Pretrain/continued mặc định test trên last; SFT mặc định test trên best, trừ khi config override. Khi báo test, luôn ghi nhãn weights.

### Vì sao learning rate giảm mà loss vẫn dao động?

Các batch khác nhau về nội dung và lượng token chịu loss; model còn thay đổi và loss log tích lũy theo epoch. Quan sát xu hướng, dữ liệu và số step; không quy một đoạn dao động cho một tham số duy nhất.

### Freeze nhiều block có chắc chống overfit không?

Không. Nó giảm số tham số được cập nhật nhưng những phần còn trainable vẫn có thể quá khớp dữ liệu; cũng có thể không đủ khả năng thích nghi. Cần so cùng baseline và đánh giá thực.

### Resume có tạo một thí nghiệm mới không?

Nó tiếp tục run đang có, dùng config và checkpoint của run. Nếu mục tiêu là đổi phương pháp/params để so sánh, cần một thí nghiệm mới có lineage rõ, không sửa config của run cũ rồi gọi đó là resume tương đương.

### Val loss thấp có phải nên phát hành ngay?

Chưa. Cần biết dữ liệu độc lập, checkpoint nào được đánh giá, và hành vi suy luận trên câu hỏi phù hợp. [Đánh giá](../../eval/docs/index.md) và [Suy luận](../../inference/docs/index.md) giúp bổ sung góc nhìn ngoài token loss.

Quay lại [Tổng quan](index.md).
