---
schema_version: 1
department: eval
page_id: results
title: "Diễn giải kết quả và giới hạn hiện có"
description: "Đọc train, val, test, best và last đúng ngữ cảnh, đặc biệt với hạn chế split của SFT2."
order: 50
kind: interpretation
updated: "2026-10-25"
tags: [kết quả, overfitting, best, last, leakage, giới hạn]
sources:
  - foundation_lab/eval/lm_loss.py
  - foundation_lab/engine/loop.py
  - foundation_lab/recipes/base.py
  - foundation_lab/recipes/causal_lm.py
  - foundation_lab/recipes/sft.py
  - foundation_lab/core/configs/base.yaml
  - foundation_lab/state/metrics.py
---

## Đọc nhãn trước con số

Một giá trị `loss` cần có `stage`, `split` và nhận diện trọng số. Pretrain và SFT thường chấm token trên dữ liệu/mask khác nhau; không xếp hạng trực tiếp bằng cách lấy con số nhỏ hơn. Loss ở hai epoch còn phải hiểu cùng quy tắc tổng hợp và tập được chấm.

Val được tính bằng CE ở cuối epoch. Code lưu `best.pt` khi val thấp hơn mức tốt nhất đã có. Test ở cuối stage dùng `final_eval`, và metric test ghi `weights` để biết `best` hay `last`. `best_val_loss` là giá trị tốt nhất theo val; nó không phải test loss và không có nghĩa model tốt nhất cho mọi prompt.

## Những mẫu thường gặp

| Quan sát | Cách hiểu có thể phù hợp | Chưa thể kết luận |
| --- | --- | --- |
| Train và val cùng giảm | Dự đoán cải thiện trên hai tập được đo | Chất lượng mọi tác vụ đều tăng |
| Train giảm, val tăng | Có thể overfit hoặc phân phối/điều kiện chấm khác | Chỉ nhìn một điểm đã chắc chắn overfit |
| Test tốt hơn val | Có thể tập test dễ hơn hoặc checkpoint khác | Test bị sai chỉ vì thấp hơn val |
| CE giảm, câu sinh vẫn lặp | Teacher forcing và giải mã tự sinh đo hai điều khác nhau | CE không còn giá trị |
| Loss tụt đột ngột khi đổi stage | Dữ liệu, mask hoặc objective có thể thay | Model tiến bộ tương ứng với độ tụt |

Trước khi giải thích bằng một nguyên nhân, kiểm tra config đã resolve, checkpoint được chấm và nguồn dữ liệu. Nếu đổi objective train, nhớ val/test vẫn dùng `lm_ce`; đường train loss có thể không còn cùng ý nghĩa với CE val.

## Best và last trong cấu hình chuẩn hiện tại

Snapshot `foundation_lab/core/configs/base.yaml` ngày 25/10/2026 đặt `final_eval: last` cho pretrain và continued, `final_eval: best` cho SFT1/SFT2. Continued đang tắt trong pipeline chuẩn. Đây là trạng thái file cơ sở; experiment riêng và override có thể thay. Khi đọc run, lấy giá trị đã resolve của chính run.

Vì pretrain test có thể dùng last còn SFT1 khởi tạo từ pretrain best, một con số test pretrain không nhất thiết chấm đúng trọng số dùng để khởi tạo SFT1. Ghi rõ vai trò thay vì gộp thành một nhãn “checkpoint pretrain”.

## Hạn chế đã được source ghi nhận ở SFT2

Recipe SFT và config cơ sở đã ghi: khi `resample_per_epoch=True`, tập trộn train được tạo lại theo epoch và split lại; val/test cố định tạo từ một lần trộn khác. Một phần mẫu val/test có thể nằm trong train các epoch. Cấu hình SFT2 chuẩn hiện bật chế độ này.

Hệ quả là val/test có thể lạc quan và việc chọn best theo val bị lệch. Những kết quả đó vẫn mô tả phép tính trên tập đã dùng, nhưng **không được coi là bằng chứng độc lập về khả năng tổng quát hoá**. Khi lập báo cáo phải ghi hạn chế này; không che bằng việc chỉ đổi tên metric hoặc chọn checkpoint có loss nhỏ hơn.

Sổ tay chỉ phản ánh hạn chế đã biết, không sửa nguồn dữ liệu hoặc chạy lại. Một đánh giá đáng tin hơn cần phương án split/holdout phù hợp và phép chạy mới đã được đồng ý; giữ nguyên run cũ để còn truy nguyên.

## Cách viết kết luận vừa đủ

Nên ghi: “Checkpoint A có CE thấp hơn B trên cùng split X và cùng quy tắc chấm; chưa đo chất lượng hội thoại”, hoặc “SFT2 có val thấp hơn nhưng split có rủi ro rò rỉ đã biết”. Kèm số tuyệt đối, điều kiện và những điều chưa đo. Không đổi “không có dữ liệu” thành 0 hay “không có khác biệt quan sát được” thành chứng minh tương đương toàn diện.
