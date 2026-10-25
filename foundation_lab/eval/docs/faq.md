---
schema_version: 1
department: eval
page_id: faq
title: "Câu hỏi thường gặp về Đánh giá"
description: "Giải đáp các nhầm lẫn về phòng đang xây dựng, loss, perplexity, distinct và tính công bằng."
order: 60
kind: faq
updated: "2026-10-25"
tags: [FAQ, chất lượng, perplexity, đánh giá, giới hạn]
sources:
  - web/site/src/lib/lab/workbench.ts
  - foundation_lab/catalog.py
  - foundation_lab/eval/lm_loss.py
  - foundation_lab/eval/text.py
  - foundation_lab/eval/compare.py
  - foundation_lab/inference/bench.py
  - foundation_lab/recipes/base.py
  - foundation_lab/recipes/sft.py
---

## Vì sao phòng có Sổ tay nhưng chưa có nút chạy eval?

Sổ tay là nội dung hỗ trợ độc lập. Phòng web Đánh giá hiện đang được xây dựng; `eval-loss` là ý định phát triển, chưa là lệnh hoạt động ở snapshot 25/10/2026. Bạn vẫn có thể đọc và hiểu những kết quả val/test đã được pipeline tính trước đó.

## Loss bao nhiêu thì model tốt?

Không có ngưỡng chung cho mọi tokenizer, dữ liệu, mask và độ dài context. So với baseline phù hợp trên cùng tập và đọc thêm những tác vụ bạn cần model làm. Loss 1 của tập dễ không tự tốt hơn loss 2 của tập khó.

## Lower loss có luôn trả lời đúng hơn không?

Loss đo xác suất token đúng trong ngữ cảnh dữ liệu đã cho. Sinh câu còn phụ thuộc decoder, ngữ cảnh thực tế và lỗi tích luỹ. Loss là tín hiệu hữu ích nhưng không thay việc kiểm tra câu trả lời, tính đúng kiến thức hoặc bám hướng dẫn.

## Hai tokenizer khác nhau có so PPL trực tiếp được không?

PPL theo token đổi khi đơn vị token đổi. Cần một giao thức chung phù hợp trước khi xếp hạng; không chỉ đặt hai con số cạnh nhau. Phòng [Tokenizer](../../tokenizer/docs/index.md) giúp hiểu cách tách token và chi phí biểu diễn.

## Distinct-2 cao có nghĩa ít lỗi hơn không?

Không. Nó đo phần n-gram khác nhau trong các chuỗi đã sinh. Nội dung sai nhưng đa dạng vẫn có thể đạt cao. Văn bản rất ngắn còn làm tỷ lệ nhạy với mẫu nhỏ; khi không có bigram, helper trả 0. Luôn xem độ dài và văn bản gốc.

## Có thể chọn checkpoint theo test loss nhỏ nhất không?

Nếu dùng test để chọn trong nhiều checkpoint, bạn đã dùng nó như tập chọn model. Dùng validation cho quyết định đó và giữ test cho báo cáo sau cùng theo giao thức đã định. Pipeline hiện chọn best theo val, còn test chọn trọng số theo `final_eval`.

## Tại sao best và last cho kết quả khác?

Best là thời điểm val tốt nhất đã quan sát, last là trạng thái cuối vòng train. Train tiếp không bảo đảm val luôn giảm. Đọc metric `weights` và config run để biết test chấm cái nào; đừng suy ra từ tên thư mục stage.

## Compare-kernel Khớp có chứng minh checkpoint mới tốt không?

Không. Phép so dùng tensor ngẫu nhiên, không có checkpoint được huấn luyện trong lượt đo. Nó kiểm chứng implementation kernel trên điều kiện được chọn. Tương tự, compare-objective Khớp không là đánh giá khả năng sinh văn bản.

## Run bị hủy hoặc thiếu metric thì có thể điền 0 không?

Không. Thiếu metric nghĩa chưa có số đo hoặc lượt chạy chưa hoàn tất. Số 0 có ý nghĩa toán học riêng và có thể khiến bảng xếp hạng sai. Giữ trạng thái thiếu, lý do và phạm vi kết quả đã có.

## Có thể dùng val/test SFT2 chuẩn để kết luận tổng quát hoá không?

Source ghi nhận rủi ro split lại khi resample mỗi epoch, khiến mẫu giữ lại lọt vào train. Cần báo rõ giới hạn và có thiết kế đánh giá độc lập trước kết luận đó. Xem [Diễn giải kết quả](results.md); không sửa/xoá run cũ để che vấn đề.
