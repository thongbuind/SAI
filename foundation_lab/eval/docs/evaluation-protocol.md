---
schema_version: 1
department: eval
page_id: evaluation-protocol
title: "Thiết kế phép đánh giá công bằng"
description: "Xác định câu hỏi, baseline, dữ liệu giữ lại và điều kiện so sánh trước khi đọc kết quả."
order: 30
kind: guide
updated: "2026-10-25"
tags: [phương pháp, baseline, tái lập, leakage, thí nghiệm]
sources:
  - foundation_lab/eval/compare.py
  - foundation_lab/eval/lm_loss.py
  - foundation_lab/inference/bench.py
  - foundation_lab/state/run.py
  - foundation_lab/core/configs/base.yaml
  - foundation_lab/recipes/sft.py
---

## Bắt đầu bằng quyết định muốn đưa ra

Viết một câu rõ: “Chọn kernel chạy cùng phép tính với chi phí thấp hơn”, “chọn checkpoint có CE thấp hơn trên tập giữ lại cố định”, hoặc “chọn cách giải mã trả lời ít lặp hơn mà vẫn đúng”. Mỗi câu cần dữ liệu và tiêu chí khác nhau. Không dùng kết quả benchmark số học để kết luận model biết nhiều hơn.

## Mốc phải đại diện cấu hình đang dùng

Với so kernel, mốc lấy từ `attention_kernel` của model đã resolve. Với so objective, mốc lấy từ params của stage đã gộp mặc định recipe. Với inference, spec phải có biến thể decoder chính thức cùng `GenerationConfig()` mặc định. Không viết tay một bộ tham số gần giống rồi gọi là mặc định.

Khi lập phương án so checkpoint, hãy chọn mốc là trọng số phát hành phù hợp model/tokenizer và cách đánh giá mặc định hiện tại. Phòng Đánh giá chưa có công cụ độc lập thực hiện phép so này trong snapshot hiện tại.

## Giữ cùng điều kiện, đổi có chủ đích

| Loại phép so | Giữ cố định | Phần thay |
| --- | --- | --- |
| Kernel tương đương | Q/K/V, shape, dtype, mask, thiết bị | Implementation kernel |
| Objective tương đương | Hidden/head/targets/weight, dtype, thiết bị | Implementation hoặc chunk |
| Checkpoint trên CE | Tập dữ liệu, tokenizer, mask, cách cắt/context, quy tắc tổng hợp | Trọng số tương thích |
| Decoder trên cùng checkpoint | Checkpoint, prompt, cách chuẩn bị input, thiết bị | Các tham số giải mã được nêu rõ |

Nếu muốn đổi nhiều yếu tố, ghi rõ kết quả là đánh giá một cấu hình tổng thể; không quy công cho riêng một yếu tố. Khi so hiệu năng, giữ điều kiện máy và ghi chi phí thực tế. Khi sampling ngẫu nhiên, một seed có ích cho tái lập nhưng không chứng minh kết luận vững qua mọi seed.

## Bảo vệ ý nghĩa của dữ liệu giữ lại

Train dùng để cập nhật model, validation để chọn cấu hình/checkpoint, test để báo cáo cuối. Dữ liệu test lọt vào train hoặc bị dùng nhiều lần để chọn phương án làm số đo lạc quan. Nguyên tắc tách dữ liệu và tránh rò rỉ được trình bày trong [tài liệu chính thức scikit-learn](https://scikit-learn.org/stable/common_pitfalls.html).

Repo đã ghi hạn chế của SFT2: `resample_per_epoch: true` trộn/split lại khiến mẫu val/test có thể lọt vào train các epoch, nên không dùng các số này như bằng chứng tổng quát hoá độc lập. Xem [giới hạn hiện có](results.md). Sổ tay không tự sửa split, di chuyển run hoặc thay dữ liệu để xử lý vấn đề đó.

## Ghi đủ để người khác đọc lại

Một ghi chép có ích gồm câu hỏi, baseline, biến thay đổi, config đã resolve, nhận diện checkpoint, dữ liệu/tokenizer/split, số lượng mẫu hoặc token được chấm, điều kiện thiết bị, phiên bản code/lab, kết quả và giới hạn. Dùng lineage và bản ghi run hiện có khi sẵn có; không tự bổ sung giá trị không đo.

Kết luận nên vừa phạm vi: “chunk nhỏ hơn khớp loss/gradient trên workload đã đo” hoặc “checkpoint A có CE thấp hơn B trên tập X với quy tắc Y”. Không mở rộng thành “A tốt nhất” nếu chưa đo các tác vụ cần dùng.

Đây là hướng dẫn phương pháp dựa trên các công cụ hiện hữu ngày 25/10/2026; khung đánh giá nhiều checkpoint và thống kê qua nhiều seed chưa được tuyên bố là chức năng web đã có.
