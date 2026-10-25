---
schema_version: 1
department: data
page_id: read-results
title: Đọc số liệu dữ liệu và tránh kết luận sai
description: Diễn giải số mẫu, số token, tỷ lệ mask và hiệu suất padding trong đúng bối cảnh.
order: 60
kind: interpretation
updated: "2026-10-25"
tags: [kết quả, FAQ, token, padding, thống kê]
sources:
  - foundation_lab/data/io.py
  - foundation_lab/data/loaders.py
  - foundation_lab/data/samplers.py
  - foundation_lab/tokenizer/encode.py
  - foundation_lab/recipes/sft.py
  - foundation_lab/eval/lm_loss.py
  - web/site/src/lib/lab/workbench.ts
---

## Một bảng số liệu cần những định nghĩa nào?

Phòng web chưa sinh báo cáo `inspect-data`. Bảng dưới đây định nghĩa cách đọc log/artifact hiện có và ý nghĩa cần giữ khi chức năng thống kê được triển khai; không phải danh sách metric đã có sẵn trong UI.

| Đại lượng | Định nghĩa | Cần tránh |
| --- | --- | --- |
| Số tài liệu giữ lại | Mẫu thô có ít nhất một kết quả encode | Đồng nhất với số cửa sổ |
| Số mẫu encode | Tổng mẫu X/Y sau chunk | Cho rằng mỗi mẫu là một tài liệu độc lập |
| Số token encode | Tổng `lengths`, tức độ dài X | Gọi là token chịu loss của SFT |
| Token chịu loss | Tổng `loss_weight` | Tính prompt vào lượng giám sát SFT |
| Ô token đã pad | Tổng B×T của các batch | Gọi là lượng dữ liệu nội dung |
| Hiệu suất padding | Token thật chia ô token đã pad | Chỉ lấy tỷ lệ của một batch làm đại diện toàn dataset |

Độ dài trung bình không mô tả đuôi dài. Khi có thống kê phân bố, cần xem cả trung vị, các phân vị cao và giá trị lớn nhất. So hai cách batching trên cùng tập mẫu; chia tổng token thật cho tổng ô token giúp tổng hợp đúng trọng số của batch lớn và nhỏ.

## Token thật nhiều nhưng token học ít

Một hội thoại có prompt dài và câu trả lời ngắn vẫn tốn chi phí forward trên prompt, nhưng mask chỉ cho học phần model. Vì vậy hiệu suất padding cao không đồng nghĩa nhiều token giám sát. Cần đọc đồng thời tỷ lệ token chịu loss trên token thật và tỷ lệ token thật trên ô đã pad.

## Những câu hỏi thường gặp

### Vì sao nhiều mẫu hơn chưa chắc nhiều thông tin hơn?

Một bài dài được chia thành nhiều cửa sổ có overlap. Số mẫu và số token tăng, nhưng nội dung chồng lấn được học lại. Hai nguồn có trùng lặp cũng làm tăng số lượng mà chưa tăng độ đa dạng tương ứng. Lab hiện không cung cấp chứng nhận deduplication.

### Vì sao batch SFT có số mẫu thay đổi?

Sampler ưu tiên gom đủ số token chịu loss và chặn số mẫu tối đa. `batch_size` đóng vai trò xác định mục tiêu token từ trung bình dataset, không bắt buộc mọi batch có đúng số hội thoại đó.

### Manifest nhỏ có nghĩa là dataset nhẹ không?

Không. Manifest chỉ là bảng kê; loader hiện nạp và ghép các shard vào RAM. Cần biết tổng dataset và cách nạp, không chỉ kích thước JSON hoặc NPZ nén.

### Cùng split_seed thì so sánh có công bằng không?

Chỉ khi đầu vào và thứ tự đầu vào không đổi mới có cùng phép chia theo vị trí. Đổi mix, shard, chunk hoặc tokenizer có thể đổi membership split. Cần xem [split và trộn nguồn](split-mix.md).

### Val/test rất thấp có đủ để kết luận tốt hơn không?

Chưa đủ. Kiểm nguồn đánh giá có từng lọt train không, mask có chấm cùng phần không, tokenizer có giống nhau không, và test dùng `last` hay `best`. Đặc biệt SFT2 resample hiện có giới hạn rò rỉ đã biết. Loss thấp trên tập bị nhiễm không chứng minh khả năng tổng quát hóa.

Quay lại [Tổng quan](index.md) hoặc sang [Phương pháp huấn luyện](../../recipes/docs/index.md).
