---
schema_version: 1
department: tokenizer
page_id: chunking
title: Chia cửa sổ pretrain và overlap
description: Hiểu cách giữ văn bản dài bằng cửa sổ chồng lấn và đánh đổi giữa ngữ cảnh với lượng token học lại.
order: 40
kind: concept
updated: "2026-10-25"
tags: [chunking, overlap, context, pretrain, ranh giới]
sources:
  - foundation_lab/tokenizer/chunking.py
  - foundation_lab/tokenizer/encode.py
  - tests/test_encode.py
---

## Vì sao phải chia?

Model có giới hạn `max_seq_len`. Một bài dài hơn giới hạn không thể đưa nguyên vào một mẫu. Đường encode pretrain hiện tokenize cả bài một lần, thêm BOS/EOS rồi chọn các cửa sổ trên chuỗi token đó. Không tokenize lại từng đoạn văn sau khi cắt, giúp giữ nguyên cách phân đoạn của bài.

Với `s = [BOS] + tokens + [EOS]`, cửa sổ `[start, end)` tạo:

```text
x = s[start : end - 1]
y = s[start + 1 : end]
```

Cửa sổ tối đa `max_seq_len` token nên X dài tối đa `max_seq_len - 1`. Bài ngắn vừa cửa sổ cho X là BOS + nội dung và Y là nội dung + EOS. Mẫu giữa bài không tự thêm BOS/EOS mới; các token này chỉ ở đầu/cuối chuỗi toàn bài.

## Chọn ranh giới có ý nghĩa

Thuật toán tham lam ưu tiên theo thứ tự: hết đoạn văn → xuống dòng → hết câu → hết vế → đầu mục liệt kê → giữa từ → giữa một từ. Các vị trí bên trong khối liệt kê bị hạ ưu tiên; khi cần chia khối, cố gắng cắt giữa các mục. Đây là heuristic theo dấu và khoảng trắng, không phải bộ phân tích cú pháp hiểu nghĩa câu.

Snapshot các hằng số ngày 25/10/2026:

| Hằng số | Giá trị | Ý nghĩa |
| --- | --- | --- |
| `MIN_FILL_RATIO` | 0,6 | Vùng tìm điểm cuối bắt đầu quanh 60% cửa sổ |
| `OVERLAP_RATIO` | 0,2 | Độ chồng lấn mục tiêu bằng 20% cửa sổ |
| `OVERLAP_TOLERANCE` | 0,1 | Cho phép lệch để tìm ranh giới tốt |

Các tỷ lệ này là hằng số implementation, chưa phải các trường tùy chọn của CLI. Khi hiển thị mốc sống trong UI, đọc nguồn cấu hình có thật, không biến bảng snapshot thành bộ điều khiển giả.

## Overlap giữ chuyển tiếp, đồng thời tăng token học lại

```text
Cửa sổ 1: [A B C D E F]
Cửa sổ 2:         [E F G H I J]
```

Các cửa sổ kề nhau luôn chồng ít nhất một token. Vì thế mỗi cặp token và token kế tiếp trong chuỗi nằm trọn trong ít nhất một cửa sổ, không bỏ mất target ở mép cắt. Khi overlap dài hơn, nhiều target được học lặp với ngữ cảnh khác nhau. Số token encode sẽ lớn hơn số token nội dung duy nhất.

Overlap không tạo khả năng attention vượt context: mỗi mẫu vẫn được xử lý độc lập. Không gọi nó là “model nhớ toàn bộ bài”. Nó giữ một phần ngữ cảnh lân cận cho mẫu sau với chi phí thêm tính toán.

## Các lựa chọn nền và tình trạng hỗ trợ

| Phương án | Đánh đổi | Trong encoder hiện tại |
| --- | --- | --- |
| Bỏ bài dài | Dễ làm nhưng mất dữ liệu | Không dùng cho pretrain; SFT quá dài hiện bị loại |
| Cắt cứng phần đầu | Giữ được một mẫu nhưng mất phần sau | Không phải chính sách pretrain hiện tại |
| Cửa sổ không overlap | Ít học lặp nhưng cần xử lý target ở mép cắt cẩn thận | Chưa là option cấu hình |
| Cửa sổ overlap theo ranh giới | Giữ nội dung dài và ngữ cảnh mép, tăng token lặp | Đang dùng cho pretrain |

Khi đánh giá, xem các cửa sổ từ cùng bài có thật sự được giữ ngoài train hay không; bộ split theo vị trí mẫu không tự chứng minh độc lập theo tài liệu. Xem [Dữ liệu](../../data/docs/index.md).
