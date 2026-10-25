---
schema_version: 1
department: core
page_id: overview
title: Sổ tay Ban điều hành
description: Hiểu cách Lab ghép model, dữ liệu và các stage thành một thí nghiệm có thể kiểm tra và truy vết.
order: 0
kind: overview
updated: "2026-10-25"
tags: [core, cấu hình, config, experiment, bắt đầu]
sources:
  - foundation_lab/__init__.py
  - foundation_lab/core/experiment.py
  - foundation_lab/core/tasks.py
  - foundation_lab/cli.py
  - web/site/src/lib/lab/workbench.ts
---

## Phòng này giúp bạn quyết định điều gì?

Trước khi chạy một thí nghiệm, bạn cần trả lời: đang dùng model nào, dữ liệu nào, học theo phương pháp nào, chạy những giai đoạn nào và lấy trọng số ban đầu từ đâu. Ban điều hành quy định cách viết và kiểm tra những lựa chọn đó.

Đầu vào thường là một file YAML. Đầu ra của bước kiểm tra là cấu hình đã giải quyết các tham chiếu và kế thừa, cùng kết quả hợp lệ hoặc lỗi cụ thể. **Hợp lệ về cấu hình không có nghĩa là thí nghiệm chắc chắn vừa RAM, dữ liệu không rò rỉ hay model đã tốt.**

## Hiện phòng dùng được đến đâu?

Tại lần đối chiếu 25/10/2026, trang Ban điều hành trên web đang là trang chờ. Chức năng dự kiến là xem config đã resolve và những thay đổi so với mặc định. CLI `validate` đã có, nhưng cờ `validate --json` ghi trên trang chờ chưa có trong parser hiện tại. Đọc [cách dùng](workflow.md) để biết lệnh thật và giới hạn kiểm tra.

Sổ tay này cung cấp kiến thức ngay cả khi chưa có bàn cấu hình web. Các ví dụ lệnh là nội dung tham khảo, không tự chạy khi mở bài.

## Đường đọc theo nhu cầu

- Lần đầu dùng Lab: [Experiment, stage và task](concepts.md) → [Cấu hình được ghép thế nào](config-resolution.md) → [Cách kiểm tra](workflow.md).
- Muốn biết SAI đang chạy theo gì: [Mặc định hiện tại](current-defaults.md) → [Mốc so sánh và tái lập](reproducibility.md).
- Thấy tên option lạ hoặc lỗi không tương thích: [Registry và capability](registry-capabilities.md) → [Câu hỏi thường gặp](faq.md).

## Lab đi qua những phòng nào?

```text
Config/spec → Ban điều hành kiểm tra
            → Kiến trúc dựng model
            → Dữ liệu tạo batch
            → Phương pháp huấn luyện chọn recipe và Objective
            → Vận hành chạy bước học
            → Đánh giá đo kết quả
            → Lưu trữ ghi run, config và trọng số
```

Tokenizer chuẩn bị token trước pipeline. Suy luận đọc checkpoint và thử cách sinh câu trả lời sau hoặc ngoài quá trình train. Một tác vụ so kernel có đường đi ngắn hơn, không cần train hay đọc tập văn bản thật.

## Những phân biệt đầu tiên cần nhớ

| Hai khái niệm | Khác nhau ở đâu? |
| --- | --- |
| Model config và experiment config | Một cái mô tả cấu trúc model; một cái mô tả cả lượt thí nghiệm |
| Tên option và implementation | Tên là khóa tra trong registry; không tự tạo khả năng mới khi viết một tên bất kỳ |
| Validate và run | Validate kiểm cấu hình; run thực sự tiêu thụ tài nguyên và ghi kết quả |
| Init và resume | Init lấy trọng số làm điểm bắt đầu; resume tiếp tục trạng thái một lượt train |
| Mặc định hiện tại và config run cũ | Mặc định có thể đã đổi; run cũ có resolved config và lineage của chính nó |

Đọc tiếp về [Phương pháp huấn luyện](../../recipes/docs/index.md), [Lưu trữ](../../state/docs/index.md) hoặc [Kiến trúc](../../architecture/docs/index.md) khi cần đi sâu.
