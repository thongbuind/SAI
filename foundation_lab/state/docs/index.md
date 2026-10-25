---
schema_version: 1
department: state
page_id: overview
title: Sổ tay Lưu trữ
description: Hiểu run, checkpoint, nguồn gốc thí nghiệm và điều kiện sử dụng lại kết quả.
order: 0
kind: overview
updated: "2026-10-25"
tags: [run, checkpoint, lưu trữ]
sources:
  - foundation_lab/state/__init__.py
  - foundation_lab/state/run.py
  - web/site/src/lib/lab/workbench.ts
  - web/site/src/components/lab/PlannedBench.astro
---

## Một kết quả cần đi kèm điều gì?

Một con số loss hoặc một câu trả lời không đủ để tái hiện thí nghiệm. Cần biết code, config, model, dữ liệu, tokenizer, môi trường, checkpoint và quá trình đã chạy. Phòng Lưu trữ tổ chức các bằng chứng đó trong **run**.

Run là một thư mục của một lượt thí nghiệm hoặc tác vụ. Một run huấn luyện có nhiều stage; mỗi stage có thể có trọng số tốt nhất, checkpoint tiếp tục và kết quả cuối. Một tác vụ như `infer` có nhật ký tiến trình và kết quả chung, không có pipeline stage huấn luyện.

## Trạng thái hiện tại

Snapshot 25/10/2026: code lưu artifact đã được pipeline và các tác vụ sử dụng. Trang web Lưu trữ vẫn đang được xây dựng, dự kiến đọc trực tiếp metadata trong `runs/`; phòng này không cần một lệnh CLI riêng. Không suy rằng đã có nút tải, xóa, đổi tên hay resume run trên trang này.

Sổ tay chỉ đọc kiến thức. Mở nó không quét toàn bộ run, không nạp checkpoint và không đổi artifact.

## Chọn bài theo nhu cầu

| Câu hỏi | Bài đọc |
| --- | --- |
| File nào nằm ở đâu và có ý nghĩa gì? | [Cấu trúc run](run-layout.md) |
| `best.pt` khác `last.ckpt.pt` thế nào? | [Trọng số và checkpoint](checkpoints.md) |
| Khi nào init, khi nào resume? | [Khởi tạo và chạy tiếp](init-and-resume.md) |
| Vì sao cùng config chưa đủ tái lập? | [Lineage và nguồn gốc](lineage.md) |
| Metric, event và result khác nhau thế nào? | [Hợp đồng kết quả](metrics-and-events.md) |
| Run từ phiên bản khác còn dùng được không? | [Phiên bản và tương thích](versions.md) |
| Thiếu file, không thấy checkpoint hoặc run dừng giữa chừng? | [Câu hỏi thường gặp](faq.md) |

## Ba phân biệt quan trọng

- **Trọng số tốt nhất** là lựa chọn theo val trong stage, không tự động là model tốt nhất về mọi câu hỏi.
- **Checkpoint tiếp tục** lưu cả trạng thái huấn luyện, không chỉ trọng số.
- **Có artifact** không tự chứng minh run hoàn thành hoặc kết quả đủ điều kiện so sánh. Cần đọc đúng dấu hoàn thành, config và lineage.

Sau khi chọn checkpoint để sinh câu trả lời, đọc [Sổ tay Suy luận](../../inference/docs/index.md).
