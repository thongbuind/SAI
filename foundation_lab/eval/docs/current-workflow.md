---
schema_version: 1
department: eval
page_id: current-workflow
title: "Cách dùng phần đánh giá hiện tại"
description: "Tìm kết quả val/test trong run, dùng đúng phòng đã có và phân biệt lệnh dự kiến với CLI hoạt động."
order: 40
kind: guide
updated: "2026-10-25"
tags: [hướng dẫn, trạng thái, run, checkpoint, CLI]
sources:
  - web/site/src/lib/lab/workbench.ts
  - web/site/src/components/lab/PlannedBench.astro
  - foundation_lab/catalog.py
  - foundation_lab/cli.py
  - foundation_lab/state/run.py
  - foundation_lab/state/metrics.py
  - foundation_lab/recipes/base.py
  - foundation_lab/engine/loop.py
  - foundation_lab/pipeline.py
---

## Phòng web đang ở giai đoạn nào?

Snapshot 25/10/2026: phòng Đánh giá hiện trang chờ “Phòng đang được xây dựng”. Dòng `eval-loss` diễn tả lệnh dự kiến khi phòng hoàn thiện. Nó chưa được đăng ký thành tác vụ/CLI hiện tại, chưa có form checkpoint, split, stage hay nút chạy riêng.

Trong lúc phòng còn là trang chờ, bạn có thể dùng Sổ tay để hiểu định nghĩa, cách diễn giải và nơi lưu kết quả đã có. Việc đọc tài liệu không thực hiện một phép đánh giá mới.

## Đánh giá val/test đang được gọi ở đâu?

```text
Recipe train stage
  → train một epoch
  → evaluate_lm_loss trên val
  → nếu val tốt hơn: lưu best.pt
  → kết thúc các epoch
  → chọn last hoặc best theo final_eval
  → evaluate_lm_loss trên test
  → ghi metric và kết quả stage
```

Đó là một phần pipeline huấn luyện hiện có. Không dùng lệnh `run` như một lối tắt “eval-only”: chạy pipeline có thể train, cập nhật trọng số và tạo dữ liệu run mới. Nếu chỉ cần xem kết quả đã tính, đọc các file nhẹ của run trước.

## Tìm kết quả trong run

| File tương đối trong run experiment | Nội dung cần xem |
| --- | --- |
| `resolved_config.yaml` | Model, pipeline, stage, dữ liệu, params và runtime đã resolve |
| `lineage.json` | Dấu vết nguồn tạo run, môi trường/code và đầu vào theo hợp đồng lineage |
| `invocations.jsonl` | Những lần gọi CLI trên run, hữu ích khi có resume |
| `metrics.jsonl` | Metric theo thời gian; lọc đúng `stage`, `split`, epoch/step |
| `stages/<stage>/done.json` | Kết quả hoàn tất: best val, test loss, test weights, optimizer steps của recipe |
| `stages/<stage>/best.pt` | Trọng số best; không cần mở file lớn chỉ để đọc số loss |
| `stages/<stage>/last.ckpt.pt` | Checkpoint để resume; khác vai trò với báo cáo metric |

Run experiment có dạng `runs/<experiment>/<run_id>/`. Tác vụ so kernel/objective/infer dùng `result.json` và `events.jsonl`; đừng tìm đúng một định dạng cho mọi loại run. Xem [Lưu trữ](../../state/docs/index.md) để hiểu artifact và lineage.

## Những công cụ đã có thể dùng đúng mục đích

Để kiểm chứng kernel, mở [Kiến trúc](../../architecture/docs/index.md). Để so loss implementation hoặc chunk, mở [Objective](../../objectives/docs/index.md). Để đọc các câu trả lời từ cùng checkpoint dưới nhiều cách giải mã, mở [Suy luận](../../inference/docs/index.md). Các hướng dẫn ở những phòng đó chứa lệnh CLI đã có thật.

CLI hiện có lệnh mô tả schema/registry:

```bash
python -m foundation_lab describe
```

Đây là lệnh tra cứu được ghi lại từ source, không được thực thi trong phiên viết tài liệu. Mọi lệnh thực tế vẫn theo quy định chạy code của workspace. Không có ví dụ `eval-loss` thực thi trong sổ tay vì lệnh đó đang dự kiến.

## Trước khi có một phép eval mới

Chuẩn bị câu hỏi, checkpoint tương thích, tập dữ liệu/split cố định, tokenizer, quy tắc mask, baseline và ngân sách chạy; ghi thành phương án để triển khai công cụ phù hợp. Không tự đổi file trọng số, data hoặc run cũ để làm khớp bảng so sánh.
