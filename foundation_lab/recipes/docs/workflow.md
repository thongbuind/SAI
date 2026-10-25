---
schema_version: 1
department: recipes
page_id: workflow
title: Chuẩn bị và thực hiện một thí nghiệm recipe
description: Quy trình từ giả thuyết, baseline và validate đến run, init hoặc resume bằng CLI hiện có.
order: 50
kind: guide
updated: "2026-10-25"
tags: [hướng dẫn, CLI, run, validate, baseline]
sources:
  - foundation_lab/cli.py
  - foundation_lab/core/experiment.py
  - foundation_lab/pipeline.py
  - foundation_lab/recipes/base.py
  - web/site/src/lib/lab/workbench.ts
---

## Bắt đầu từ một câu hỏi cụ thể

Ví dụ câu hỏi tốt là “giữ nguyên dữ liệu và init, giảm learning rate ở SFT1 có làm val CE độc lập tốt hơn ở cùng ngân sách không?”. Câu hỏi này xác định một thay đổi, một baseline, một metric và những gì cần giữ giống nhau.

Phòng web hiện đang planned; các lệnh dưới đây đã có trong CLI nhưng chưa phải nút chạy trên web. Tất cả ví dụ chỉ để đọc, chưa được thực hiện trong phiên viết sổ tay.

## Quy trình thực hành

1. Chọn experiment và stage, đọc [mốc hiện tại](baseline.md). Ghi model, tokenizer, nguồn/split dữ liệu và checkpoint init.
2. Nêu một thay đổi candidate. Mốc phải lấy từ config hiện hành sau resolve; không tạo một baseline bằng các con số nhớ từ tài liệu.
3. Kiểm recipe nhận đúng data kind và capability model. SFT cần mask; causal_lm nhận lm.
4. Xem tác động tài nguyên: model, context, logical batch, microbatch, workers, số epoch và checkpoint. Không suy rằng ví dụ SAI_35M là test tí hon.
5. Sau khi người dùng cho phép chạy lệnh, validate config trước. Validate không nạp dữ liệu/model để chứng minh run chắc chắn thành công.
6. Chạy vào run mới hoặc resume đúng run cũ tùy mục tiêu. Giữ config, lineage, metrics và checkpoint theo run.
7. Đọc val/test với nhãn weights, kiểm rò rỉ dữ liệu, rồi bổ sung đánh giá phản hồi nếu câu hỏi liên quan chất lượng hội thoại.

## Các lệnh thật, chỉ là ví dụ

Resolve và kiểm schema một experiment:

```bash
python -m foundation_lab validate foundation_lab/core/configs/SAI_35M.yaml
```

Kiểm một candidate đổi learning rate, không viết config baseline thủ công:

```bash
python -m foundation_lab validate foundation_lab/core/configs/SAI_35M.yaml --set pipeline.sft1.params.learning_rate=1.0e-5
```

Chạy pipeline theo config hiện hành, sau khi đã có dữ liệu và quyền chạy phù hợp:

```bash
python -m foundation_lab run foundation_lab/core/configs/SAI_35M.yaml
```

Các giá trị candidate chỉ minh hoạ cú pháp, không phải khuyến nghị tối ưu đã đo. Lệnh run có thể train model thật, đọc dữ liệu lớn và ghi checkpoint; phải được người dùng duyệt trước theo quy tắc workspace.

## Chọn riêng stage hoặc tiếp tục

`--stages sft1,sft2` chọn stage theo tên, nhưng init `pretrain.best` vẫn cần file best trong run hoặc override đường dẫn bằng `--init sft1=...`. Không hướng người mới chọn stage sau rồi coi lỗi thiếu init là lỗi recipe.

`run --resume` nhận đường dẫn run thực đã có; dùng config đã lưu và không nhận experiment/`--set` mới. Không viết một path timestamp hư cấu vào nút copy như thể run đó tồn tại. Người dùng chọn run của mình trong [Lưu trữ](../../state/docs/index.md) hoặc từ đường dẫn CLI đã trả.

## Hoàn tất thí nghiệm có nghĩa gì?

Một stage hoàn tất khi có kết quả stage được ghi, không chỉ khi loss giảm hoặc log ngừng chạy. Sau đó đối chiếu `best_val_loss`, `test_loss`, `test_weights`, optimizer steps và điều kiện dữ liệu. Sổ tay không tự tạo run, khởi chạy lệnh, resume, xóa hay ghi đè checkpoint.
