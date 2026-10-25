---
schema_version: 1
department: state
page_id: init-and-resume
title: Khởi tạo từ trọng số và chạy tiếp một run
description: Chọn init hay resume theo mục tiêu và hiểu config, stage hoàn thành, RNG trong đường chạy hiện tại.
order: 30
kind: guide
updated: "2026-10-25"
tags: [init, resume, checkpoint, cli]
sources:
  - foundation_lab/cli.py
  - foundation_lab/pipeline.py
  - foundation_lab/state/checkpoint.py
  - foundation_lab/engine/loop.py
  - tests/test_pipeline.py
---

## Chọn hành động theo ý định

| Ý định | Cách phù hợp | Trạng thái tối ưu |
| --- | --- | --- |
| Tiếp tục stage bị ngắt, cùng thí nghiệm | `run --resume <run_dir>` | Khôi phục checkpoint stage nếu có |
| Bắt đầu thí nghiệm mới từ model đã học | Run mới với `init` hoặc `--init` | Optimizer/lịch mới theo recipe |
| Chạy stage sau từ best của stage trước | `init: <stage>.best` | Stage mới, chỉ nạp trọng số |
| Sinh câu trả lời | Tác vụ `infer` chọn checkpoint | Không tiếp tục optimizer |

Init chuyển điểm xuất phát của model. Resume tiếp tục một trạng thái huấn luyện đã lưu. Hai hành động không thể thay thế nhau chỉ vì đều đọc `.pt`.

## Resume dùng cấu hình nào?

CLI đọc `resolved_config.yaml` ngay trong run. Không được truyền thêm file experiment hoặc `--set` cùng `--resume`; muốn đổi config thì tạo run mới. Ví dụ minh hoạ, chưa chạy:

```bash
python -m foundation_lab run --resume runs/SAI_375M/<run_id>
```

`<run_id>` là chỗ cần thay bằng định danh thật trước khi dùng lệnh. Đây là thao tác có thể nạp model, dữ liệu và train dài; sổ tay không cấp quyền chạy tự động.

Pipeline xử lý theo thứ tự:

1. Stage có `done.json`: bỏ qua và đọc lại kết quả stage.
2. Stage chưa xong nhưng có `last.ckpt.pt`: nạp model, optimizer, scheduler và các trạng thái có trong checkpoint.
3. Stage chưa xong và chưa có checkpoint: bắt đầu stage theo quy tắc init hiện tại.

CLI vẫn nhận `--stages` và `--init` khi resume. `--init` chỉ có tác dụng khi stage thực sự đi qua nhánh khởi tạo; stage đã xong được bỏ qua, còn checkpoint resume của chính stage được ưu tiên. Không dùng `--init` để giả định reset optimizer của stage đang resume.

## Run mới với init

Ví dụ minh hoạ chọn SFT1 bắt đầu từ trọng số pretrain:

```bash
python -m foundation_lab run foundation_lab/core/configs/SAI_375M.yaml --stages sft1 --init sft1=model/pretrained_375M.pt
```

Đây là run mới. Tham số training của SFT1 đến từ experiment; optimizer và scheduler được dựng mới. Chỉ chạy stage SFT1 mà vẫn để `init: pretrain.best` đòi best của pretrain tồn tại trong **cùng run**; không tự đi tìm run pretrain tốt nhất ở nơi khác. `--init` là cách chỉ định một nguồn trọng số ngoài run một cách tường minh.

Mỗi lần gọi được ghi vào `invocations.jsonl`, gồm stage được chọn và init override. Vì override không cần viết lại resolved config, đọc config một mình có thể bỏ sót nguồn init của một lần gọi cụ thể.

## Vì sao RNG khôi phục ở vòng lặp?

Tạo iterator DataLoader có thể tiêu thụ số ngẫu nhiên. Loader checkpoint trả RNG cho engine; engine khôi phục trước hoặc sau bước tạo iterator tùy checkpoint ở cuối hay giữa epoch. Đồng thời sampler khôi phục thứ tự batch và bỏ qua phần đã train.

Test tí hon trong `tests/test_pipeline.py` đối chiếu resume với chạy liền, gồm dropout và loader dựng lại. Đây là bằng chứng phạm vi test CPU, không phải cam kết kết quả bitwise giống nhau trên mọi môi trường. Đọc [lineage](lineage.md) và [phiên bản](versions.md) trước khi tái sử dụng run.
