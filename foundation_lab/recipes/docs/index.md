---
schema_version: 1
department: recipes
page_id: overview
title: Sổ tay phòng Phương pháp huấn luyện
description: Hiểu recipe, stage và pipeline để thiết kế một thí nghiệm huấn luyện có mốc rõ ràng.
order: 0
kind: overview
updated: "2026-10-25"
tags: [recipe, huấn luyện, bắt đầu, pipeline]
sources:
  - foundation_lab/recipes/__init__.py
  - foundation_lab/recipes/base.py
  - foundation_lab/recipes/causal_lm.py
  - foundation_lab/recipes/sft.py
  - foundation_lab/core/configs/base.yaml
  - web/site/src/lib/lab/workbench.ts
---

## Recipe là gì?

Recipe là phương pháp huấn luyện: nhận model, dữ liệu, runtime và tham số của một stage, rồi tổ chức vòng học và trả kết quả. Stage là một bước có tên trong pipeline; cùng recipe có thể dùng cho nhiều stage với dữ liệu, số epoch hoặc trọng số khởi tạo khác nhau.

```text
Experiment = model + runtime + pipeline
Pipeline   = stage 1 → stage 2 → ...
Stage      = recipe + data + params + init + enabled
Recipe     = cách tạo loader, optimizer, objective và gọi vòng train
```

`pretrain` và `continued` dùng recipe `causal_lm`; `sft1` và `sft2` dùng recipe `sft`. Không có bốn thuật toán độc lập chỉ vì pipeline có bốn tên stage.

## Bản đồ đọc

| Bạn cần | Trang |
| --- | --- |
| Phân biệt pretrain, continued và SFT | [Các phương pháp đang có](methods.md) |
| Chọn điểm bắt đầu, init, resume và frozen blocks | [Stage, khởi tạo và tiếp tục run](stages-init-resume.md) |
| Hiểu learning rate, loss, accumulation và optimizer | [Tối ưu và chuẩn hóa loss](optimization.md) |
| Biết pipeline mặc định SAI và cách lấy baseline | [Mốc hiện tại của SAI](baseline.md) |
| Dùng CLI hiện tại và chuẩn bị thí nghiệm | [Quy trình sử dụng](workflow.md) |
| Đọc loss, best, last và kết luận đúng | [Đọc kết quả và câu hỏi thường gặp](read-results.md) |

## Trạng thái hiện tại

Snapshot ngày 25/10/2026: registry recipe có `causal_lm` và `sft`; CLI `run` đã thực hiện pipeline huấn luyện. Trang web phòng này vẫn `planned: run`, chưa cung cấp giao diện chạy/giám sát pipeline như mô tả đích. Không nhầm việc CLI đã có với việc command đã tích hợp vào web.

Các hướng distillation, DPO hoặc RLHF được docstring nhắc như khả năng tương lai, chưa được đăng ký thành recipe hiện hành. Sổ tay không hiển thị chúng như lựa chọn chạy được.

## Trước khi tin kết quả

Recipe phụ thuộc dữ liệu và mask, nên đọc thêm [Dữ liệu](../../data/docs/index.md) và [Tokenizer](../../tokenizer/docs/index.md). SFT2 hiện có giới hạn rò rỉ val/test khi resample giữa epoch; [mốc hiện tại](baseline.md) và [đọc kết quả](read-results.md) giải thích vì sao loss thấp chưa đủ chứng minh tiến bộ.
