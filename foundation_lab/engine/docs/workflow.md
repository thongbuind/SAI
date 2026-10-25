---
schema_version: 1
department: engine
page_id: workflow
title: Quy trình sử dụng phòng Vận hành
description: Chuẩn bị cấu hình, chọn phạm vi thử nghiệm và hiểu ranh giới giữa CLI hiện có với bench-step dự kiến.
order: 60
kind: guide
updated: "2026-10-25"
tags: [hướng dẫn, cli, benchmark]
sources:
  - foundation_lab/cli.py
  - foundation_lab/core/experiment.py
  - foundation_lab/engine/loop.py
  - web/site/src/lib/lab/workbench.ts
  - web/site/src/components/lab/PlannedBench.astro
---

## Bước 1: xác định việc cần trả lời

Viết một câu cụ thể, chẳng hạn “giảm microbatch có giảm đỉnh bộ nhớ của cùng stage không?” hoặc “compile có giảm thời gian sau warmup không?”. Chọn một experiment, một stage và một biến thay đổi. Nếu thay cả batch, objective, precision và model, kết quả sẽ khó quy nguyên nhân.

Ghi mốc từ config hiện tại và tách rõ mốc khỏi ứng viên. Giữ model, dữ liệu, cách split, loss, mẫu số và số mẫu phù hợp với câu hỏi. Khi thay `batch_size` hay `accumulation_steps`, cần ghi nhận đây có thể là thay đổi cách tối ưu, không chỉ thay phân bổ RAM.

## Bước 2: đọc config bằng lệnh đã có

Ví dụ dưới đây chỉ là tài liệu, chưa được chạy:

```bash
python -m foundation_lab validate foundation_lab/core/configs/SAI_375M.yaml
```

`validate` resolve và kiểm tra schema/tương thích config trước khi nạp dữ liệu hay dựng model; nó không chứng minh một cấu hình đủ RAM hoặc đủ nhanh. Theo quy tắc workspace, agent vẫn cần quyền trước khi chạy lệnh Python ngoài test nhẹ được cho phép.

Ví dụ đọc cấu hình ứng viên qua override:

```bash
python -m foundation_lab validate foundation_lab/core/configs/SAI_375M.yaml --set runtime.compile=false --set pipeline.sft1.params.max_microbatch_tokens=8192
```

Số 8.192 chỉ minh hoạ cách override. Mốc vẫn là cấu hình resolve không có override đó. Khi lựa chọn đã được thống nhất, chỉ chạy trong môi trường đủ tài nguyên và theo quyền người dùng cấp.

## Bước 3: dùng đúng đường chạy hiện có

Trang web Vận hành hiện chưa có Control center hay nút thực thi `bench-step`. Sổ tay đọc được, nhưng mọi màn “so một bước train” là thiết kế tương lai.

Đường chạy hiện có là pipeline `run`. Một ví dụ có khả năng nạp dữ liệu/model thật và chạy lâu:

```bash
python -m foundation_lab run foundation_lab/core/configs/SAI_375M.yaml --stages sft1 --init sft1=model/pretrained_375M.pt --set runtime.profile_vram=true
```

Lệnh minh hoạ này đòi checkpoint khởi tạo phù hợp và dữ liệu đã chuẩn bị. Nó không phải smoke test, cũng không bảo đảm chạy trên máy 8 GB. Không tự chạy chỉ vì xuất hiện trong sổ tay.

## Bước 4: đọc kết quả có điều kiện

Đối chiếu train/val/test loss, thời gian, VRAM và config cùng nhau. Nếu compile có khởi động chậm, phân biệt startup với phần ổn định. Nếu token-slot/s tăng mà padding tăng, không kết luận token hữu ích/s tăng tương ứng. Nếu chọn test trên last, không gán số đó cho best.

Kết quả cần kèm run, stage, config đã resolve, lineage và điều kiện đo. Mở [Sổ tay Lưu trữ](../../state/docs/index.md) để hiểu artifact. Chuyển sang [profiling](profiling.md) khi cần giải thích từng con số.

## Khi `bench-step` được triển khai

Luồng dự kiến là chọn experiment/stage → nạp mốc tự động → chọn ứng viên → xem số lượng phép đo và tài nguyên → xác nhận chạy → đọc so sánh. Đây là mô tả hướng sử dụng, chưa phải cam kết API, tên field hoặc CLI. Coding agent phải dựa vào thiết kế tính năng và nguồn mới nhất khi triển khai.
