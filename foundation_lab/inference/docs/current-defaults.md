---
schema_version: 1
department: inference
page_id: current-defaults
title: Mốc SAI và cấu hình khởi đầu hiện tại
description: Snapshot GenerationConfig cùng khác biệt giữa mốc backend, form web và spec mẫu CLI.
order: 40
kind: reference
updated: "2026-10-25"
tags: [sai, mặc định, baseline, config]
sources:
  - SAI/generate.py
  - SAI/version.py
  - foundation_lab/inference/bench.py
  - foundation_lab/inference/configs/compare_375M.yaml
  - foundation_lab/inference/configs/benchmark.yaml
  - web/site/src/data/lab_fields.json
  - web/site/src/lib/lab/specForm.ts
---

## Mốc phải lấy từ code hiện tại

Ngày 25/10/2026, bản phát hành model trong working tree là `dev3.0.1`; component generate vẫn có version riêng `dev3.0.0`. Mốc của phòng infer không được dựng bằng cách suy từ số version: backend kiểm tra decoder là `sai` và `generation == GenerationConfig()` tại lúc chạy.

Cách khai báo bền nhất trong spec:

```yaml
variants:
  - name: default
```

Không ghi lại toàn bộ giá trị trong bài này vào baseline. Khi mặc định code đổi, biến thể không ghi `generation` tiếp tục bám mặc định đó. Trong một run đã lưu, xem config và lineage để biết chính xác thời điểm/nguồn chạy; không dùng snapshot sổ tay để sửa run cũ.

## GenerationConfig tại thời điểm soạn

| Field | Default snapshot |
| --- | --- |
| `strategy` | `auto` |
| `max_new_tokens` | 500 |
| `beam_size` | 5 |
| `stable_steps` | 8 |
| `temperature` | 0,8 |
| `top_k` | 0 |
| `top_p` | 0,95 |
| `repetition_penalty` | 1,2 |
| `no_repeat_ngram_size` | 3 |
| `length_penalty` | 1,0 |
| `lowercase` | `true` |
| `seed` | `null` |
| `stop_sequences` | Danh sách rỗng |

Trong phòng infer, auto đi nhánh beam; các giá trị sampling nằm trong config nhưng không điều khiển nhánh beam. `strategy: beam, beam_size: 5` có thể cho đường sinh giống auto trong ngữ cảnh này, nhưng không bằng nguyên `GenerationConfig()` vì field strategy đã khác, nên không được nhận làm mốc duy nhất.

## Form web mở sẵn gì?

`lab_fields.json` có gợi ý khởi đầu model `SAI_375M`, checkpoint `model/sft1_375M.pt`, và hai biến thể:

1. `default`: giữ mặc định.
2. `sample`: strategy sample, temperature 0,8, top-p 0,95, seed 0; các field còn lại lấy default.

Đây là gợi ý UI, không là bằng chứng file trọng số tồn tại. Danh sách model/checkpoint thực tế đến từ metadata backend. Form hiện giới hạn tối đa 4 biến thể, gắn nhãn mốc theo cấu hình và không cho xóa mốc cuối cùng; nếu sửa hết các mốc thành ứng viên thì form/backend từ chối spec.

Một số giới hạn hiển thị: beam slider 1–10, temperature 0,05–2, top-p 0,05–1, max token nhập trên web tối đa 1.024. Không lấy chúng làm toàn bộ giới hạn của validator CLI. `stop_sequences` có trong schema nhưng bị ẩn ở web.

## Spec mẫu CLI và benchmark

`compare_375M.yaml` kế thừa `benchmark.yaml`, chọn model/checkpoint SFT1 375M và ba biến thể default, greedy, sample. Vì thế số lượt sinh của spec mẫu khác form web ban đầu nếu không chỉnh biến thể.

Benchmark snapshot có 10 câu hỏi ngắn về chào hỏi, tri thức, giải thích và sáng tác. Bộ này là danh sách prompt, không kèm đáp án chuẩn hay rubric chấm tính đúng. Nội dung có thể được người dùng thay bằng “Lưu vào benchmark”; không hardcode số câu 10 vào UI hoặc tổng workload.

Đọc [quy trình](workflow.md) để kiểm tra số lượt thực tế, và [đọc kết quả](interpreting-results.md) để hiểu giới hạn metric.
