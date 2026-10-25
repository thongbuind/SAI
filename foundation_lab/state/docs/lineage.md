---
schema_version: 1
department: state
page_id: lineage
title: Lineage và khả năng tái lập
description: Hiểu bằng chứng nguồn gốc trong lineage, giới hạn của fingerprint và vai trò của invocation.
order: 40
kind: concept
updated: "2026-10-25"
tags: [lineage, reproducibility, fingerprint, config]
sources:
  - foundation_lab/state/lineage.py
  - foundation_lab/state/run.py
  - foundation_lab/cli.py
  - foundation_lab/core/experiment.py
  - foundation_lab/core/seed.py
---

## Lineage trả lời câu hỏi nào?

Lineage là nguồn gốc của một run: nó nối kết quả với code, cấu hình, đầu vào và môi trường tại lúc tạo run. Không cần mở trọng số để đọc file JSON này.

| Field | Điều được ghi |
| --- | --- |
| `created_at` | Thời điểm tạo |
| `config_hash` | Hash của mapping cấu hình đã resolve |
| `lab_version` | Phiên bản nội bộ lab |
| `git` | Commit, branch, cờ dirty |
| `env` | Python, hệ điều hành, PyTorch, CUDA và tên GPU |
| `tokenizer` | Fingerprint tokenizer chính thức |
| `data` | Fingerprint các đầu vào theo nhãn/stage |

Trong tác vụ infer, đầu vào checkpoint được ghi dưới nhãn `checkpoint` của `data`. Tên field `data` ở đây chỉ tập đầu vào của tác vụ, không nhất thiết là dataset huấn luyện.

## Config đã resolve là gì?

Các tham chiếu `extends`, tên model và nguồn data được biến thành nội dung cụ thể; override và stage defaults được gộp theo contract của resolver. File này giúp loại bớt phụ thuộc vào cấu hình tên ngắn có thể bị sửa sau đó.

Tự đủ về cấu hình không nghĩa là đã chứa code, dữ liệu hay tensor checkpoint. Để tái hiện, vẫn cần đúng file đầu vào và môi trường. Một `resolved_config.yaml` cũ không biến checkpoint tương thích với code mới một cách tự động.

## Fingerprint rẻ và giới hạn của nó

`file_fingerprint` ghi đường dẫn, tồn tại hay không, kích thước byte và mtime. Chỉ file không quá 16 MiB có thêm SHA-256 nội dung. Vì vậy một checkpoint hoặc dataset lớn có thể chỉ được nhận diện bằng metadata rẻ, không có hash nội dung.

Không diễn giải kích thước + mtime thành chứng minh nội dung giống hệt. Nếu cần chứng minh mạnh hơn, phải có quy trình kiểm chứng riêng được duyệt; mở sổ tay không được tự băm dữ liệu lớn. Trường `exists:false` là thông tin file không tồn tại lúc thu fingerprint, không phải fingerprint rỗng hợp lệ.

## Lineage ban đầu và lịch sử gọi

`lineage.json` ghi lúc tạo run. Các lần resume không thay nó thành môi trường mới; `invocations.jsonl` thêm dòng gồm thời gian, argv, stages, init override, resume, config hash và git hiện tại. Muốn biết code của lần resume, đọc invocation tương ứng.

`git.dirty=true` nghĩa là working tree có thay đổi chưa commit. Commit hash khi đó không chứa đầy đủ trạng thái code thực tế đã dùng. Lineage không lưu nguyên diff và không là một bản backup repo. Không tự khẳng định “checkout commit là đủ” khi thiếu phần thay đổi chưa commit.

## Seed là một phần, không phải toàn bộ

Pipeline có thể đặt seed cho Python, NumPy và PyTorch; checkpoint hiện lưu RNG PyTorch theo thiết bị. Việc phục hồi run còn phụ thuộc sampler, implementation, thư viện và phần cứng. PyTorch không bảo đảm tái lập hoàn toàn giữa mọi release, nền tảng hoặc CPU/GPU. [Giới hạn tái lập của PyTorch](https://docs.pytorch.org/docs/stable/notes/randomness.html).

Một kết luận dễ kiểm tra nên ghi rõ “cùng config, checkpoint, dữ liệu và môi trường trong phạm vi thử nghiệm này”, rồi nêu phạm vi sai số chấp nhận. Đọc [phiên bản](versions.md) để quyết định run cũ được dùng làm mốc hay chỉ tham khảo.
