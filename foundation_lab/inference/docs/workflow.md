---
schema_version: 1
department: inference
page_id: workflow
title: Sử dụng phòng Suy luận trên web và CLI
description: Tạo phép so sánh có mốc, giữ bộ câu hỏi, chạy có chủ đích và đọc lịch sử kết quả.
order: 50
kind: guide
updated: "2026-10-25"
tags: [hướng dẫn, web, cli, benchmark]
sources:
  - web/site/src/components/lab/TaskBench.astro
  - web/site/src/lib/lab/views/infer.ts
  - web/site/src/data/lab_fields.json
  - web/site/src/lib/lab/specForm.ts
  - foundation_lab/inference/bench.py
  - foundation_lab/inference/configs/compare_375M.yaml
  - foundation_lab/cli.py
---

## Bước 1: chọn model và trọng số

Trong Control center, chọn model rồi checkpoint tương ứng. Model xác định cấu trúc; checkpoint xác định trạng thái đã học. Muốn so decoder, giữ cùng checkpoint cho mọi biến thể. Nếu đổi cả model lẫn decoder giữa các run, cần ghi rõ đây không còn là phép thử một biến.

Nếu checkpoint không hiện, đọc [FAQ](faq.md) trước. Tên file không phải bằng chứng model tương thích hoặc máy đủ RAM.

## Bước 2: chuẩn bị câu hỏi

Ô Câu hỏi mở sẵn benchmark của lab; mỗi dòng là một câu hỏi độc lập. Chọn một bộ cố định phù hợp mục tiêu: câu ngắn, yêu cầu dài, tên riêng, nội dung dễ lặp và kiểu tác vụ bạn quan tâm. Không thay prompt giữa các biến thể trong cùng so sánh.

Sửa ô prompt chỉ đổi cấu hình đang chuẩn bị. “Lưu vào benchmark” ghi lại danh sách vào file benchmark của lab cho lần sau; nó là thao tác lưu nội dung riêng, không phải bắt đầu infer và không chỉ là lưu riêng cho run. Không cần bấm lưu để thử một bộ câu hỏi tạm trong form.

## Bước 3: giữ mốc và thêm ứng viên

Giữ một biến thể `sai` có mọi field GenerationConfig ở mặc định. Đặt tên duy nhất cho từng biến thể để đọc bảng dễ hơn, ví dụ `default`, `greedy`, `sample_seed0`. Tên mốc không bắt buộc là `default`; các giá trị cấu hình mới quyết định.

Mở cấu hình sinh, chọn strategy; nhóm sampling hoặc beam chỉ hiện các field tương ứng, còn nhóm Nâng cao chứa token budget, seed, chống lặp và các field ít dùng. Không tạo một candidate đổi nhiều tham số mà không ghi mục đích.

Để thử một thay đổi so với sampling, có thể giữ default, thêm sampling tham chiếu rồi thêm sampling chỉ đổi một field. Baseline phát hành vẫn cần tồn tại dù câu hỏi riêng của bạn là so hai cấu hình sample.

## Bước 4: đọc workload và xác nhận

Số lượt sinh bằng `số câu hỏi × số biến thể`. Mỗi lượt có thể chạy đến giới hạn token; workload này không phải dự đoán chính xác giây chạy hay RAM. Bấm Chạy mở hộp xác nhận model, checkpoint và số lượt; backend xếp hàng, mỗi lúc chạy một lệnh.

Trên máy 8 GB, không tự suy rằng config 375M/beam mặc định là nhẹ. Việc xem sổ tay hay soạn spec không thực thi; chạy inference là thao tác nạp model thật và cần quyền tương ứng trong workspace.

## Bước 5: theo dõi và đọc

Lưới đi theo hàng câu hỏi, cột biến thể. Backend xử lý mọi biến thể của câu thứ nhất rồi mới chuyển câu sau; mỗi record điền một ô. “Đang chờ” nghĩa là chưa có record, không phải thất bại. Bốn biến thể làm lưới cuộn ngang để giữ cột đủ rộng.

Khi xong, bảng metric tổng hợp xuất hiện. Chọn run cũ từ lịch sử để đọc lại; dòng “Chạy lại từ dòng lệnh” dùng spec đã lưu. Nếu hủy, các ô hoàn tất có thể còn đọc được nhưng chưa có tổng hợp cuối. Đọc [cách diễn giải](interpreting-results.md).

## CLI hiện có

Lệnh minh hoạ sau chưa được chạy trong phiên soạn sổ tay:

```bash
python -m foundation_lab infer foundation_lab/inference/configs/compare_375M.yaml
```

Spec mẫu có checkpoint thật cần tồn tại và nạp vào RAM. Có thể override device qua `--set device=cpu`, nhưng CPU không đồng nghĩa dùng ít RAM hoặc hoàn thành nhanh.

Để chạy lại một cấu hình đã lưu, tạo run mới từ file của run cũ:

```bash
python -m foundation_lab infer runs/infer/<run_id>/resolved_config.yaml
```

`<run_id>` phải thay bằng run thực tế. Đây là chạy lại tác vụ, không phải resume quá trình infer dang dở. `run --resume` thuộc pipeline huấn luyện. Spec tác vụ nhận YAML/JSON, stdin `-` hoặc các override theo CLI chung; các variant phải giữ mốc và tên duy nhất.
