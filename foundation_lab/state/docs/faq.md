---
schema_version: 1
department: state
page_id: faq
title: Câu hỏi thường gặp về run và checkpoint
description: Giải thích thiếu artifact, checkpoint không xuất hiện, resume và giới hạn kết luận từ metadata.
order: 70
kind: faq
updated: "2026-10-25"
tags: [faq, run, checkpoint, resume]
sources:
  - foundation_lab/state/run.py
  - foundation_lab/state/checkpoint.py
  - foundation_lab/state/lineage.py
  - foundation_lab/pipeline.py
  - web/backend/lab_runs.py
---

## Có best.pt nhưng không có done.json, stage đã xong chưa?

Chưa đủ bằng chứng. Best được ghi khi val cải thiện, còn `done.json` chỉ được pipeline ghi sau khi recipe hoàn tất. Stage có thể đang chạy, bị ngắt hoặc chưa xong test cuối. Xem log, checkpoint và trạng thái tiến trình.

## Có last.ckpt.pt thì mọi bước trước lúc ngắt đều giữ lại?

Chỉ giữ đến điểm checkpoint gần nhất. Các bước chạy sau đó nhưng chưa được lưu không nằm trong file. Khi resume, chúng phải chạy lại. Xem epoch/global step/batches done thay vì đoán từ thời gian sửa tên file.

## Có thể resume bằng best.pt không?

Best thuần không có optimizer, scheduler và vị trí huấn luyện. Dùng nó để init run mới là mục tiêu khác. Resume dùng thư mục run với checkpoint stage. Đọc [init và resume](init-and-resume.md).

## Vì sao đổi init mà resume dường như không dùng nó?

Checkpoint của chính stage được ưu tiên khi resume, và stage có done được bỏ qua. Init chỉ được dùng khi pipeline đi vào nhánh khởi tạo stage. Muốn bắt đầu thí nghiệm khác từ một trọng số, tạo run mới với cấu hình/init rõ ràng.

## Vì sao một checkpoint có trên đĩa mà không hiện trong form?

Catalog dùng quy tắc tên/path và config model. File tên khác quy tắc, `last.ckpt.pt`, best từ model đã override khác catalog hoặc file ở ngoài mẫu dò có thể không được gán vào model trong form. Không sửa tên hoặc config run cũ chỉ để ép nó hiện; trước hết xác định tính tương thích và cách chỉ định đường dẫn phù hợp.

## Không có result.json có nghĩa tác vụ chưa chạy gì?

Không. Nó có thể đã ghi nhiều event và record nhưng chưa hoàn tất. Khi server không còn job, backend gọi tình trạng chưa có kết quả hoặc failed event cuối là `incomplete`. Không tự biến các ô chưa có record thành số 0.

## Cùng config_hash có bảo đảm cùng trọng số đầu ra?

Không. Hash config không bao hàm toàn bộ code chưa commit, nội dung file dữ liệu lớn, môi trường hay RNG. Đọc [lineage](lineage.md) và so phạm vi thí nghiệm trước khi kết luận tái lập.

## Dòng dirty=true có nghĩa run không dùng được?

Nó vẫn là kết quả đã chạy, nhưng commit hash không đủ chứa toàn bộ trạng thái nguồn của run. Cần giữ và ghi lại phần thay đổi đã dùng nếu muốn tái hiện. Không tự bỏ run hoặc sửa lineage để xóa cờ này.

## Sổ tay có mở trực tiếp checkpoint để kiểm tra không?

Không. Sổ tay là nội dung tĩnh; trang kho run dự kiến cũng nên đọc metadata trước. Việc nạp tensor chỉ khi người dùng chủ động thực hiện tác vụ phù hợp. Không cần train hoặc tải model để xem định nghĩa `best`, `resume` hay `lineage`.
