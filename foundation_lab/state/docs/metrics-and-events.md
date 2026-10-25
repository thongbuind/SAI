---
schema_version: 1
department: state
page_id: metrics-and-events
title: Metrics, events và kết quả tác vụ
description: Đọc đúng dữ liệu ghi dần, dữ liệu hoàn chỉnh và chiều tốt hơn của từng metric.
order: 50
kind: reference
updated: "2026-10-25"
tags: [metrics, events, jsonl, result]
sources:
  - foundation_lab/state/metrics.py
  - foundation_lab/state/events.py
  - foundation_lab/state/results.py
  - foundation_lab/engine/loop.py
  - foundation_lab/recipes/base.py
  - web/backend/lab_runs.py
---

## Ba artifact phục vụ ba mục đích

| Artifact | Dạng | Mục đích |
| --- | --- | --- |
| `metrics.jsonl` | Mỗi dòng một metric record | Chuỗi theo dõi train/val/test |
| `events.jsonl` | Mỗi dòng một event | Tiến độ và kết quả từng phần của tác vụ |
| `result.json` | Một object hoàn chỉnh | Tổng hợp cuối của tác vụ |

JSONL cho phép thêm bản ghi mà không viết lại toàn bộ danh sách. Trong lúc tiến trình ghi, dòng cuối có thể chưa kết thúc. Backend theo dõi event chỉ nhận các dòng đã có ký tự xuống dòng; phần còn lại đợi lần đọc tiếp theo. Không suy ra file hỏng chỉ vì đang đọc đúng lúc ghi.

## Metric huấn luyện

Record train thường có `stage`, `split`, `epoch`, `batch`, `optimizer_step`, `loss`, `lr`, thêm `time`. Val có `train_loss` tổng hợp đi kèm `loss` val. Test có `weights: best|last` để cho biết loại trọng số được đánh giá.

Không phải tất cả record đều có cùng bộ field. Không ghép loss của các stage khác objective/data thành một đường liên tục rồi gọi đó là tiến bộ của cùng bài toán. Sau resume giữa epoch, trung bình train log được tính lại trên phần batch tiếp tục, nên nhãn epoch giống nhau chưa chắc có cùng cửa sổ tổng hợp.

## Event của tác vụ

| `kind` | Dữ liệu chính | Ý nghĩa |
| --- | --- | --- |
| `started` | `total` | Tác vụ bắt đầu |
| `progress` | `done`, `total`, có thể `message` | Mức tiến độ đã hoàn tất |
| `record` | `record` | Một đơn vị kết quả đã có |
| `finished` | Không bắt buộc dữ liệu thêm | Đã ghi kết quả cuối |
| `failed` | `error` | Tác vụ ghi nhận lỗi |

Event có `seq` và thời gian. Thứ tự `seq` giúp theo dõi luồng trong một lượt tác vụ; không dùng nó làm định danh toàn cục giữa các run. Bản ghi `record` cũng được đưa vào `result.records` khi hoàn tất. Người đọc không cộng cả event record lẫn result record thành hai lần đo khác nhau.

## RunResult phiên bản 1

Các khóa hiện có: `command`, `metrics`, `records`, `timings`, `baseline`, `format_version`.

Mỗi metric có `group`, `name`, `value`, `unit`, `better`. `better` nhận `lower`, `higher`, `none`; `none` nghĩa là không gán hướng tốt hơn, không phải metric thiếu chất lượng. UI cần tôn trọng hướng do dữ liệu cung cấp thay vì mặc định số lớn tô xanh.

`baseline` là tên group mốc nếu mốc nằm ở cấp nhóm, chẳng hạn một decoder variant trong infer. Nó có thể `null` nếu mốc nằm trong từng record như một số tác vụ kernel. Không chọn dòng đầu bảng làm mốc chỉ vì trường này null.

`timings` dùng mili giây, tên pha tùy command. `records` cũng tùy tác vụ, nên cần view theo `command`. Đổi ý nghĩa hoặc xóa khóa của format chung yêu cầu tăng format version; backend hiện từ chối result có version khác version nó hiểu.

## Đọc kết quả dở dang

Record đã ghi cho biết việc nào hoàn tất; các ô chưa có record không phải kết quả bằng 0. Tác vụ bị hủy có thể còn record hữu ích, nhưng chưa có metric tổng hợp đầy đủ. Xem [Sổ tay Suy luận](../../inference/docs/index.md) để hiểu riêng latency, tokens/s và distinct của infer.
