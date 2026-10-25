---
schema_version: 1
department: core
page_id: concepts
title: Experiment, pipeline, stage và tác vụ
description: Phân biệt các đơn vị làm việc của Lab để chọn đúng cấu hình và hiểu kết quả thuộc về đâu.
order: 10
kind: concept
updated: "2026-10-25"
tags: [experiment, pipeline, stage, task, run, spec, YAML]
sources:
  - foundation_lab/core/experiment.py
  - foundation_lab/core/tasks.py
  - foundation_lab/pipeline.py
  - foundation_lab/cli.py
---

## Experiment là gì?

Experiment là bản mô tả một lượt thí nghiệm huấn luyện. Nó có tên, model, runtime và pipeline. Tên là nhãn tổ chức run; model xác định cấu trúc; runtime xác định môi trường thực thi; pipeline cho biết các stage theo thứ tự.

Một experiment không phải checkpoint. Cùng experiment có thể chạy nhiều lần với trọng số ngẫu nhiên hoặc môi trường khác nhau; mỗi lần cần một run để lưu dấu vết. File cấu hình không chứa toàn bộ dữ liệu hay trọng số.

## Stage là gì?

Stage là một giai đoạn huấn luyện có recipe, nguồn dữ liệu, tham số và cách khởi tạo. Ví dụ pretrain học dự đoán token trên văn bản; SFT học cách trả lời từ hội thoại có nhãn. Tên `sft1` và `sft2` là tên stage trong pipeline, không phải hai thuật toán mới: cả hai có thể dùng cùng recipe `sft` với dữ liệu và tham số khác nhau.

| Thành phần stage | Câu hỏi nó trả lời |
| --- | --- |
| recipe | Học theo quy trình nào? |
| data | Đọc nguồn gì, trộn/chia theo quy tắc nào? |
| params | Epoch, learning rate, batch, objective và lựa chọn của recipe là gì? |
| init | Khởi tạo mới hay lấy trọng số từ đâu? |
| enabled | Stage được chọn mặc định khi chạy toàn pipeline hay không? |

Stage sau không đơn giản tiếp tục cùng một optimizer của stage trước. Pipeline dựng model cho từng stage, rồi nạp trọng số theo `init` hoặc checkpoint resume. Quy trình độc lập này cho phép thay tham số và nguồn dữ liệu giữa các giai đoạn. Đọc [Phương pháp huấn luyện](../../recipes/docs/index.md) trước khi thay pipeline.

## Pipeline là gì?

Pipeline là mapping tên stage → cấu hình, được đi theo thứ tự khai báo. Tham chiếu `pretrain.best` nghĩa là trọng số best của stage `pretrain` trong cùng run. Nó phải trỏ tới stage đứng trước, nhưng việc cấu hình hợp lệ chưa đảm bảo file best đã được tạo.

`enabled: false` bỏ stage trong lượt chạy mặc định. Việc chỉ định rõ `--stages` chọn theo tên, còn thứ tự thực thi vẫn theo pipeline. Trong implementation hiện tại, danh sách `--stages` có thể chọn một stage vốn mặc định tắt. Vì vậy đọc lại phạm vi lệnh trước khi chạy, đừng chỉ nhìn cờ enabled trong file.

## Task khác experiment thế nào?

Task là tác vụ độc lập nhận một spec, kiểm tra, chạy và ghi kết quả. Hiện `compare-kernel`, `compare-objective`, `infer` theo hợp đồng này. Mỗi task sở hữu schema riêng; không đưa `pipeline` vào một infer spec nếu schema không có field đó.

| Thuộc tính | Experiment huấn luyện | Task độc lập |
| --- | --- | --- |
| Điều phối | Chuỗi stage trong pipeline | Một tác vụ được đăng ký |
| Cấu hình | ExperimentSpec và recipe params | Dataclass spec của task |
| Ví dụ đầu ra | Checkpoint, loss theo stage | Record so sánh và metrics |
| Thư mục run | Theo tên experiment | Theo tên lệnh |

`spec` là từ chung cho mô tả đầu vào có schema; `config` là dữ liệu cấu hình. Chúng không có nghĩa mọi lệnh nhận cùng một dạng file.

## Run là gì?

Run là một lần thực thi có thư mục riêng, resolved config, lineage và kết quả. Tên experiment trả lời “thử cái gì”, run ID trả lời “lần nào”. Không nhận diện hai run chỉ bằng tên model vì cùng model có thể khác tokenizer, dữ liệu, mục tiêu, seed hoặc decoder.

Task chạy lại bằng resolved config thường tạo run mới. Resume train dùng trạng thái run cũ. Hai hành động có mục đích và ràng buộc khác nhau; xem [Sổ tay Lưu trữ](../../state/docs/index.md).

## YAML cần hiểu tới mức nào?

YAML mô tả mapping bằng `key: value`, danh sách bằng `[...]` hoặc từng dòng `-`, cấu trúc lồng qua thụt lề. `true/false` là boolean; `null` là không có giá trị, không phải số 0. Chuỗi tên như `SAI_35M` chỉ được thay bằng config ở những field resolver hỗ trợ. Dòng có `#` là comment, không phải tham số.

Bạn không cần hiểu mọi file Python để dùng Lab, nhưng cần biết một giá trị đang ở cấp runtime, model hay params của stage để override đúng chỗ.
