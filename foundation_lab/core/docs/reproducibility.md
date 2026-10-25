---
schema_version: 1
department: core
page_id: reproducibility
title: Mốc so sánh và khả năng tái lập
description: Biết cần giữ gì giống nhau và lưu gì để một kết quả có thể được giải thích hoặc chạy lại.
order: 60
kind: interpretation
updated: "2026-10-25"
tags: [baseline, mốc, tái lập, reproducibility, seed, lineage, config hash]
sources:
  - foundation_lab/core/config.py
  - foundation_lab/core/experiment.py
  - foundation_lab/core/configs/base.yaml
  - foundation_lab/core/tasks.py
  - foundation_lab/pipeline.py
  - foundation_lab/cli.py
---

## Baseline trả lời câu hỏi nào?

Baseline là cấu hình mốc để biết một thay đổi làm kết quả khác đi bao nhiêu. Trong Lab, mốc cần lấy từ cấu hình mặc định hiện tại của model/experiment đang dùng. Không viết tay một tập con số rồi coi nó vẫn là mặc định khi code đã thay đổi.

Ứng viên nên chỉ khác ở yếu tố muốn đo. Nếu thay cả kernel, dtype, chiều dài sequence và thiết bị cùng lúc, kết quả nhanh hơn không chỉ ra yếu tố nào gây ra khác biệt. Bạn vẫn có thể khảo sát nhiều yếu tố, nhưng cần đặt câu hỏi và ghi rõ thiết kế so sánh.

## Những thứ nên giữ giống khi so sánh

| Câu hỏi | Các yếu tố phải đối chiếu |
| --- | --- |
| Kernel nào nhanh hơn với cùng phép tính? | Shape, dtype, device, đầu vào, mask, warmup, số lần đo, forward/backward |
| Loss implementation có tương đương? | Hidden states, head weights, targets, loss weights, mẫu số và gradient |
| Checkpoint nào có loss tốt hơn? | Tokenizer, split đánh giá, masking/normalization, checkpoint được chọn và version |
| Decoder nào phù hợp hơn? | Cùng checkpoint, prompts, giới hạn sinh, seed/chiến lược sampling, cách tính thời gian |

Đây là tiêu chí diễn giải, không phải khẳng định mọi bàn chạy hiện có đo được toàn bộ tình huống. Xem sổ tay phòng tương ứng để biết phạm vi benchmark đã triển khai.

## Seed có tác dụng và giới hạn gì?

Seed khởi tạo các chuỗi ngẫu nhiên để giảm một nguồn biến thiên. Runtime chung hiện có `seed: null`, nghĩa là không chủ động đặt seed theo experiment đó. Khi seed có giá trị, pipeline dùng seed cộng chỉ số stage trong thứ tự pipeline; stage tắt vẫn ảnh hưởng vị trí chỉ số trong danh sách.

Giữ seed giống nhau không tự đảm bảo mọi thiết bị/kernel/phiên bản thư viện cho kết quả bitwise giống nhau. Cũng không sửa được rò rỉ dữ liệu hoặc phép đo thiếu warmup. Với sampling, một câu trả lời khác nhau không tự chứng minh mô hình thay đổi chất lượng.

## Config hash chứng minh điều gì?

Hash giúp nhận diện một mapping config đã resolve. Nó không chứa toàn bộ code, nội dung dataset hay môi trường. Hai run có cùng config hash vẫn có thể khác nếu những yếu tố ngoài mapping thay đổi.

Muốn hiểu run cần xem config cùng lineage và thông tin lần gọi. Trong cùng repo, working tree có thay đổi chưa commit cũng quan trọng: commit ID không bao trùm phần sửa chưa commit. Không nói “đã tái lập” chỉ vì cùng tên file YAML.

## Chạy lại và resume

Chạy lại task từ resolved config là thực hiện một phép đo/lượt suy luận mới và tạo kết quả mới. Resume train khôi phục trạng thái một run, có optimizer/RNG/sampler nếu checkpoint và luồng hỗ trợ. Init từ best chỉ mang trọng số làm điểm xuất phát.

Việc dùng lại run cũ phải theo chính sách phiên bản: đổi định dạng, loss/metric hoặc dữ liệu có thể làm kết quả cũ không còn so trực tiếp. Không chuyển/ghi đè run cũ để làm như cùng một thí nghiệm. Xem [Lưu trữ](../../state/docs/index.md) để phân biệt các loại file và phiên bản.

## Cách viết một kết luận hữu ích

Một kết luận tốt có đối tượng, điều kiện và giới hạn: “Ứng viên có sai khác số học trong ngưỡng đã chọn và thời gian trung vị thấp hơn mốc trên shape/dtype/device này.” Nó không tự kéo theo “mọi model sẽ nhanh hơn” hoặc “câu trả lời sẽ thông minh hơn”.

Với số liệu chất lượng, phải hỏi tập đánh giá có độc lập với train không. Lab hiện có giới hạn trộn/split được ghi trong [Dữ liệu](../../data/docs/index.md); đọc giới hạn đó trước khi xem một loss giảm là tiến bộ tổng quát hóa.
