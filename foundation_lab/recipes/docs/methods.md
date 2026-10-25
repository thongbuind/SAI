---
schema_version: 1
department: recipes
page_id: methods
title: Pretrain, continued pretrain và SFT
description: Phân biệt mục đích, dữ liệu, token chịu loss và lựa chọn recipe hiện có.
order: 10
kind: concept
updated: "2026-10-25"
tags: [causal_lm, pretrain, continued, SFT, phương pháp]
sources:
  - foundation_lab/recipes/__init__.py
  - foundation_lab/recipes/causal_lm.py
  - foundation_lab/recipes/sft.py
  - foundation_lab/recipes/base.py
  - foundation_lab/tokenizer/encode.py
  - foundation_lab/data/loaders.py
---

## Cùng dự đoán token tiếp theo, khác dữ liệu và phạm vi chấm điểm

Pretrain học từ văn bản, với nhãn token kế tiếp trên mọi vị trí thật. Continued pretrain tiếp tục mục tiêu đó từ trọng số đã học, thường nhằm bổ sung một miền dữ liệu. SFT dùng ví dụ hội thoại có câu trả lời mẫu để hướng model tạo dạng phản hồi mong muốn. Việc học từ các câu trả lời minh hoạ là bước supervised fine-tuning trong [Ouyang và cộng sự, 2022](https://arxiv.org/abs/2203.02155); điều này không hàm ý SAI triển khai toàn bộ pipeline RLHF của nghiên cứu.

| Thuộc tính | `causal_lm` | `sft` |
| --- | --- | --- |
| Capability model yêu cầu | `causal_lm` | `causal_lm` |
| Data kind nhận | `lm` | `sft` |
| Vị trí chịu loss | Mọi target thật, xác định theo độ dài | Mask do encoder tạo: segment model và EOS |
| Batch train | Sortish theo số mẫu | Sortish theo mục tiêu token chịu loss |
| Mẫu số gradient | Trung bình độ dài train × batch_size | `target_loss_tokens` của sampler |
| Tham số optimizer | Toàn bộ tham số model | Tham số còn trainable sau freeze |
| Chọn trọng số test mặc định của recipe | `last` | `best` |

Trong cả hai, objective train mặc định là `lm_ce`; các giá trị ghi trong experiment có thể override mặc định recipe. Không nhầm nhãn đã shift với việc objective tự tạo nhãn; X/Y đã chuẩn bị trước khi recipe chạy.

## Khi nào một stage là continued pretrain?

Đó là lựa chọn dữ liệu, `init` và params của stage dùng `causal_lm`, không phải một tên recipe riêng. Nếu không đặt `init`, pipeline dựng model mới; tên stage có chữ continued không tự làm model nạp trọng số trước đó.

Dữ liệu mới có thể giúp model tiếp xúc thêm miền mong muốn, nhưng nếu chỉ tăng dữ liệu một miền hoặc train quá mức, hành vi ở miền khác cũng có thể đổi. Muốn đánh giá tác động cần baseline và bộ đánh giá phù hợp, không suy chỉ từ training loss.

## SFT1 và SFT2 khác nhau ở đâu?

Cả hai gọi cùng `SFTRecipe`. Config chọn nguồn dữ liệu, init, freeze, learning rate, epoch và cơ chế resample. Khác tên stage giúp ghi lineage và tổ chức pipeline; không đổi hợp đồng X/Y/mask.

SFT không đảm bảo model đúng sự thật hoặc làm tốt mọi yêu cầu. Nó học xác suất từ câu trả lời mục tiêu và ngữ cảnh đã thấy; chất lượng, độ đa dạng và tính nhất quán của câu trả lời mẫu vẫn quyết định điều được khuyến khích học.

## Những option không có trong registry hiện tại

Distillation, DPO và RLHF chưa là recipe chạy được trong package này. Không đặt các tên đó vào YAML rồi mong config tự suy thuật toán. Biến thể recipe mới phải đăng ký với `config_cls`, `requires`, `data_kinds` và `run()` đúng hợp đồng; việc đó thuộc một lần triển khai được duyệt riêng.

Tiếp theo: [Stage, init và resume](stages-init-resume.md).
