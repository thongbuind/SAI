---
schema_version: 1
department: objectives
page_id: loss-contract
title: "Cross-entropy và hợp đồng objective"
description: "Hiểu phép tính next-token loss và trách nhiệm của hàm objective trong kiến trúc Lab."
order: 10
kind: concept
updated: "2026-10-25"
tags: [cross-entropy, gradient, logits, registry, hợp đồng]
sources:
  - foundation_lab/objectives/registry.py
  - foundation_lab/objectives/lm_ce.py
  - SAI/modeling.py
  - foundation_lab/recipes/base.py
  - tests/test_objectives.py
---

## Từ hidden state đến một con số loss

Sau các decoder block và norm cuối, model cho một vector hidden tại mỗi vị trí. `lm_head` chiếu vector này lên từ vựng thành logits. Cross-entropy so phân phối dự đoán với ID token đúng của vị trí đó.

Viết gọn cho token `i`: `CE_i = -log p(target_i | các token trước đó)`. Nếu model gán xác suất cao hơn cho token đúng, loss nhỏ hơn. Ví dụ minh hoạ: xác suất `0.5` có loss khoảng `0.693`; xác suất `0.1` có loss khoảng `2.303`, dùng log tự nhiên. Đây là ví dụ toán học, không phải số đo SAI.

Không đưa softmax thủ công trước `cross_entropy`: implementation nhận logits và tự xử lý phép tính cần thiết. Cross-entropy chuẩn không phải độ chính xác top-1: hai dự đoán cùng chọn đúng token vẫn có loss khác vì mức tin tưởng khác. [Tài liệu CrossEntropyLoss của PyTorch](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html).

## Hợp đồng hiện tại

| Đầu vào/ra | Shape, kiểu và ý nghĩa |
| --- | --- |
| `hidden` | `(N,d_model)`, hidden đã flatten từ batch |
| `head_weight` / tham số `weight` của `lm_ce` | `(V,d_model)`, trọng số `lm_head` |
| `targets` | `(N,)`, số nguyên `long`, nhãn đã shift ở lớp dữ liệu |
| `token_weight` | `(N,)`, số thực; trọng số loss của từng vị trí |
| `chunk_size` | Số hàng hidden xử lý trong một lần tạo logits |
| Kết quả | Scalar float32, tổng `sum(token_weight_i × CE_i)` |

`N` có thể gồm cả hàng padding có trọng số bằng không. Objective không nhận câu văn, không tokenize, không shift target lại, không chọn mẫu số và không gọi optimizer. Các tensor phải khớp số hàng, chiều head và thiết bị của phép tính. Target phải theo hợp đồng chỉ số từ vựng của nguồn dữ liệu; trọng số 0 không phải một cơ chế tự sửa target sai.

## Vì sao trả tổng thay vì trung bình?

Engine có thể chia một logical batch thành nhiều microbatch. Nếu objective tự chia theo số token từng microbatch, rồi engine lại cộng/chia, các nhóm có độ dài khác nhau dễ bị đặt trọng số sai. Trả tổng cho phép recipe/engine áp một mẫu số đã xác định nhất quán.

Objective là một hàm thuần theo nghĩa trách nhiệm: không đọc checkpoint, không sửa config, không lưu kết quả ra đĩa. Hàm trả tensor còn đồ thị gradient để bên gọi backward. Gradient theo hidden và head weight đều quan trọng; kiểm tra chỉ giá trị loss chưa đủ phát hiện mọi lỗi đạo hàm.

## Những lựa chọn để học thêm

`lm_ce` là lựa chọn thực thi duy nhất trong snapshot 25/10/2026. Các ý tưởng như label smoothing, thay trọng số token, distillation hoặc objective theo cặp ưu tiên thay đổi mục tiêu hoặc dữ liệu cần dùng. Chúng không phải menu hiện có. Một objective mới phải có hợp đồng rõ, đăng ký đúng registry và test riêng trước khi xuất hiện trong UI.

Đặc biệt, objective cần đầu vào ngoài hidden/head/target/weight có thể không phù hợp chữ ký hiện tại; không giấu dữ liệu phụ trong trạng thái toàn cục để vượt qua hợp đồng. Xem [mask và mẫu số](mask-normalization.md) để hiểu ranh giới với recipe.
