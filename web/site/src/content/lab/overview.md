---
title: Giới thiệu
description: Foundation Lab là nơi SAI được huấn luyện, thí nghiệm và chuẩn bị phát hành. Mọi thứ, từ tokenizer tới checkpoint trên Hugging Face, đi qua lab này.
group: Tổng quan
order: 1
---

## Foundation Lab là gì

SAI có hai phần tách biệt:

- **`SAI/`**: mô hình chính thức, thứ được phát hành. Package này chỉ phụ thuộc PyTorch và không biết gì về lab.
- **`foundation_lab/`**: lab huấn luyện và thí nghiệm. Lab dùng `SAI/`, chiều ngược lại thì không.

Một biến thể kiến trúc chỉ được đưa vào `SAI/` sau khi đã thử trong lab. Nhờ vậy model phát hành luôn gọn và ổn định, còn lab thì thoải mái thử nghiệm.

## Điểm nổi bật

- **Tổ chức như phòng ban.** 10 phòng, mỗi phòng sở hữu một phần việc và một hợp đồng giao diện rõ ràng. Thêm biến thể chỉ cần đăng ký vào registry của phòng đó, không sửa code phòng khác.
- **Một file mô tả cả lượt chạy.** Model, dữ liệu, runtime và mọi stage nằm trong một file YAML. Lab kiểm tra toàn bộ trước khi nạp dữ liệu hay dựng model, nên lỗi gõ sai khoá hiện ra sau một giây chứ không phải sau một giờ train.
- **Truy vết được mọi kết quả.** Mỗi run ghi lại config đã resolve, commit git, môi trường, dấu vân tay tokenizer và dữ liệu, cùng phiên bản lab.
- **Tái lập tới từng trọng số.** Khi chuyển từ pipeline huấn luyện cũ sang lab, một test đối chiếu chạy cả hai trên cùng dữ liệu và seed: trọng số mỗi stage lệch không quá `1e-6`, test loss không quá `1e-4`.
- **Chạy từ laptop tới GPU.** Tự chọn CUDA, Apple MPS hoặc CPU; mixed precision, `torch.compile`, và batch tự chia thành microbatch theo giới hạn số token để vừa bộ nhớ.
- **Đi tới tận bản phát hành.** Từ thư mục run ra repo Hugging Face: export, verify như người dùng tải về, rồi push kèm tag phiên bản.

## Một vài con số

| Hạng mục | Số lượng |
| --- | --- |
| Phòng ban | 10 |
| Registry | 11 |
| Lệnh CLI | 8 |
| Cỡ mô hình đã phát hành | 3 (35M, 100M, 375M) |
| Thư viện cần cho lab | 4 (PyTorch, NumPy, PyYAML, SentencePiece) |

## Đọc tiếp

- [Chức năng](../features/): lab làm được những gì.
- [Cấu tạo](../structure/): các phòng ban và cách một experiment đi qua chúng.
- [Quickstart](../quickstart/): chạy experiment đầu tiên.
