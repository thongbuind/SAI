---
title: Khái niệm
description: Các thuật ngữ dùng trong lab và trong phần còn lại của tài liệu.
group: Bắt đầu
order: 10
---

## Tổ chức

| Thuật ngữ | Nghĩa |
| --- | --- |
| **Phòng ban** | Một package con của `foundation_lab/`, sở hữu một phần việc (kiến trúc, dữ liệu, huấn luyện...). Xem [Cấu tạo](../structure/#các-phòng-ban). |
| **Registry** | Bảng tra từ tên sang implementation, do một phòng sở hữu. Config chọn implementation bằng tên. Xem [Mở rộng lab](../extending/#registry). |
| **Hợp đồng** | Quy ước giao diện mà mọi implementation trong một registry phải theo (shape tensor, tham số, giá trị trả về), ghi trong docstring của registry. |

## Mô hình

| Thuật ngữ | Nghĩa |
| --- | --- |
| **Model chính thức** | Model trong package `SAI/`, thứ được phát hành lên Hugging Face. Config ở `config/models/`. |
| **Họ mô hình** | Một kiểu model đăng ký trong `MODELS`: `SAI` là model chính thức, `SAI_modular` là bản lắp ghép để thí nghiệm. |
| **Capability** | Khả năng mà một họ mô hình có (vd. `causal_lm`). Recipe khai báo capability nó cần; lab báo lỗi nếu họ mô hình không có. |

## Huấn luyện

| Thuật ngữ | Nghĩa |
| --- | --- |
| **Experiment** | Một file YAML mô tả đầy đủ một lượt huấn luyện: model, runtime và các stage. Xem [Experiments](../experiments/). |
| **Stage** | Một bước trong pipeline của experiment (vd. `pretrain`, `sft1`), gồm recipe, dữ liệu, trọng số khởi tạo và tham số. |
| **Recipe** | Phương pháp huấn luyện một stage: `causal_lm` (pretrain) hoặc `sft` (hội thoại). Recipe quyết định thuật toán: dữ liệu nào, loss nào, optimizer và lịch learning rate nào. |
| **Nguồn dữ liệu** | Dữ liệu đã tokenize mà một stage đọc, loại `lm` (chuỗi token) hoặc `sft` (hội thoại có loss mask). Config ở `foundation_lab/data/configs/`. |
| **Objective** | Hàm loss, mặc định `lm_ce` (cross-entropy dự đoán token kế tiếp). |
| **Runtime** | Môi trường chạy: thiết bị, seed, `compile`, đo đạc, số worker. Đổi runtime không đổi thuật toán. |

## Kết quả

| Thuật ngữ | Nghĩa |
| --- | --- |
| **Run** | Một lần chạy experiment hoặc tác vụ, ghi vào `runs/<tên>/<thời gian>/`. Xem [Thư mục run](../runs/). |
| **`best.pt`** | Trọng số tốt nhất theo val loss của một stage, dạng `state_dict` thuần. Dùng để khởi tạo stage sau hoặc phát hành. |
| **`last.ckpt.pt`** | Checkpoint đầy đủ để chạy tiếp: model, optimizer, scheduler, vị trí huấn luyện, trạng thái RNG và sampler. |
| **Lineage** | Bản ghi nguồn gốc của run: commit git, môi trường, tokenizer, dữ liệu, phiên bản lab. |
| **Tác vụ** | Lệnh chạy một lần không qua pipeline stage, ví dụ `infer`; kết quả ghi vào `result.json` của run. |
