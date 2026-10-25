---
schema_version: 1
department: tokenizer
page_id: chat-masking
title: Chat template và mask SFT
description: Hiểu schema hội thoại, role model và cách đặt nhãn để chỉ học câu trả lời.
order: 30
kind: concept
updated: "2026-10-25"
tags: [SFT, chat template, mask, role, shift]
sources:
  - foundation_lab/tokenizer/encode.py
  - SAI/generate.py
  - tests/test_encode.py
  - foundation_lab/data/loaders.py
---

## Schema dữ liệu hiện có

Encoder `sft` đọc file JSON chứa mảng hội thoại; nếu gốc không phải mảng, reader xử lý như một phần tử. Mẫu hợp lệ cần `history` không rỗng. Mỗi phần tử history dùng khóa `user` và `model`; `instruction` tùy chọn đóng vai trò system.

```json
[
  {
    "instruction": "Bạn là trợ lý giải thích ngắn gọn.",
    "history": [
      {"user": "Token là gì?", "model": "Token là đơn vị văn bản mà mô hình xử lý."}
    ]
  }
]
```

Đây là dữ liệu minh hoạ, không chứa token ID thật. Schema encode hiện không tự nhận danh sách message dạng `{role, content}` như một số giao diện chat khác. API suy luận có hỗ trợ chuẩn hóa messages, nhưng không vì vậy mà encoder SFT có cùng schema input.

## Template biến hội thoại thành một chuỗi

Mỗi segment được dựng thành `<|im_start|>` + token của `role` và xuống dòng, nội dung + `<|im_end|>`. Role trả lời trong chuỗi SAI là `model`. Wrapper chuyển role và nội dung sang chữ thường trước encode.

```text
BOS
im_start system \n instruction im_end
im_start user   \n câu hỏi    im_end
im_start model  \n câu trả lời im_end
EOS
```

Sơ đồ dùng khoảng trắng để dễ đọc; không đại diện cho cách thêm khoảng trắng thực tế. `im_start`, `im_end`, BOS, EOS là token cấu trúc; chữ `system`, `user`, `model` và nội dung được tokenizer encode.

## Mask bám vào target đã shift

Encoder tạo X gồm toàn bộ chuỗi và Y bằng X bỏ token đầu. Mask có cùng độ dài Y, không phải cùng độ dài X trước padding. Mask bằng 0 ở segment system/user; bằng 1 ở **toàn bộ segment model**, gồm role và dấu phân cách, và ở EOS cuối.

Khi dự đoán token đầu của segment model, vị trí tương ứng trong X còn là token cuối của segment user. Vì vậy không tô mask theo token X đang nhìn thấy rồi coi đó là mask của Y; phải tô theo token đích. Vị trí X cuối không có nhãn kế tiếp được collator đệm mask bằng 0.

Prompt vẫn tham gia attention và tạo ngữ cảnh dù không chịu loss. Điều đó khác việc xóa prompt khỏi input. Xem [sổ tay Dữ liệu](../../data/docs/index.md) để phân biệt attention mask và loss weight.

## Mẫu nào bị bỏ?

Lượt không có câu trả lời `model` được bỏ cả lượt. Hội thoại không tạo được token nào bị loại. Nếu chuỗi X gồm BOS/EOS vượt `max_seq_len`, encoder SFT hiện loại cả mẫu; chưa có cơ chế chunk hoặc cắt hội thoại dài như pretrain. Theo dõi log `skipped` để không nhầm số mẫu thô với mẫu thực sự dùng.

## Vì sao template lúc train và generate phải khớp?

Suy luận dựng phần prompt và mở segment `model` để model viết tiếp. Role, newline, dấu phân cách, chuẩn hóa và tokenization đều cần nhất quán. Test nguồn `test_generate_prompt_is_prefix_of_sft_training_sample` kiểm tra prompt là tiền tố token của mẫu train tương ứng. Đây là hợp đồng kỹ thuật, không nên thay dấu hoặc đổi role chỉ để nội dung hiển thị đẹp hơn.

Tiếp theo: [Chia cửa sổ pretrain](chunking.md).
