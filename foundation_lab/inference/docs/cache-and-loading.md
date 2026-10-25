---
schema_version: 1
department: inference
page_id: cache-and-loading
title: Nạp model, ngân sách context và KV cache
description: Hiểu quan hệ model với checkpoint, quy trình prompt và bộ nhớ của prefill/decode.
order: 30
kind: concept
updated: "2026-10-25"
tags: [checkpoint, kv cache, prefill, decode, context]
sources:
  - foundation_lab/inference/loading.py
  - foundation_lab/inference/bench.py
  - foundation_lab/state/checkpoint.py
  - SAI/generate.py
  - SAI/modeling.py
  - SAI/attention.py
  - config/models/SAI_35M.yaml
  - config/models/SAI_100M.yaml
  - config/models/SAI_375M.yaml
---

## Config dựng hình dạng, checkpoint cung cấp trọng số

Loader dựng model từ config đã resolve, nạp trọng số bằng loader chung, chuyển sang device và gọi `eval()`. Tokenizer của phòng infer hiện luôn là tokenizer chính thức. Một checkpoint chỉ có thể dùng đúng khi kiến trúc và bộ ID token tương thích.

Spec đòi capability `causal_lm` và `kv_cache`. Không phải mọi họ model trong registry tự động đáp ứng đường generate này. Kiểm tra config và file tồn tại xảy ra trước khi tạo run, nhưng lỗi nội dung hoặc shape checkpoint có thể chỉ xuất hiện lúc nạp.

`infer.device` dùng lựa chọn auto/CUDA/MPS/CPU, không đưa toàn bộ `RuntimeSpec` training vào suy luận. Spec hiện không có selector quantization, dtype, compile hoặc attention kernel riêng cho decoder. Không suy rằng chính sách AMP của engine train tự áp dụng cho tác vụ này.

## Prompt được dựng như thế nào?

Decoder `sai` chuyển mỗi prompt thành message user trong định dạng chat có BOS, dấu đầu/cuối message và tiền tố lượt `model`. Với default, phần văn bản được viết thường trước khi tokenize. Vì có các token định dạng, `prompt_tokens` không chỉ là số token của chuỗi bạn gõ.

Ngân sách prompt:

```text
prompt_budget = model.max_seq_len - max_new_tokens
```

Hệ thống phải còn chỗ cho BOS, prefix model và message cần giữ. Nếu system cùng câu user mới nhất không vừa, generate báo lỗi; nó không âm thầm cắt câu user mới nhất. API chat tổng quát có thể bỏ các nhóm lịch sử cũ để vừa ngân sách, nhưng spec `infer` hiện chỉ nhận danh sách prompt độc lập, không expose lịch sử hội thoại hoặc system field.

Ba config SAI chính thức 35M/100M/375M tại snapshot này đều có `max_seq_len=10240`. Đây là giá trị của config, không phải cam kết mọi prompt dài đó chạy đủ bộ nhớ trên máy đang dùng. UI phải lấy model hiện tại từ config, không cứng hóa con số này.

## Prefill và decode

**Prefill** xử lý prompt, tính key/value cho các token đã thấy và lấy logits ở vị trí cuối để chọn token đầu tiên. `SAIModel.prefill` chỉ chiếu hidden state vị trí cuối cần dùng qua LM head, tránh tạo toàn bộ logits prompt × vocab cho đường này.

**Decode** lần lượt đưa token mới nhất vào model. Mỗi layer ghi key/value mới vào cache rồi attention tới cache trước đó. Nhờ giữ key/value, các token cũ không phải được tính lại từ đầu ở mỗi bước. Tuy vậy attention của token mới vẫn đọc lịch sử cache, nên độ dài context vẫn ảnh hưởng công việc và bộ nhớ.

Trong greedy/sample, `decode_steps` có thể bằng `completion_tokens - 1` khi sinh liên tục, vì token đầu đến từ prefill. Nếu dừng, con số phản ánh số lần gọi decode thực tế. Với beam, mỗi lần decode xử lý nhiều nhánh; không đồng nhất decode step với một token của chuỗi cuối.

## Bộ nhớ KV cache

Mỗi layer giữ hai tensor key/value có shape:

```text
(số_nhánh, num_kv_heads, sức_chứa_context, head_dim)
```

Ước lượng riêng tensor cache là `2 × layers × số_nhánh × H_kv × sức_chứa × head_dim × byte_mỗi_phần_tử`. Đây chưa gồm trọng số, activation, logits hoặc bộ nhớ tạm.

Greedy/sample cấp cache cho một nhánh; beam cấp theo beam size. Cache sử dụng số KV head, không phải số query head. Trong SAI 375M snapshot, 16 query head chia sẻ 4 KV head, giúp cache khác với MHA có 16 KV head; không vì vậy mà cache trở nên miễn phí.

Tăng prompt, max token sinh hoặc beam size có thể tăng nhu cầu cache. Xem [cấu hình sinh](generation-options.md) trước khi điều chỉnh và [Sổ tay Lưu trữ](../../state/docs/index.md) khi chọn trọng số.
