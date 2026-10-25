---
schema_version: 1
department: inference
page_id: decoding
title: Greedy, beam và sampling
description: Hiểu cách biến logits thành chuỗi token và đánh đổi của các chiến lược SAI đang hỗ trợ.
order: 10
kind: concept
updated: "2026-10-25"
tags: [greedy, beam, sampling, logits]
sources:
  - SAI/generate.py
  - foundation_lab/inference/decoders.py
  - foundation_lab/inference/registry.py
---

## Model dự đoán token, decoder chọn đường đi

Sau khi đọc prompt và các token đã sinh, model trả logits: điểm số cho từng token trong vocab. Softmax đổi chúng thành phân phối xác suất. Decoder quyết định chọn token nào, giữ bao nhiêu đường ứng viên và khi nào dừng.

Một token không nhất thiết là một từ. Với tokenizer subword, một từ có thể gồm nhiều token; dấu và khoảng trắng cũng ảnh hưởng cách mã hóa. Số token và số từ của câu trả lời vì vậy khác nhau.

## Greedy: chọn tốt nhất tại từng bước

Greedy lấy token có logit lớn nhất sau các quy tắc xử lý của decoder rồi tiếp tục một đường duy nhất. Nó không quay lại sửa lựa chọn ở bước trước.

Ưu điểm là cách chọn đơn giản, ít nhánh và dễ dùng làm phép đối chiếu. Hạn chế là một lựa chọn tốt cục bộ không bảo đảm cả câu có xác suất hoặc chất lượng tốt nhất. Greedy vẫn có thể lặp, sai sự thật hoặc dừng quá sớm.

`temperature`, `top_k`, `top_p` và seed lấy mẫu không điều khiển greedy trong implementation hiện tại. Tham số chống lặp và điều kiện dừng vẫn có vai trò. Đừng dùng `temperature=0` để yêu cầu greedy: validator yêu cầu temperature lớn hơn 0; hãy chọn `strategy: greedy`.

## Beam: giữ nhiều chuỗi ứng viên

Beam giữ tối đa `beam_size` nhánh đang sống và một tập nhánh đã kết thúc. Nó mở rộng các nhánh, xếp theo điểm log probability có điều chỉnh độ dài, rồi giữ các ứng viên phù hợp. Trong SAI, điểm xếp là:

```text
score = tổng log probability / độ_dài^length_penalty
```

`length_penalty` không đơn giản là “càng lớn càng phạt dài”: vì log probability thường âm, chia cho một số lớn hơn có thể bớt bất lợi cho chuỗi dài. Cần đọc đầu ra để đánh giá tác động thực tế.

Beam mở rộng không gian tìm kiếm nhưng tăng công việc và lượng cache cho các nhánh. Beam rộng hơn không bảo đảm câu trả lời đúng hơn. Một chuỗi xác suất cao có thể vẫn rập khuôn; nghiên cứu về text degeneration chỉ ra việc tối đa hóa likelihood khi giải mã không đồng nghĩa chất lượng văn bản mở cao nhất. [Holtzman và cộng sự, ICLR 2020](https://arxiv.org/abs/1904.09751).

SAI còn có `stable_steps` để dừng khi điều kiện ổn định của nhóm beam đã hoàn tất được đáp ứng. `null` tắt tiêu chí ổn định này nhưng vẫn còn token giới hạn và điều kiện hết beam.

## Sampling: lấy mẫu từ phân phối đã điều chỉnh

Sampling đi một đường, nhưng chọn token theo xác suất sau temperature và bộ lọc:

1. Áp các quy tắc chống lặp.
2. Chia logits cho `temperature`.
3. Nếu `top_k > 0`, giữ các token ở vùng điểm cao nhất theo ngưỡng top-k.
4. Nếu `top_p < 1`, lấy nucleus theo tổng xác suất tích luỹ và giữ ít nhất một token.
5. Softmax rồi lấy mẫu.

Temperature thấp làm phân phối tập trung hơn; cao làm phẳng hơn. Top-k giới hạn vùng ứng viên theo thứ hạng, top-p điều chỉnh theo khối xác suất nên số ứng viên thay đổi theo bước. Hai bộ lọc có thể cùng hoạt động theo thứ tự trên. SAI không có `min_p` trong `GenerationConfig` hiện tại.

Sampling thuận tiện để khảo sát sự đa dạng nhưng cũng có thể làm câu trả lời kém ổn định. Cố định seed giúp đối chiếu trong cùng điều kiện; nó không thay việc kiểm tra qua nhiều câu và nhiều lần chạy khi cần kết luận.

## `auto` trong phòng này là gì?

Trong `infer`, decoder `sai` gọi generate không truyền token constraint, nên `auto` chọn beam. API generate rộng hơn có thể dùng greedy khi truyền constraint; đó không phải control đang được expose trong spec `infer`.

Đọc [các tùy chọn](generation-options.md) trước khi ghép nhiều tham số, và luôn giữ [mốc mặc định](current-defaults.md) trong thí nghiệm.
