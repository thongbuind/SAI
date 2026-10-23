---
$front_matter
---

# SAI — Mô hình ngôn ngữ **decoder-only Transformer** thuần Việt

## Kiến trúc mô hình

$architecture_table

### Tokenizer

Sử dụng thư viện **SentencePiece Unigram** với vocabulary $vocab_size token. Tokenizer được huấn luyện trên dữ liệu tiếng Việt đã chuyển về chữ thường, giữ nguyên khoảng trắng và bật byte fallback để hạn chế token không xác định.

### Embedding

Mỗi token ID được ánh xạ thành một vector có kích thước `d_model` bằng `nn.Embedding`. Khác với Transformer trong bài báo *Attention Is All You Need*, vốn cộng sinusoidal positional encoding trực tiếp vào token embedding, SAI sử dụng **Rotary Positional Embedding (RoPE)**. RoPE mã hoá vị trí tương đối bằng cách xoay các vector query và key trong attention. Context tối đa của mô hình là $max_seq_len token.

Trọng số của output language-model head được chia sẻ với ma trận embedding (weight tying). Cách này giảm số tham số và buộc biểu diễn đầu vào, đầu ra cùng nằm trong một không gian token.

### Grouped Query Attention

Query, key và value được chiếu qua một lớp tuyến tính gộp không dùng bias. Nhiều query head có thể dùng chung một cặp key/value head, nhờ đó giảm kích thước KV cache so với Multi-Head Attention thông thường.

Số query head và KV head của bản phát hành được lấy trực tiếp từ config trong bảng trên.

Attention được tính bằng `scaled_dot_product_attention` của PyTorch, cho phép tự chọn cuDNN Attention, Flash Attention, memory-efficient attention hoặc math backend phù hợp. Causal mask ngăn mô hình nhìn thấy token tương lai; padding mask loại bỏ token đệm khi huấn luyện.

### Decoder Block

Mỗi decoder block dùng kiến trúc **pre-norm** với hai nhánh residual:

```text
x = x + Dropout(GQA(RMSNorm(x)))
x = x + Dropout(SwiGLU(RMSNorm(x)))
```

Feed-forward network dùng SwiGLU: một phép chiếu sinh đồng thời nhánh `gate` và `up`, sau đó tính `up × SiLU(gate)` và chiếu trở lại `d_model`. Tất cả lớp tuyến tính trong attention và SwiGLU đều không dùng bias. Sau chồng decoder block, mô hình áp dụng một RMSNorm cuối trước language-model head.

So với Transformer gốc, SAI chỉ giữ phần decoder và thay LayerNorm bằng RMSNorm, ReLU FFN bằng SwiGLU. Transformer gốc áp dụng normalization sau residual connection (post-norm), trong khi SAI chuẩn hoá đầu vào trước attention và feed-forward network (pre-norm).

### Training

Quá trình huấn luyện gồm bốn giai đoạn: pretraining, continued pretraining, SFT1 và SFT2. Pretraining tối ưu next-token cross-entropy trên toàn bộ chuỗi; SFT dùng loss mask để chỉ học trên token thuộc câu trả lời.

Loss được chia thành từng phần nhỏ để giảm lượng bộ nhớ cần dùng khi huấn luyện. Pipeline cũng hỗ trợ mixed precision, gradient accumulation và `torch.compile` để tăng tốc trên GPU.

### Inference

Prompt của người dùng được đóng gói theo chat template của tokenizer, sau đó thêm prefix cho vai trò `model`. Bước **prefill** xử lý toàn bộ prompt một lần và lưu key/value của từng decoder block. Các bước sau chỉ tính token mới và nối key/value vào cache, thay vì chạy lại toàn bộ chuỗi.

Mặc định, chế độ `auto` sử dụng beam search với 5 beam, sinh tối đa 500 token, repetition penalty `1.2` và no-repeat 3-gram. Beam search dừng khi phương án hoàn thành tốt nhất ổn định trong 8 bước, khi gặp `[EOS]`, `<|im_end|>` hoặc khi đạt giới hạn context. Hàm generate cũng hỗ trợ greedy decoding, sampling, system prompt, lịch sử hội thoại, custom stop sequence và token constraint.

## Phiên bản phát hành

$release_table

## Tải mô hình và chạy thử

```bash
pip install $runtime_requirements
```

Ví dụ chạy thử:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


model_id = "$repo_id"

tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    trust_remote_code=True,
    revision="$version",
)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    revision="$version",
)

device = "cuda" if torch.cuda.is_available() else (
    "mps" if torch.backends.mps.is_available() else "cpu"
)
model = model.to(device).eval()

messages = $example_messages

answer = model.generate_response(
    tokenizer=tokenizer,
    messages=messages,
)
print(answer)
```

`generate_response()` dùng thuật toán hội thoại của SAI. `generate()` nhận tensor token ID
và trả về token ID theo giao diện Transformers:

```python
inputs = tokenizer("xin chào", return_tensors="pt").to(device)
outputs = model.generate(**inputs, max_new_tokens=32, do_sample=False)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

Adapter hỗ trợ `DynamicCache`, padding trái/phải trong forward và padding trái khi sinh theo batch.
Các tuỳ chọn trả attention, hidden state trung gian và các loại cache khác chưa được hỗ trợ.
`release_manifest.json` ghi hash nội dung cùng phiên bản thư viện dùng lúc export;
báo cáo verify local ghi môi trường thực tế và kết quả kiểm chứng.
