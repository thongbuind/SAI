# SAI - Mô hình ngôn ngữ **decoder-only Transformer** thuần Việt

## Kiến trúc mô hình

| Cấu hình | `d_model` | Attention heads | KV heads | Layers | FFN dimension |
|---|---:|---:|---:|---:|---:|
| 35M | 512 | 8 | 8 | 8 | 2.048 |
| 100M | 768 | 12 | 6 | 12 | 3.072 |
| 375M | 1.024 | 16 | 4 | 24 | 4.096 |

### Tokenizer

Sử dụng thư viện **SentencePiece Unigram** với vocabulary 10.000 token dành riêng cho những tác vụ tiếng Việt. Tokenizer được huấn luyện trên dữ liệu đã chuyển về chữ thường, giữ nguyên khoảng trắng và bật byte fallback để hạn chế token không xác định.

### Embedding

Mỗi token ID được ánh xạ thành một vector có kích thước `d_model` bằng `nn.Embedding`. Khác với Transformer trong bài báo *Attention Is All You Need*, vốn cộng sinusoidal positional encoding trực tiếp vào token embedding, SAI sử dụng **Rotary Positional Embedding (RoPE)**. RoPE mã hoá vị trí tương đối bằng cách xoay các vector query và key trong attention. Context tối đa của mô hình là 10.240 token.

Trọng số của output language-model head được chia sẻ với ma trận embedding (weight tying). Cách này giảm số tham số và buộc biểu diễn đầu vào, đầu ra cùng nằm trong một không gian token.

### Grouped Query Attention

Query, key và value được chiếu qua một lớp tuyến tính gộp không dùng bias. Nhiều query head có thể dùng chung một cặp key/value head, nhờ đó giảm kích thước KV cache so với Multi-Head Attention thông thường.

- Cấu hình 35M có 8 query head và 8 KV head (tương đương Multi-Head Attention).
- Cấu hình 100M có 12 query head và 6 KV head.
- Cấu hình 375M có 16 query head và 4 KV head.

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

Cấu hình mặc định lấy từ `GenerationConfig` trong [`SAI/generate.py`](SAI/generate.py). Quá trình sinh dừng khi gặp `[EOS]`, `<|im_end|>`, đạt giới hạn context hoặc thoả điều kiện dừng sớm.

## Tải mô hình và chạy thử

Checkpoint hiện được phát hành là **SAI_35M**, **SAI_100M** và **SAI_375M** tại [thongbuind trên Hugging Face](https://huggingface.co/collections/thongbuind/sai). Mô hình có thể được tải trực tiếp bằng thư viện Transformers.
 
```bash
pip install torch transformers accelerate sentencepiece
```

Ví dụ chạy thử:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "thongbuind/SAI_375M"

tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    trust_remote_code=True,
)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
)

device = "cuda" if torch.cuda.is_available() else (
    "mps" if torch.backends.mps.is_available() else "cpu"
)
model = model.to(device).eval()

messages = [
    {
        "role": "system",
        "content": "Bạn là trợ lý trả lời ngắn gọn.",
    },
    {
        "role": "user",
        "content": "Trí tuệ nhân tạo là gì",
    },
]

answer = model.generate_response(
    tokenizer=tokenizer,
    messages=messages,
)
print(answer)
```

Trong lần chạy đầu tiên, Transformers sẽ tự tải model, tokenizer và config; các lần sau sẽ sử dụng bộ nhớ đệm cục bộ.

`model.generate_response(tokenizer=..., messages=...)` là cách gọi hội thoại thống nhất cho model
lõi `SAIModel` và model tải bằng Transformers. Mặc định trả chuỗi; thêm `return_result=True`
để nhận token và thông tin kết thúc. Cách truyền chuỗi qua `user_input` vẫn được hỗ trợ,
nhưng không trộn với `messages`.

Với model Transformers, `model.generate(input_ids=...)` là giao diện sinh token ID của thư viện;
không truyền `tokenizer` hoặc `messages` vào hàm này.
