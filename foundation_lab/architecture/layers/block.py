import torch.nn as nn
from foundation_lab.architecture.registry import ATTENTIONS, FFNS, NORMS

class DecoderBlock(nn.Module):
    """Pre-norm decoder block. Thành phần attention/FFN/norm chọn theo tên;
    tên thuộc tính (gqa, ffn, norm1, norm2) giữ nguyên để state_dict khớp các checkpoint đã phát hành."""

    def __init__(
        self, d_model: int, num_heads: int, num_kv_heads: int, ff_dim: int, dropout: float,
        attention: str = "gqa", attention_kernel: str = "sdpa", ffn: str = "swiglu", norm: str = "rmsnorm",
    ):
        super().__init__()
        self.gqa = ATTENTIONS.get(attention)(d_model, num_heads, num_kv_heads, dropout, kernel=attention_kernel)
        self.ffn = FFNS.get(ffn)(d_model, ff_dim)
        self.norm1 = NORMS.get(norm)(d_model)
        self.norm2 = NORMS.get(norm)(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, cos, sin, attn_mask=None):
        x = x + self.drop(self.gqa(self.norm1(x), cos, sin, attn_mask))
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x

    def prefill(self, x, cos, sin):
        attn, kv = self.gqa.prefill(self.norm1(x), cos, sin)
        x = x + self.drop(attn)
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x, list(kv)

    def forward_with_cache(self, x, kv, cache_len: int, cos, sin):
        x = x + self.drop(self.gqa.forward_with_cache(self.norm1(x), kv, cache_len, cos, sin))
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x
