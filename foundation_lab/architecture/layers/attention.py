import torch
import torch.nn as nn
from SAI.rope import RotaryPositionalEmbedding
from foundation_lab.architecture.registry import ATTENTIONS, ATTENTION_KERNELS

@ATTENTIONS.register("gqa")
class ModularGroupedQueryAttention(nn.Module):
    """GQA như ``SAI.attention.GroupedQueryAttention``, nhưng phép tính attention được uỷ cho kernel
    chọn theo tên (mặc định ``sdpa``). Kernel không có tham số nên đổi kernel không đổi state_dict."""

    def __init__(self, d_model: int, num_heads: int, num_kv_heads: int, dropout: float, kernel: str = "sdpa"):
        super().__init__()
        assert d_model % num_heads == 0
        assert num_heads % num_kv_heads == 0, \
            "num_heads phải chia hết cho num_kv_heads"

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_groups = num_heads // num_kv_heads
        self.d_k = d_model // num_heads
        self.dropout_rate = dropout
        self.q_dim = num_heads * self.d_k
        self.kv_dim = num_kv_heads * self.d_k
        self.kernel_name = kernel
        self.kernel = ATTENTION_KERNELS.get(kernel)
        self.wqkv = nn.Linear(d_model, self.q_dim + 2 * self.kv_dim, bias=False)
        self.wo = nn.Linear(self.q_dim, d_model, bias=False)

    def extra_repr(self) -> str:
        return f"heads={self.num_heads}, kv_heads={self.num_kv_heads}, kernel={self.kernel_name}"

    def _project_qkv(self, x: torch.Tensor):
        B, T, _ = x.shape
        qkv = self.wqkv(x)
        q, k, v = qkv.split([self.q_dim, self.kv_dim, self.kv_dim], dim=-1)
        q = q.view(B, T, self.num_heads, self.d_k)
        k = k.view(B, T, self.num_kv_heads, self.d_k)
        v = v.view(B, T, self.num_kv_heads, self.d_k)
        return q, k, v

    def _merge(self, out: torch.Tensor) -> torch.Tensor:
        B, _, T, _ = out.shape
        return self.wo(out.transpose(1, 2).contiguous().view(B, T, self.num_heads * self.d_k))

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, attn_mask=None) -> torch.Tensor:
        q, k, v = self._project_qkv(x)

        q = RotaryPositionalEmbedding.apply_rope(q, cos, sin)
        k = RotaryPositionalEmbedding.apply_rope(k, cos, sin)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        dropout_p = self.dropout_rate if self.training else 0.0

        out = self.kernel(
            q, k, v, attn_mask=attn_mask, is_causal=(attn_mask is None), dropout_p=dropout_p,
        )
        return self._merge(out)

    def prefill(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        q, k, v = self._project_qkv(x)

        q = RotaryPositionalEmbedding.apply_rope(q, cos, sin)
        k = RotaryPositionalEmbedding.apply_rope(k, cos, sin)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        out = self.kernel(q, k, v, is_causal=True)
        return self._merge(out), (k, v)

    def forward_with_cache(self, x: torch.Tensor, past_kv, cache_len: int, cos: torch.Tensor, sin: torch.Tensor):
        B, T, _ = x.shape
        q, k, v = self._project_qkv(x)

        q = RotaryPositionalEmbedding.apply_rope(q, cos, sin)
        k = RotaryPositionalEmbedding.apply_rope(k, cos, sin)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        past_kv[0][:B, :, cache_len:cache_len + T, :] = k
        past_kv[1][:B, :, cache_len:cache_len + T, :] = v

        k_full = past_kv[0][:B, :, :cache_len + T, :]
        v_full = past_kv[1][:B, :, :cache_len + T, :]

        out = self.kernel(q, k_full, v_full, is_causal=False)
        return self._merge(out)
