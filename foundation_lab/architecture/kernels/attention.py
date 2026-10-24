"""Attention kernel: phép tính softmax(QKᵀ/√d)·V, không có tham số.

Hợp đồng chung cho mọi kernel đăng ký vào ATTENTION_KERNELS:
- ``q``: (B, H, Tq, D); ``k``, ``v``: (B, Hkv, Tk, D) với H chia hết cho Hkv (GQA: query head h dùng
  KV head h // (H / Hkv)).
- ``is_causal=True``: causal mask căn góc trên-trái (token i thấy token j ≤ i), giống SDPA.
- ``attn_mask``: bool (True = được nhìn) hoặc float (cộng vào score). Không dùng cùng ``is_causal``.
- ``dropout_p``: dropout trên trọng số attention.
- Trả về (B, H, Tq, D).

"""

import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from foundation_lab.architecture.registry import ATTENTION_KERNELS

_SDPA_BACKENDS = [
    SDPBackend.FLASH_ATTENTION,
    SDPBackend.EFFICIENT_ATTENTION,
    SDPBackend.MATH,
]

@ATTENTION_KERNELS.register("sdpa")
def sdpa_attention(q, k, v, *, attn_mask=None, is_causal=False, dropout_p=0.0):
    """Kernel PyTorch SDPA, ưu tiên Flash → memory-efficient → math."""
    with sdpa_kernel(_SDPA_BACKENDS):
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=is_causal, dropout_p=dropout_p,
            enable_gqa=True,
        )

@ATTENTION_KERNELS.register("eager")
def eager_attention(q, k, v, *, attn_mask=None, is_causal=False, dropout_p=0.0):
    """Bản tham chiếu viết tay bằng phép toán PyTorch thường; chậm, dùng để đối chiếu."""
    if is_causal and attn_mask is not None:
        raise ValueError("Không dùng đồng thời attn_mask và is_causal")
    groups = q.size(1) // k.size(1)
    if groups > 1:
        k = k.repeat_interleave(groups, dim=1)
        v = v.repeat_interleave(groups, dim=1)

    scores = (q @ k.transpose(-2, -1)) * (q.size(-1) ** -0.5)
    if is_causal:
        t_q, t_k = q.size(-2), k.size(-2)
        allowed = torch.ones(t_q, t_k, dtype=torch.bool, device=q.device).tril()
        scores = scores.masked_fill(~allowed, float("-inf"))
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            scores = scores.masked_fill(~attn_mask, float("-inf"))
        else:
            scores = scores + attn_mask

    weights = torch.softmax(scores.float(), dim=-1).to(q.dtype)
    if dropout_p > 0.0:
        weights = F.dropout(weights, p=dropout_p, training=True)
    return weights @ v
