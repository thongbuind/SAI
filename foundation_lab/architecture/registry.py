from dataclasses import dataclass
from typing import Callable
from foundation_lab.core.registry import Registry

@dataclass(frozen=True)
class ModelFamily:
    """Một họ mô hình trong lab: dataclass config, hàm dựng model từ config, capability khai báo."""
    config_cls: type
    build: Callable
    capabilities: frozenset

# Phép tính attention thuần (không tham số): (q, k, v, *, attn_mask, is_causal, dropout_p) -> out.
ATTENTION_KERNELS = Registry("attention_kernel")
# Module attention có tham số: (d_model, num_heads, num_kv_heads, dropout, kernel) -> nn.Module.
ATTENTIONS = Registry("attention")
# Feed-forward: (d_model, ff_dim) -> nn.Module.
FFNS = Registry("ffn")
# Normalization: (dim) -> nn.Module.
NORMS = Registry("norm")
# Họ mô hình: ModelFamily.
MODELS = Registry("model")
