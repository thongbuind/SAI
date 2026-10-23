import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from foundation_lab.objectives.registry import OBJECTIVES

@OBJECTIVES.register("lm_ce")
def chunked_lm_loss(hidden: torch.Tensor, weight: torch.Tensor, targets: torch.Tensor,
                    token_weight: torch.Tensor, chunk_size: int = 4096) -> torch.Tensor:
    """Cross-entropy next-token, trả về TỔNG có trọng số (chưa chia mẫu số).

    Tính logits theo từng chunk ``chunk_size`` hàng (kèm activation checkpoint khi có grad) để không
    vật lý hoá full logits (N, vocab). Việc chuẩn hoá (chia số token) do recipe/engine quyết định.

    - ``hidden``: (N, d_model) đầu ra ``forward_features`` đã flatten.
    - ``weight``: (vocab, d_model) ma trận của lm_head.
    - ``targets``: (N,) long; ``token_weight``: (N,) float.
    """
    N = hidden.shape[0]
    total_loss = hidden.new_zeros((), dtype=torch.float32)

    def _chunk_fn(h_chunk, t_chunk, w_chunk):
        logits_chunk = F.linear(h_chunk, weight).float()
        loss_per_token = F.cross_entropy(logits_chunk, t_chunk, reduction='none')
        return (loss_per_token * w_chunk).sum()

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        chunk_args = (
            hidden[start:end],
            targets[start:end],
            token_weight[start:end],
        )
        if torch.is_grad_enabled():
            chunk_loss = checkpoint(
                _chunk_fn, *chunk_args, use_reentrant=False,
            )
        else:
            chunk_loss = _chunk_fn(*chunk_args)
        total_loss = total_loss + chunk_loss

    return total_loss
