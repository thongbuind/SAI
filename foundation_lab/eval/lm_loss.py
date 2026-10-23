import time
import torch
from foundation_lab.engine.step import get_adaptive_microbatch_shape
from foundation_lab.objectives.lm_ce import chunked_lm_loss

def evaluate_lm_loss(model, features, loader, *, runtime, max_microbatch_tokens, chunk_size, label, log_every=100) -> float:
    """Loss CE trung bình theo token chịu loss trên toàn bộ ``loader`` (val/test)."""
    device = runtime.device
    head_weight = model.lm_head.weight
    model.eval()
    loss_sum_total = torch.zeros((), device=device)
    valid_tokens = torch.zeros((), device=device)
    eval_start = time.time()

    with torch.no_grad():
        for eval_idx, batch in enumerate(loader):
            batch = batch.to(device, non_blocking=True)
            logical_batch_size, padded_length = batch.shape
            _, microbatch_size = get_adaptive_microbatch_shape(
                logical_batch_size, padded_length, max_microbatch_tokens
            )

            for micro_start in range(0, logical_batch_size, microbatch_size):
                micro_end = micro_start + microbatch_size
                with runtime.autocast():
                    hidden = features(
                        batch.input_ids[micro_start:micro_end],
                        attention_mask=batch.attention_mask[micro_start:micro_end],
                        has_padding=batch.has_padding,
                    )
                    hidden_flat = hidden.reshape(-1, hidden.size(-1))
                    targets_flat = batch.labels[micro_start:micro_end].reshape(-1)
                    weight_flat = batch.loss_weight[micro_start:micro_end].reshape(-1)
                    loss_sum = chunked_lm_loss(
                        hidden_flat, head_weight, targets_flat, weight_flat,
                        chunk_size=chunk_size,
                    )
                loss_sum_total += loss_sum.detach()
                valid_tokens += weight_flat.sum()

                del hidden, hidden_flat, targets_flat, weight_flat, loss_sum

            if (eval_idx + 1) % log_every == 0 or (eval_idx + 1) == len(loader):
                print(f"\r{label} {eval_idx+1}/{len(loader)} - {time.time() - eval_start:.0f}s", end="", flush=True)

            del batch

    print()
    return (loss_sum_total / (valid_tokens + 1e-8)).item()
