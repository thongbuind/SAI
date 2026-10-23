"""Công cụ so sánh của phòng Đánh giá.

``compare_attention_kernels``: so một attention kernel với kernel tham chiếu trên cùng input ngẫu nhiên
— sai khác output, sai khác gradient theo q/k/v, thời gian forward+backward. Kernel không có tham số
nên kết quả khớp nghĩa là có thể thay kernel mà dùng lại nguyên trọng số đã train.
"""

import statistics
import time
import torch
from foundation_lab.architecture.registry import ATTENTION_KERNELS

def _sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()

def compare_attention_kernels(
    reference: str = "sdpa", candidate: str = "eager", *,
    batch_size: int = 2, num_heads: int = 8, num_kv_heads: int = 2, seq_len: int = 128,
    head_dim: int = 64, dtype: torch.dtype = torch.float32, device: str = "cpu",
    causal: bool = True, repeats: int = 5, seed: int = 0,
) -> dict:
    device = torch.device(device)
    generator = torch.Generator(device="cpu").manual_seed(seed)

    def randn(*shape):
        return torch.randn(*shape, generator=generator).to(device=device, dtype=dtype)

    q = randn(batch_size, num_heads, seq_len, head_dim)
    k = randn(batch_size, num_kv_heads, seq_len, head_dim)
    v = randn(batch_size, num_kv_heads, seq_len, head_dim)
    grad_out = randn(batch_size, num_heads, seq_len, head_dim)

    def run(name):
        kernel = ATTENTION_KERNELS.get(name)
        inputs = [t.clone().requires_grad_(True) for t in (q, k, v)]
        out = kernel(*inputs, is_causal=causal)
        out.backward(grad_out)
        return out.detach(), [t.grad for t in inputs]

    def timed(name):
        times = []
        for _ in range(repeats):
            _sync(device)
            start = time.perf_counter()
            run(name)
            _sync(device)
            times.append((time.perf_counter() - start) * 1000)
        return statistics.median(times)

    ref_out, ref_grads = run(reference)
    cand_out, cand_grads = run(candidate)
    report = {
        "reference": reference,
        "candidate": candidate,
        "dtype": str(dtype).replace("torch.", ""),
        "shape": {"B": batch_size, "H": num_heads, "Hkv": num_kv_heads, "T": seq_len, "D": head_dim},
        "max_abs_diff_out": (ref_out - cand_out).abs().max().item(),
    }
    for name, ref_grad, cand_grad in zip(("q", "k", "v"), ref_grads, cand_grads):
        report[f"max_abs_diff_grad_{name}"] = (ref_grad - cand_grad).abs().max().item()
    report["reference_ms"] = timed(reference)
    report["candidate_ms"] = timed(candidate)
    return report
