"""Công cụ đo đạc: VRAM theo pha, ước tính FLOPs, đo TFLOPS, log kernel."""

import json
import time
from pathlib import Path
import torch

VRAM_PROFILE_STEPS = 20

def cuda_memory_gib():
    """Chụp VRAM sau khi đồng bộ, tách tensor đang sống và allocator cache."""
    torch.cuda.synchronize()
    gib = 1024 ** 3
    return {
        "allocated": torch.cuda.memory_allocated() / gib,
        "reserved": torch.cuda.memory_reserved() / gib,
        "peak_allocated": torch.cuda.max_memory_allocated() / gib,
        "peak_reserved": torch.cuda.max_memory_reserved() / gib,
    }

def format_vram_mark(mark):
    return f"A={mark['allocated']:.2f}GiB/R={mark['reserved']:.2f}GiB"

def print_vram_profile(
    batch_idx, batch_shape, has_padding, optimizer_updated,
    microbatch_count, microbatch_size, marks,
):
    peak = marks["released"]
    timeline = " | ".join(
        f"{name}:{format_vram_mark(marks[name])}"
        for name in ("start", "h2d", "forward", "loss", "backward", "update", "released")
    )
    print()
    print(
        f"[VRAM] batch={batch_idx + 1} B={batch_shape[0]} T={batch_shape[1]} "
        f"padding={has_padding} micro={microbatch_count}x<={microbatch_size} "
        f"optimizer_step={optimizer_updated} | {timeline} | "
        f"PEAK A={peak['peak_allocated']:.2f}GiB/R={peak['peak_reserved']:.2f}GiB"
    )


def format_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h}h {m}m {s:.1f}s"

def estimate_training_flops(
    model: torch.nn.Module,
    num_layers: int,
    max_seq_len: int,
    d_model: int,
    epochs: int,
    batches_per_epoch: int,
    batch_size: int,
) -> dict:
    n_params = sum(p.numel() for p in model.parameters())
    flops_per_token = 6 * n_params + 6 * num_layers * max_seq_len * d_model
    total_token_slots = epochs * batches_per_epoch * batch_size * max_seq_len
    total_flops = flops_per_token * total_token_slots
    total_tflops = total_flops / 1e12

    return {
        "n_params": n_params,
        "flops_per_token": flops_per_token,
        "total_batches_per_epoch": batches_per_epoch,
        "total_token_slots": total_token_slots,
        "total_flops": total_flops,
        "total_tflops": total_tflops,
    }

def print_training_flops_summary(flops_info: dict, epochs: int) -> None:
    """In bảng tổng FLOPs cần cho training."""
    print("╠════════════════════════════════════════════════════════════════════════════════════╣")
    print("║                      TINH TONG FLOPs CAN CHO TOAN BO TRAINING                      ║")
    print("╠════════════════════════════════════════════════════════════════════════════════════╣")
    print(f"So tham so model: {flops_info['n_params']:,}")
    print(
        f"Tong so TFLOP can cho toan bo qua trinh training "
        f"({epochs} epoch, {flops_info['total_batches_per_epoch']} batch/epoch): "
        f"{flops_info['total_tflops']:,.2f} TFLOP"
    )
    print("╠════════════════════════════════════════════════════════════════════════════════════╣")

class TflopsBenchmarker:
    def __init__(
        self,
        flops_per_token: float,
        total_flops_needed: float,
        total_batches_per_epoch: int,
        epochs: int,
        device: torch.device,
        warmup_steps: int = 500,
        target_steps: int = 1000,
    ):
        self.flops_per_token = flops_per_token
        self.total_flops_needed = total_flops_needed
        self.total_batches_per_epoch = total_batches_per_epoch
        self.epochs = epochs
        self.device = device

        self.warmup_steps = min(warmup_steps, total_batches_per_epoch)
        self.target_steps = min(target_steps, total_batches_per_epoch)
        self.reported = self.target_steps <= self.warmup_steps

        self.tokens = 0
        self.start_time = None

    def is_bench_step(self, epoch: int, start_epoch: int, batch_idx: int) -> bool:
        return (
            (not self.reported)
            and (epoch == start_epoch)
            and (self.warmup_steps <= batch_idx < self.target_steps)
        )

    def on_step_begin(self, batch_idx: int, batch_tokens: int) -> None:
        """Gọi khi bắt đầu step nằm trong vùng đo."""
        if batch_idx == self.warmup_steps:
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            self.start_time = time.perf_counter()
        self.tokens += batch_tokens

    def on_step_end(self, batch_idx: int) -> None:
        """Gọi ở step cuối vùng đo → in kết quả."""
        if batch_idx != self.target_steps - 1:
            return

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - self.start_time
        measured_steps = self.target_steps - self.warmup_steps
        flops = self.flops_per_token * self.tokens
        tflops_per_sec = (flops / elapsed) / 1e12 if elapsed > 0 else 0.0
        avg_time_per_step = elapsed / measured_steps if measured_steps > 0 else 0.0
        token_slots_per_sec = self.tokens / elapsed if elapsed > 0 else 0.0

        print()
        print("╠════════════════════════════════════════════════════════════════════════════════════╣")
        print(
            f"[DO TFLOPS] Do {measured_steps} step training that "
            f"(step {self.warmup_steps + 1} → {self.target_steps}), "
            f"mat {elapsed:.2f}s (da bo qua {self.warmup_steps} step warm-up dau)"
        )
        print(f"[DO TOC DO] Trung binh moi step: {avg_time_per_step * 1000:,.2f} ms/step")
        print(f"[DO TOC DO] Thong luong tensor thuc te: {token_slots_per_sec:,.0f} token-slot/s")

        if tflops_per_sec > 0:
            print(
                f"[DO TFLOPS] TFLOPS uoc tinh theo cong thuc model: "
                f"{tflops_per_sec:,.2f} TFLOPS"
            )

            total_steps_all_epochs = self.epochs * self.total_batches_per_epoch

            est_time_actual = avg_time_per_step * total_steps_all_epochs
            print(f"[Cach 1 - Thuc te] Uoc tinh thoi gian train (ngoai suy toc do do duoc): {format_time(est_time_actual)}")

            # Cách 2 – lý thuyết
            est_time_theory = self.total_flops_needed / (tflops_per_sec * 1e12)
            print(f"[Cach 2 - Ly thuyet] Uoc tinh thoi gian train (Tong FLOPs / TFLOPS do duoc): {format_time(est_time_theory)}")
        else:
            print("[CẢNH BÁO] Thời gian đo quá nhỏ để tính TFLOPS đáng tin cậy.")

        print("╠════════════════════════════════════════════════════════════════════════════════════╣")
        self.reported = True

class KernelLogger:
    def __init__(self, enabled: bool = False, log_step: int = 1000, out_dir=None):
        self.enabled = enabled
        self.log_step = log_step
        self.out_dir = Path(out_dir) if out_dir is not None else Path(".")
        self.reported = not enabled
        self._prof_ctx = None

    def should_log(self, epoch: int, start_epoch: int, batch_idx: int) -> bool:
        return (
            (not self.reported)
            and (epoch == start_epoch)
            and (batch_idx == self.log_step)
        )

    def start(self):
        self._prof_ctx = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=True,
        )
        self._prof_ctx.__enter__()

    def stop_and_report(self, batch_idx: int, epoch: int, device: torch.device):
        if device.type == "cuda":
            torch.cuda.synchronize()

        self._prof_ctx.__exit__(None, None, None)

        # Gom theo operator + input shape để chẩn đoán GEMM/Linear shape
        # mà không phải lưu timeline gồm hàng triệu event.
        key_avgs = self._prof_ctx.key_averages(group_by_input_shape=True)

        def _self_cuda_us(e):
            if hasattr(e, "self_cuda_time_total"):
                return e.self_cuda_time_total
            return getattr(e, "self_device_time_total", 0)

        def _cuda_total_us(e):
            if hasattr(e, "cuda_time_total"):
                return e.cuda_time_total
            return getattr(e, "device_time_total", 0)

        total_self_cuda = sum(_self_cuda_us(e) for e in key_avgs)
        total_self_cpu = sum(e.self_cpu_time_total for e in key_avgs)

        rows = []
        for e in key_avgs:
            self_cuda = _self_cuda_us(e)
            cuda_total = _cuda_total_us(e)

            cuda_pct = (
                self_cuda / total_self_cuda * 100
                if total_self_cuda > 0 else 0.0
            )

            cpu_pct = (
                e.self_cpu_time_total / total_self_cpu * 100
                if total_self_cpu > 0 else 0.0
            )

            if cuda_pct >= 0.1 or cpu_pct >= 1.0:
                rows.append({
                    "name": e.key,
                    "input_shapes": e.input_shapes,
                    "self_cuda_ms": self_cuda / 1000,
                    "cuda_total_ms": cuda_total / 1000,
                    "cuda_pct": cuda_pct,
                    "self_cpu_ms": e.self_cpu_time_total / 1000,
                    "cpu_total_ms": e.cpu_time_total / 1000,
                    "cpu_pct": cpu_pct,
                    "calls": e.count,
                    "flops": getattr(e, "flops", 0),
                    "self_cpu_memory_bytes": getattr(e, "self_cpu_memory_usage", 0),
                    "self_device_memory_bytes": getattr(e, "self_device_memory_usage", 0),
                    "source_stack": list(getattr(e, "stack", []))[:5],
                })

        rows.sort(
            key=lambda r: (r["self_cuda_ms"], r["self_cpu_ms"]),
            reverse=True,
        )
        rows = rows[:20]

        print()
        print("╠════════════════════════════════════════════════════════════════════════════════════╣")
        print(f"[LOG KERNEL] Top kernel tại step {batch_idx} (epoch {epoch+1})")
        print(
            f"{'Name':<55}"
            f"{'Self CUDA':>12}"
            f"{'CUDA Total':>12}"
            f"{'CUDA %':>9}"
            f"{'CPU %':>9}"
            f"{'Calls':>9}"
        )

        for row in rows:
            print(
                f"{row['name'][:55]:<55}"
                f"{row['self_cuda_ms']:>9.2f}ms"
                f"{row['cuda_total_ms']:>9.2f}ms"
                f"{row['cuda_pct']:>8.2f}%"
                f"{row['cpu_pct']:>8.2f}%"
                f"{row['calls']:>9}"
            )

        device_info = {"type": device.type}
        if device.type == "cuda":
            device_info.update({
                "name": torch.cuda.get_device_name(device),
                "capability": list(torch.cuda.get_device_capability(device)),
            })

        report = {
            "format": "compact-kernel-profile-v1",
            "torch_version": torch.__version__,
            "device": device_info,
            "epoch": epoch + 1,
            "batch_idx": batch_idx,
            "display_step": batch_idx + 1,
            "totals": {
                "self_cuda_ms": total_self_cuda / 1000,
                "self_cpu_ms": total_self_cpu / 1000,
            },
            "filters": {
                "min_cuda_pct": 0.1,
                "min_cpu_pct": 1.0,
                "max_rows": 20,
            },
            "operators": rows,
        }

        trace_file = self.out_dir / (
            f"torch_profile_epoch{epoch+1}_step{batch_idx}_summary.json"
        )
        trace_file.write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        print("────────────────────────────────────────────────────────────────────────────────────")
        print(f"Compact profiler JSON: {trace_file}")
        print("╠════════════════════════════════════════════════════════════════════════════════════╣")

        self.reported = True
