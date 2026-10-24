import logging
import torch
import torch._dynamo
from torch.amp import GradScaler, autocast
from foundation_lab.core.experiment import RuntimeSpec
from foundation_lab.core.log import log_progress

_LIMITED_MAX_AUTOTUNE = {
    "max_autotune": True,
    "max_autotune_gemm_backends": "ATEN,TRITON",
    "max_autotune_gemm_search_space": "DEFAULT",
    "epilogue_fusion": True,
    "triton.cudagraphs": False,
}

def select_device(name: str) -> torch.device:
    """``auto`` chọn theo thứ tự cuda > mps > cpu."""
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("runtime.device=cuda nhưng không có CUDA")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("runtime.device=mps nhưng không có MPS")
    return torch.device(name)

class Runtime:
    """Môi trường thực thi dùng chung cho mọi stage: device, mixed precision, GradScaler, compile.

    Chính sách precision: chỉ bật autocast trên CUDA; BF16 nếu GPU hỗ trợ,
    ngược lại FP16 + GradScaler. GradScaler dùng chung cho các stage của một lần chạy.
    """

    def __init__(self, spec: RuntimeSpec):
        self.spec = spec
        torch._dynamo.config.cache_size_limit = spec.dynamo_cache_size_limit
        logging.getLogger("torch._inductor.select_algorithm").setLevel(logging.ERROR)

        self.device = select_device(spec.device)
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True

        self.amp_enabled = self.device.type == "cuda"
        self.amp_dtype = torch.bfloat16 if self.amp_enabled and torch.cuda.is_bf16_supported() else torch.float16
        self.scaler = GradScaler("cuda", enabled=self.amp_enabled and self.amp_dtype == torch.float16)
        log_progress(f"Sử dụng device: {self.device}")
        if self.amp_enabled:
            precision_name = "BF16" if self.amp_dtype == torch.bfloat16 else "FP16 + GradScaler"
            log_progress(f"CUDA mixed precision: {precision_name}")

    @property
    def profile_vram(self) -> bool:
        return self.spec.profile_vram

    @property
    def profile_kernels(self) -> bool:
        return self.spec.profile_kernels

    def autocast(self):
        return autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.amp_enabled)

    def compile_features(self, model):
        """Trả về hàm ``forward_features`` (đã compile nếu bật) mà vòng train/eval sẽ gọi."""
        if not self.spec.compile:
            return model.forward_features
        if self.spec.limited_max_autotune:
            log_progress("torch.compile: limited max-autotune (ATEN,TRITON; DEFAULT; dynamic=True; CUDA Graph off)")
            return torch.compile(model.forward_features, dynamic=True, options=_LIMITED_MAX_AUTOTUNE)
        return torch.compile(model.forward_features, dynamic=True)
