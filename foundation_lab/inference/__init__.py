"""Phòng Suy luận: nạp checkpoint để sinh câu trả lời, so sánh các cách giải mã về tốc độ và đầu ra.

Cách giải mã đăng ký vào ``DECODERS`` (``sai`` là ``SAI.generate`` chính thức). Lệnh
``python -m foundation_lab infer <spec>`` chạy một bộ câu hỏi trên nhiều biến thể (decoder + GenerationConfig)
và ghi kết quả vào ``runs/infer/<run_id>/``; xem ``foundation_lab.inference.bench``.
"""

from foundation_lab.inference import decoders  # noqa: F401  (đăng ký implementation)
from foundation_lab.inference.bench import COMMAND, InferenceSpec, VariantSpec, resolve_inference, run_inference, validate_inference
from foundation_lab.inference.loading import load_for_inference
from foundation_lab.inference.registry import DECODERS

__all__ = [
    "COMMAND", "DECODERS", "InferenceSpec", "VariantSpec",
    "load_for_inference", "resolve_inference", "run_inference", "validate_inference",
]
