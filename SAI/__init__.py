"""Model SAI và cấu hình sinh; gọi hội thoại bằng ``model.generate_response(tokenizer=..., messages=...)``.

Hàm ``generate`` được giữ cho mã gọi trực tiếp thuật toán sinh của lõi.
"""

from SAI.generate import GenerationConfig, GenerationResult, generate, validate_generation_config
from SAI.modeling import SAIConfig, SAIModel
from SAI.version import COMPONENT_VERSIONS, MODEL_VERSION

__version__ = MODEL_VERSION

__all__ = [
    "COMPONENT_VERSIONS", "GenerationConfig", "GenerationResult", "MODEL_VERSION", "SAIConfig", "SAIModel",
    "__version__", "generate", "validate_generation_config",
]
