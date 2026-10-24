"""Phòng Kiến trúc.

Model chính thức nằm ở ``SAI/modeling.py`` (họ ``SAI``). Phòng này chứa bản lắp ghép
``SAI_modular`` và các thành phần để thí nghiệm. Bốn cấp độ thay đổi kiến trúc:

| Cấp độ             | Ví dụ                         | Cách làm                                              |
|--------------------|-------------------------------|-------------------------------------------------------|
| Giá trị cấu hình   | num_layers, num_kv_heads      | Thêm file vào ``architecture/configs/``               |
| Phép tính          | attention kernel tự viết      | Đăng ký vào ``ATTENTION_KERNELS`` (không đổi tham số) |
| Thành phần         | attention/FFN/norm khác       | Đăng ký vào ``ATTENTIONS``/``FFNS``/``NORMS``         |
| Họ mô hình         | encoder–decoder, SAI-VL       | ``ModelFamily`` mới đăng ký vào ``MODELS``            |

Hai cấp "phép tính" và "thành phần" dùng họ ``SAI_modular``. Biến thể được chọn sau thí nghiệm
mới được đưa vào ``SAI/`` như một phiên bản kiến trúc mới.
"""

from SAI.modeling import SAIConfig
from foundation_lab.architecture import kernels, layers, models  # noqa: F401  (đăng ký implementation)
from foundation_lab.architecture.registry import ATTENTION_KERNELS, ATTENTIONS, FFNS, MODELS, NORMS, ModelFamily
from foundation_lab.core.config import ConfigError, from_dict, load_with_extends
from foundation_lab.paths import MODEL_CONFIG_DIR

def resolve_model_config(model: dict) -> tuple[ModelFamily, object]:
    """Trả về (họ mô hình, config đã kiểm tra) từ mapping có khoá ``family``."""
    if not isinstance(model, dict) or "family" not in model:
        raise ConfigError("model: cần mapping có khoá 'family'")
    family = MODELS.get(model["family"])
    fields = {key: value for key, value in model.items() if key != "family"}
    return family, from_dict(family.config_cls, fields, "model")

def official_model_config(model: str | dict, where: str) -> SAIConfig:
    """Config của model chính thức (``family: SAI``) cho nơi dùng model ngoài lab (phát hành, web).

    ``model`` là tên file trong ``config/models/`` (không có đuôi .yaml) hoặc mapping đã đọc sẵn.
    """
    if isinstance(model, str):
        model = load_with_extends(MODEL_CONFIG_DIR / f"{model}.yaml")
    if model.get("family") != "SAI":
        raise ConfigError(f"{where}: chỉ dùng được model chính thức (family: SAI), nhận {model.get('family')!r}")
    fields = {key: value for key, value in model.items() if key != "family"}
    return from_dict(SAIConfig, fields, where)

def build_model(model: dict):
    family, config = resolve_model_config(model)
    return family.build(config)

__all__ = [
    "ATTENTION_KERNELS", "ATTENTIONS", "FFNS", "MODELS", "NORMS", "ModelFamily",
    "build_model", "official_model_config", "resolve_model_config",
]
