from SAI.modeling import SAIConfig, SAIModel
from foundation_lab.architecture.models.modular import SAIModularConfig, SAIModularModel
from foundation_lab.architecture.registry import MODELS, ModelFamily

CAUSAL_LM = frozenset({"causal_lm", "kv_cache"})

# "SAI": model chính thức trong SAI/modeling.py — dùng cho baseline và mọi run sẽ phát hành.
MODELS.register("SAI")(ModelFamily(SAIConfig, SAIModel, CAUSAL_LM))
# "SAI_modular": cùng phép tính nhưng thành phần chọn theo registry — dùng để thí nghiệm kiến trúc.
MODELS.register("SAI_modular")(ModelFamily(SAIModularConfig, SAIModularModel, CAUSAL_LM))

__all__ = ["SAIConfig", "SAIModel", "SAIModularConfig", "SAIModularModel"]
