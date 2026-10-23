"""Release spec: một file YAML trong ``hf_release/releases/`` mô tả một lần phát hành lên Hugging Face.

Nguồn trọng số là MỘT trong hai dạng:
- ``run`` + ``stage``: best.pt của một stage trong thư mục run của lab (có đủ config và lineage).
- ``weights`` + ``model``: một file state_dict bất kỳ (vd. checkpoint dev_2) cùng config model của nó.
"""

from dataclasses import dataclass, field
from foundation_lab.core.config import ConfigError, from_dict, load_with_extends

# Vai trò mà ``SAI.generate`` nhận trong ``messages``.
EXAMPLE_MESSAGE_ROLES = ("system", "user", "model")

@dataclass
class ReleaseSource:
    run: str | None = None          # thư mục run, vd. runs/dev2_35M/20261021-120000
    stage: str | None = None        # stage lấy best.pt, vd. sft2
    weights: str | None = None      # file .pt (state_dict)
    model: str | dict | None = None # tên file trong config/models/ hoặc mapping config model

@dataclass
class ModelCardSpec:
    model_name: str
    example_messages: list[dict[str, str]]  # ví dụ `messages` trong README, vd. [{role: user, content: ...}]
    language: list[str] = field(default_factory=lambda: ["vi"])
    license: str = "apache-2.0"
    tags: list[str] = field(default_factory=lambda: ["custom-code", "vietnamese", "causal-lm"])
    base_model: list[str] = field(default_factory=list)

@dataclass
class ReleaseSpec:
    repo_id: str                    # vd. thongbuind/SAI_35M
    version: str                    # tag trên Hub, khớp git tag, vd. v1.0
    source: ReleaseSource
    card: ModelCardSpec

def load_release_spec(path) -> ReleaseSpec:
    spec = from_dict(ReleaseSpec, load_with_extends(path), str(path))
    source = spec.source
    from_run = source.run is not None or source.stage is not None
    from_weights = source.weights is not None or source.model is not None
    if from_run == from_weights:
        raise ConfigError(f"{path}: source dùng (run + stage) hoặc (weights + model), không trộn hai dạng")
    if from_run and (source.run is None or source.stage is None):
        raise ConfigError(f"{path}: source cần cả run và stage")
    if from_weights and (source.weights is None or source.model is None):
        raise ConfigError(f"{path}: source cần cả weights và model")
    _check_example_messages(spec.card.example_messages, path)
    return spec

def _check_example_messages(messages: list[dict[str, str]], path):
    if not messages:
        raise ConfigError(f"{path}: card.example_messages không được rỗng")
    for i, message in enumerate(messages):
        where = f"{path}: card.example_messages[{i}]"
        if set(message) != {"role", "content"}:
            raise ConfigError(f"{where} cần đúng hai khoá role và content, nhận {sorted(message)}")
        if message["role"] not in EXAMPLE_MESSAGE_ROLES:
            raise ConfigError(f"{where}.role phải là một trong {list(EXAMPLE_MESSAGE_ROLES)}, nhận {message['role']!r}")
    # generate báo lỗi nếu tin nhắn cuối không phải của user.
    if messages[-1]["role"] != "user":
        raise ConfigError(f"{path}: tin nhắn cuối của card.example_messages phải có role user")
