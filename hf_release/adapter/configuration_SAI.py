"""Chuyển cấu hình SAI sang Transformers, giữ một nguồn cho các trường kiến trúc."""

from dataclasses import asdict, fields
from typing import TypeVar
from transformers import PreTrainedConfig

CoreConfig = TypeVar("CoreConfig")


class SAIConfig(PreTrainedConfig):
    model_type = "SAI"

    def __init__(
        self,
        vocab_size=10000,
        d_model=512,
        num_heads=8,
        num_kv_heads=8,
        num_layers=8,
        ff_dim=2048,
        max_seq_len=10240,
        dropout=0.05,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=True,
        sai_version=None,
        component_versions=None,
        use_cache=True,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.hidden_size = d_model
        self.num_heads = num_heads
        self.num_attention_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_key_value_heads = num_kv_heads
        self.num_layers = num_layers
        self.num_hidden_layers = num_layers
        self.ff_dim = ff_dim
        self.intermediate_size = ff_dim
        self.max_seq_len = max_seq_len
        self.max_position_embeddings = max_seq_len
        self.dropout = dropout
        self.sai_version = sai_version
        self.component_versions = component_versions
        self.use_cache = use_cache
        if not tie_word_embeddings:
            raise ValueError("SAI yêu cầu tie_word_embeddings=True")
        # Các tên chuẩn HF là bí danh; hai giá trị mâu thuẫn phải bị từ chối.
        for alias in ("hidden_size", "num_attention_heads", "num_key_value_heads", "num_hidden_layers", "intermediate_size", "max_position_embeddings"):
            if alias in kwargs and kwargs.pop(alias) != getattr(self, alias):
                raise ValueError(f"Config mâu thuẫn ở {alias}")
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @classmethod
    def from_core_config(cls, config: CoreConfig, *, version: str, component_versions: dict[str, str]) -> "SAIConfig":
        return cls(**asdict(config), sai_version=version, component_versions=dict(component_versions))

    def to_core_config(self, config_class: type[CoreConfig]) -> CoreConfig:
        """Truyền class tường minh để dùng được cả trước và sau khi export."""
        return config_class(**{field.name: getattr(self, field.name) for field in fields(config_class)})
