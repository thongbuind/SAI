"""Adapter giữa model chính thức của SAI và API tải model của Transformers.

File này chỉ chạy trong thư mục đã export (repo trên Hugging Face), nơi ``modeling.py``,
``generate.py``, ``rope.py``... là bản copy của các file cùng tên trong ``SAI/``.
"""

import torch.nn as nn
from transformers import PreTrainedModel

from .configuration_SAI import SAIConfig
from .generate import generate as _sai_generate
from .modeling import SAIModel
from .modeling import SAIConfig as _CoreConfig
from .rope import RotaryPositionalEmbedding

# Khi tải từ thư mục local, Transformers chỉ chép vào cache các file mà file này import trực tiếp,
# không chép các file mà chúng import tiếp. Vì vậy mọi file thành phần mà modeling.py cần
# (trực tiếp hay gián tiếp) phải được import ở đây, dù file này không dùng tên nào trong đó.
from .attention import GroupedQueryAttention  # noqa: F401
from .block import DecoderBlock  # noqa: F401
from .ffn import SwiGLU  # noqa: F401

class SAIForCausalLM(PreTrainedModel):
    config_class = SAIConfig
    base_model_prefix = "SAI"
    main_input_name = "input_ids"
    _tied_weights_keys = {"lm_head.weight": "embed.weight"}

    def __init__(self, config):
        PreTrainedModel.__init__(self, config)

        core = SAIModel(_CoreConfig(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            num_layers=config.num_layers,
            ff_dim=config.ff_dim,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout,
            pad_token_id=config.pad_token_id,
        ))

        # Giữ nguyên tên state_dict của checkpoint gốc, không thêm prefix "model.".
        self.d_model = core.d_model
        self.num_heads = core.num_heads
        self.num_kv_heads = core.num_kv_heads
        self.num_layers = core.num_layers
        self.pad_token_id = core.pad_token_id
        self.max_seq_len = core.max_seq_len
        self.embed = core.embed
        self.rope = core.rope
        self.blocks = core.blocks
        self.norm = core.norm
        self.lm_head = core.lm_head

        # Transformers 5.x tạo model trên device meta khi from_pretrained().
        # RoPE cache non-persistent cần được dựng lại ở lần chạy đầu.
        self._runtime_buffers_need_materialization = self.rope.cos_cached.is_meta
        self.post_init()

    def _init_weights(self, module):
        std = self.d_model ** -0.5
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def _ensure_runtime_buffers(self, device):
        if not self._runtime_buffers_need_materialization:
            return
        head_dim = self.d_model // self.num_heads
        self.rope = RotaryPositionalEmbedding(head_dim, self.max_seq_len).to(device)
        self._runtime_buffers_need_materialization = False

    def get_input_embeddings(self):
        return self.embed

    def set_input_embeddings(self, value):
        self.embed = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

    def forward_features(self, input_ids, attention_mask=None, has_padding=True):
        self._ensure_runtime_buffers(input_ids.device)
        return SAIModel.forward_features(self, input_ids, attention_mask, has_padding)

    def forward(self, input_ids, attention_mask=None, has_padding=True, **kwargs):
        return SAIModel.forward(self, input_ids, attention_mask, has_padding)

    def init_cache(self, batch_size, max_gen_len, device):
        return SAIModel.init_cache(self, batch_size, max_gen_len, device)

    def prefill(self, input_ids, kv_cache=None):
        self._ensure_runtime_buffers(input_ids.device)
        return SAIModel.prefill(self, input_ids, kv_cache)

    def decode_step(self, token_ids, kv_cache, cache_len):
        self._ensure_runtime_buffers(token_ids.device)
        return SAIModel.decode_step(self, token_ids, kv_cache, cache_len)

    def generate_response(self, user_input=None, tokenizer=None, **kwargs):
        if tokenizer is None:
            raise ValueError("tokenizer is required")
        sentencepiece = getattr(tokenizer, "sp_model", tokenizer)
        return _sai_generate(self, user_input, sentencepiece, **kwargs)

    def generate(self, user_input=None, tokenizer=None, **kwargs):
        return self.generate_response(user_input, tokenizer, **kwargs)
