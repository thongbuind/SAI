"""Adapter Transformers chứa model SAI; thuật toán và KV tensor thuộc về lõi SAI."""

import torch
import torch.nn as nn
from transformers import DynamicCache, GenerationMixin, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from .configuration_SAI import SAIConfig
from .generate import GenerationResult, generate as _sai_generate
from .modeling import SAIConfig as CoreConfig, SAIModel
from .rope import RotaryPositionalEmbedding
# Transformers chép các phụ thuộc trực tiếp vào cache khi tải remote code từ local.
from .attention import GroupedQueryAttention  # noqa: F401
from .block import DecoderBlock  # noqa: F401
from .ffn import SwiGLU  # noqa: F401


class SAIForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = SAIConfig
    base_model_prefix = "model"
    main_input_name = "input_ids"
    _tied_weights_keys = {"model.lm_head.weight": "model.embed.weight"}

    def __init__(self, config: SAIConfig):
        super().__init__(config)
        self.model = SAIModel(config.to_core_config(CoreConfig))
        self._runtime_buffers_need_materialization = self.model.rope.cos_cached.is_meta
        self.post_init()

    def _init_weights(self, module):
        std = self.config.d_model ** -0.5
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def _ensure_runtime_buffers(self, device):
        if self._runtime_buffers_need_materialization:
            self.model.rope = RotaryPositionalEmbedding(
                self.config.d_model // self.config.num_heads, self.config.max_seq_len,
            ).to(device=device, dtype=self.model.embed.weight.dtype)
            self._runtime_buffers_need_materialization = False

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.model.embed = value

    def get_output_embeddings(self) -> nn.Linear:
        return self.model.lm_head

    def set_output_embeddings(self, value: nn.Linear) -> None:
        self.model.lm_head = value

    def forward(
        self, input_ids: torch.Tensor, attention_mask=None, position_ids=None,
        past_key_values=None, use_cache=None, labels=None, return_dict=None,
        output_attentions=False, output_hidden_states=False, logits_to_keep=0, **kwargs,
    ) -> CausalLMOutputWithPast | tuple:
        """Nhận IDs (B, T), mask (B, past + T), labels chưa shift với -100 được bỏ qua."""
        if kwargs:
            raise TypeError(f"Arguments forward chưa hỗ trợ: {sorted(kwargs)}")
        if output_attentions or output_hidden_states:
            raise NotImplementedError("SAI chưa trả attention hoặc hidden state trung gian")
        if not isinstance(logits_to_keep, int) or logits_to_keep < 0:
            raise ValueError("logits_to_keep phải là số nguyên không âm")
        if labels is not None and logits_to_keep:
            raise ValueError("labels yêu cầu logits đầy đủ")
        use_cache = self.config.use_cache if use_cache is None else use_cache
        return_dict = self.config.return_dict if return_dict is None else return_dict
        if past_key_values is not None and not isinstance(past_key_values, DynamicCache):
            raise TypeError("SAI chỉ hỗ trợ DynamicCache của Transformers")
        self._ensure_runtime_buffers(input_ids.device)
        past = None
        past_length = 0
        if past_key_values is not None and past_key_values.get_seq_length() > 0:
            past_length = past_key_values.get_seq_length()
            past = [(entry[0], entry[1]) for entry in past_key_values]
        if use_cache or past is not None:
            features, present = self.model.forward_features_cached(input_ids, attention_mask, position_ids, past)
            cache = None
            if use_cache:
                cache = DynamicCache() if past_key_values is None else past_key_values
                # Cache do người gọi truyền vào phải được cập nhật qua API công khai của HF.
                for index, (key, value) in enumerate(present):
                    cache.update(key[:, :, past_length:, :], value[:, :, past_length:, :], index)
        else:
            features = self.model.forward_features(input_ids, attention_mask, position_ids=position_ids)
            cache = None
        if logits_to_keep:
            features = features[:, -logits_to_keep:, :]
        logits = self.model.lm_head(features)
        loss = None
        if labels is not None:
            if labels.shape != input_ids.shape:
                raise ValueError("labels phải cùng shape input_ids, chưa shift")
            loss = nn.functional.cross_entropy(
                logits[:, :-1].contiguous().view(-1, self.config.vocab_size),
                labels[:, 1:].contiguous().view(-1), ignore_index=-100,
            )
        result = CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=cache)
        return result if return_dict else result.to_tuple()

    def generate_response(self, user_input=None, tokenizer=None, **kwargs) -> str | GenerationResult:
        """Hội thoại riêng của SAI; generate() giữ nguyên hợp đồng Transformers."""
        if tokenizer is None:
            raise ValueError("Cần truyền tokenizer")
        self._ensure_runtime_buffers(self.device)
        sentencepiece = getattr(tokenizer, "sp_model", tokenizer)
        return _sai_generate(self.model, user_input, sentencepiece, **kwargs)
