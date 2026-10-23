from dataclasses import dataclass
import torch
import torch.nn as nn
from .block import DecoderBlock
from .generate import generate
from .rope import RotaryPositionalEmbedding

VERSION = "dev3.0.1"

@dataclass
class SAIConfig:
    vocab_size: int
    d_model: int
    num_heads: int
    num_kv_heads: int
    num_layers: int
    ff_dim: int
    max_seq_len: int
    dropout: float
    pad_token_id: int = 0

class SAIModel(nn.Module):
    def __init__(self, config: SAIConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.num_layers = config.num_layers
        self.pad_token_id = config.pad_token_id
        self.max_seq_len = config.max_seq_len
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        d_k = config.d_model // config.num_heads
        self.rope = RotaryPositionalEmbedding(d_k, config.max_seq_len)
        self.blocks = nn.ModuleList([
            DecoderBlock(config.d_model, config.num_heads, config.num_kv_heads, config.ff_dim, config.dropout)
            for _ in range(config.num_layers)
        ])
        self.norm = nn.RMSNorm(config.d_model, eps=1e-6)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight

        self._init_weights()

    def _init_weights(self):
        std = self.d_model ** -0.5
        nn.init.normal_(self.embed.weight, mean=0.0, std=std)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _prepare_attention(self, input_ids, attention_mask, position_ids, past_length=0):
        if input_ids.ndim != 2 or input_ids.shape[1] == 0:
            raise ValueError("input_ids phải có shape (B, T), T > 0")
        batch_size, length = input_ids.shape
        total_length = past_length + length
        if total_length > self.max_seq_len:
            raise ValueError(f"Độ dài {total_length} vượt max_seq_len={self.max_seq_len}")
        key_mask = None
        if attention_mask is not None:
            if attention_mask.shape != (batch_size, total_length) or attention_mask.device != input_ids.device:
                raise ValueError("attention_mask phải cùng device và có shape (B, past_length + T)")
            if not torch.all((attention_mask == 0) | (attention_mask == 1)):
                raise ValueError("attention_mask chỉ nhận 0 hoặc 1")
            key_mask = attention_mask.bool()
            if not key_mask.any(dim=1).all():
                raise ValueError("Mỗi chuỗi phải có ít nhất một token thật")
        if position_ids is None:
            position_ids = (
                key_mask.long().cumsum(dim=1).sub(1).clamp_min(0)[:, -length:]
                if key_mask is not None else torch.arange(past_length, total_length, device=input_ids.device)
            )
        else:
            if position_ids.shape not in ((length,), (batch_size, length)):
                raise ValueError("position_ids phải có shape (T,) hoặc (B, T)")
            if position_ids.device != input_ids.device or position_ids.dtype != torch.long:
                raise ValueError("position_ids phải cùng device với input_ids và có dtype long")
            if (position_ids < 0).any() or (position_ids >= self.max_seq_len).any():
                raise ValueError("position_ids nằm ngoài giới hạn context")
        cos, sin = self.rope.get_cos_sin(position_ids)
        # Giữ đường SDPA causal cũ khi không cần mask, tránh đổi số học của luồng đã hỗ trợ.
        if (key_mask is None or key_mask.all()) and (past_length == 0 or length == 1):
            return cos, sin, None
        query_positions = torch.arange(past_length, total_length, device=input_ids.device)
        key_positions = torch.arange(total_length, device=input_ids.device)
        causal_mask = key_positions[None, :] <= query_positions[:, None]
        attn_mask = causal_mask[None, None, :, :]  # (1, 1, T, past_length + T)
        if key_mask is not None:
            attn_mask = attn_mask & key_mask[:, None, None, :]
        return cos, sin, attn_mask

    def forward_features(self, input_ids: torch.Tensor, attention_mask=None, has_padding: bool = True, *, position_ids=None) -> torch.Tensor:
        """Trả về (B, T, d_model); mask 1 là token thật, 0 là padding trái hoặc phải.

        Khi không truyền mask, giữ giả định cũ: không padding hoặc padding phải và loss bỏ PAD.
        ``has_padding`` giữ tương thích bên gọi cũ; mask được truyền luôn được tôn trọng.
        """
        cos, sin, attn_mask = self._prepare_attention(input_ids, attention_mask, position_ids)
        x = self.embed(input_ids)

        for block in self.blocks:
            x = block(x, cos, sin, attn_mask)
        return self.norm(x)  # (B, T, d_model)

    def forward(self, input_ids: torch.Tensor, attention_mask=None, has_padding: bool = True, *, position_ids=None) -> torch.Tensor:
        x = self.forward_features(input_ids, attention_mask, has_padding, position_ids=position_ids)
        return self.lm_head(x)

    def forward_features_cached(self, input_ids: torch.Tensor, attention_mask=None, position_ids=None, past_key_values=None) -> tuple[torch.Tensor, list]:
        """Trả hidden state sau norm (B, T, D) và KV tensor; cho phép chỉ chiếu logits cần dùng."""
        past_length = 0
        if past_key_values is not None:
            if len(past_key_values) != self.num_layers:
                raise ValueError("Số tầng KV cache không khớp model")
            past_length = past_key_values[0][0].shape[2]
            expected = (input_ids.shape[0], self.num_kv_heads, past_length, self.d_model // self.num_heads)
            for key, value in past_key_values:
                if key.shape != expected or value.shape != expected:
                    raise ValueError("KV cache phải có shape (B, H_kv, past_length, head_dim) đồng nhất")
        cos, sin, attn_mask = self._prepare_attention(input_ids, attention_mask, position_ids, past_length)
        x = self.embed(input_ids)
        present = []
        for index, block in enumerate(self.blocks):
            past = past_key_values[index] if past_key_values is not None else None
            if attn_mask is None and past is None:
                # Hợp đồng ba tham số vẫn được các block trong lab sử dụng.
                x, kv = block.prefill(x, cos, sin)
            else:
                x, kv = block.prefill(x, cos, sin, attn_mask=attn_mask, past_kv=past)
            present.append(kv)
        return self.norm(x), present

    def forward_cached(self, input_ids: torch.Tensor, attention_mask=None, position_ids=None, past_key_values=None) -> tuple[torch.Tensor, list]:
        """Trả logits (B, T, V) và KV đầy đủ; cache chỉ chứa tensor, không phụ thuộc thư viện HF."""
        features, present = self.forward_features_cached(input_ids, attention_mask, position_ids, past_key_values)
        return self.lm_head(features), present

    def init_cache(self, batch_size: int, max_gen_len: int, device: torch.device) -> list[list[torch.Tensor]]:
        d_k = self.d_model // self.num_heads
        return [
            [
                torch.empty(batch_size, self.num_kv_heads, max_gen_len, d_k, device=device, dtype=self.embed.weight.dtype),
                torch.empty(batch_size, self.num_kv_heads, max_gen_len, d_k, device=device, dtype=self.embed.weight.dtype),
            ]
            for _ in self.blocks
        ]

    def prefill(self, input_ids: torch.Tensor, kv_cache=None, *, attention_mask=None, position_ids=None) -> tuple[torch.Tensor, list]:
        B, T = input_ids.shape
        features, present = self.forward_features_cached(input_ids, attention_mask, position_ids)
        new_cache = []
        for i, kv in enumerate(present):
            if kv_cache is not None:
                kv_cache[i][0][:B, :, :T, :] = kv[0]
                kv_cache[i][1][:B, :, :T, :] = kv[1]
                new_cache.append(kv_cache[i])
            else:
                new_cache.append(kv)
        last_positions = torch.full((B,), T - 1, device=input_ids.device, dtype=torch.long)
        if attention_mask is not None:
            positions = torch.arange(T, device=input_ids.device).expand(B, T)
            last_positions = positions.masked_fill(~attention_mask.bool(), -1).max(dim=1).values
        # Chỉ chiếu token cuối qua lm_head, không tạo logits (B, T, vocab_size) cho prompt dài.
        return self.lm_head(features[torch.arange(B, device=input_ids.device), last_positions]), new_cache

    def decode_step(self, token_ids: torch.Tensor, kv_cache, cache_len: int, *, attention_mask=None, position_ids=None) -> torch.Tensor:
        token_ids = token_ids.view(-1, 1)
        x = self.embed(token_ids)
        cos, sin, attn_mask = self._prepare_attention(token_ids, attention_mask, position_ids, cache_len)

        for block, kv in zip(self.blocks, kv_cache):
            if attn_mask is None:
                x = block.forward_with_cache(x, kv, cache_len, cos, sin)
            else:
                x = block.forward_with_cache(x, kv, cache_len, cos, sin, attn_mask=attn_mask)
        return self.lm_head(self.norm(x))[:, 0, :]

    def generate_response(self, user_input, tokenizer, **kwargs):
        return generate(self, user_input, tokenizer, **kwargs)
