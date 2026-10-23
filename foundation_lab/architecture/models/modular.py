from dataclasses import dataclass
import torch.nn as nn
from SAI.modeling import SAIConfig, SAIModel
from SAI.rope import RotaryPositionalEmbedding
from foundation_lab.architecture.layers.block import DecoderBlock
from foundation_lab.architecture.registry import NORMS

@dataclass
class SAIModularConfig(SAIConfig):
    attention: str = "gqa"
    attention_kernel: str = "sdpa"
    ffn: str = "swiglu"
    norm: str = "rmsnorm"

class SAIModularModel(SAIModel):
    """Bản lắp ghép của SAI để thí nghiệm: attention, kernel, FFN, norm chọn theo tên trong registry.

    Kế thừa mọi phép tính của model chính thức (forward, prefill, decode, khởi tạo trọng số); chỉ khác
    cách dựng block. Với lựa chọn mặc định, model này phải giống hệt ``SAI.modeling.SAIModel`` từng bit
    (có test): cùng thứ tự tạo module, cùng tên khoá state_dict.
    """

    def __init__(self, config: SAIModularConfig):
        nn.Module.__init__(self)
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
            DecoderBlock(
                config.d_model, config.num_heads, config.num_kv_heads, config.ff_dim, config.dropout,
                attention=config.attention, attention_kernel=config.attention_kernel,
                ffn=config.ffn, norm=config.norm,
            )
            for _ in range(config.num_layers)
        ])
        self.norm = NORMS.get(config.norm)(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight

        self._init_weights()
