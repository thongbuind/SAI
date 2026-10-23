from transformers import PreTrainedConfig


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
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
