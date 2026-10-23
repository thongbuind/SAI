import os
import shutil
import sentencepiece as spm
from transformers import PreTrainedTokenizer

class SAITokenizer(PreTrainedTokenizer):
    vocab_files_names = {"vocab_file": "tokenizer.model"}
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file,
        unk_token="[UNK]",
        bos_token="[BOS]",
        eos_token="[EOS]",
        pad_token="[UNK]",
        additional_special_tokens=None,
        **kwargs,
    ):
        self.vocab_file = vocab_file
        self.sp_model = spm.SentencePieceProcessor(model_file=vocab_file)
        if additional_special_tokens is None:
            additional_special_tokens = ["<|im_start|>", "<|im_end|>"]
        super().__init__(
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            additional_special_tokens=additional_special_tokens,
            **kwargs,
        )

    @property
    def vocab_size(self):
        return self.sp_model.get_piece_size()

    def get_vocab(self):
        vocab = {self.sp_model.id_to_piece(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def _tokenize(self, text, **kwargs):
        return self.sp_model.encode(text, out_type=str)

    def _convert_token_to_id(self, token):
        return self.sp_model.piece_to_id(token)

    def _convert_id_to_token(self, index):
        return self.sp_model.id_to_piece(int(index))

    def convert_tokens_to_string(self, tokens):
        return self.sp_model.decode(tokens)

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        result = [self.bos_token_id] + list(token_ids_0)
        if token_ids_1 is not None:
            result += list(token_ids_1)
        return result + [self.eos_token_id]

    def save_vocabulary(self, save_directory, filename_prefix=None):
        os.makedirs(save_directory, exist_ok=True)
        filename = "tokenizer.model"
        if filename_prefix:
            filename = f"{filename_prefix}-{filename}"
        destination = os.path.join(save_directory, filename)
        if os.path.abspath(self.vocab_file) != os.path.abspath(destination):
            shutil.copyfile(self.vocab_file, destination)
        return (destination,)
