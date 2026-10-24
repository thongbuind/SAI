from foundation_lab.core.registry import Registry

# Cách giải mã một câu trả lời từ model đã nạp.
# Hợp đồng: (model ở chế độ eval, tokenizer SentencePiece, prompt str, SAI.GenerationConfig) -> SAI.GenerationResult.
# Decoder thử nghiệm đăng ký ở đây và được chọn qua ``variants[].decoder``; decoder thắng thí nghiệm mới
# được đưa vào ``SAI/generate.py``.
DECODERS = Registry("decoder")
