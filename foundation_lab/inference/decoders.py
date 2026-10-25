from SAI.generate import GenerationConfig, GenerationResult
from foundation_lab.inference.registry import DECODERS

@DECODERS.register("sai")
def sai_decoder(model, tokenizer, prompt: str, config: GenerationConfig) -> GenerationResult:
    """Gọi hội thoại bằng ``model.generate_response`` theo ``config.strategy``."""
    return model.generate_response(
        tokenizer=tokenizer, messages=[{"role": "user", "content": prompt}], config=config, return_result=True,
    )
