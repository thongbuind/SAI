from SAI.generate import GenerationConfig, GenerationResult, generate
from foundation_lab.inference.registry import DECODERS

@DECODERS.register("sai")
def sai_decoder(model, tokenizer, prompt: str, config: GenerationConfig) -> GenerationResult:
    """Decoder chính thức: ``SAI.generate`` (beam, greedy hoặc sample theo ``config.strategy``)."""
    return generate(model, prompt, tokenizer, config=config, return_result=True)
