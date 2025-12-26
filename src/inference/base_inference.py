import torch

from src.config import GENERATION_CONFIG
from src.inference.model_loading import load_base_model
from src.training.format_dataset import format_prompt


@torch.no_grad()
def predict(
    text: str,
    *,
    generation_kargs:dict | None = None) -> str:
    model, tokenizer = load_base_model()

    formatted_prompt = format_prompt({"text": text, "label": ""})["input_text"]
    
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

    gen_cfg = GENERATION_CONFIG.copy()
    if generation_kargs:
        gen_cfg.update(generation_kargs)

    outputs = model.generate(**inputs, **gen_cfg)

    return tokenizer.decode(
        outputs[0],
        skip_special_tokens=True,
    ).strip()

if __name__ == "__main__":
    print(predict("I was charged twice for my subscription."))