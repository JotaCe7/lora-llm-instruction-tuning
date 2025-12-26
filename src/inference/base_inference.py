from typing import cast

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from src.config import MODEL_NAME
from src.training.format_dataset import format_prompt


_model: PreTrainedModel | None = None
_tokenizer: PreTrainedTokenizer | None = None


def load_base_model() -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    global _model, _tokenizer

    if _model is None or _tokenizer is None:
        _tokenizer = cast(PreTrainedTokenizer, AutoTokenizer.from_pretrained(MODEL_NAME))
        _model = cast(PreTrainedModel, AutoModelForSeq2SeqLM.from_pretrained(
            MODEL_NAME,
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        ))
        _model.eval()

    return _model, _tokenizer


@torch.no_grad()
def predict(text: str, max_new_tokens: int = 8) -> str:
    model, tokenizer = load_base_model()

    formatted_prompt = format_prompt({"text": text, "label": ""})["input_text"]
    
    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
    )

    return tokenizer.decode(
        outputs[0],
        skip_special_tokens=True,
    ).strip()

if __name__ == "__main__":
    print(predict("I was charged twice for my subscription."))