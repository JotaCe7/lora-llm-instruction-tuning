from typing import cast, Optional

from peft import PeftModel
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from src.config import GENERATION_CONFIG, MODEL_NAME
from src.training.format_dataset import format_prompt

_base_model: Optional[PreTrainedModel] = None
_lora_model: Optional[PreTrainedModel] = None
_tokenizer: Optional[PreTrainedTokenizer] = None


def load_base_model() -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Load and cache the base model and tokenizer."""
    global _base_model, _tokenizer

    if _base_model is None or _tokenizer is None:
        _tokenizer = cast(
            PreTrainedTokenizer,
            AutoTokenizer.from_pretrained(MODEL_NAME)
        )
        _base_model = cast(
            PreTrainedModel,
            AutoModelForSeq2SeqLM.from_pretrained(
                MODEL_NAME,
                dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            )
        )
        _base_model.eval()

    return _base_model, _tokenizer


def load_lora_model(adapter_path: str = "outputs/lora/final") -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Load and cache the lora model and tokenizer."""
    global _lora_model

    if _lora_model is None:
        base_model, _ = load_base_model()
        _lora_model = cast(PreTrainedModel, PeftModel.from_pretrained(base_model, adapter_path))
        _lora_model.eval()

    assert _tokenizer is not None
    return _lora_model, _tokenizer


@torch.no_grad()
def predict(
    text: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    *,
    generation_kwargs: dict | None = None,
    device: torch.device | None = None,
) -> str:
    """
    Predict output for a single text input.

    Args:
        text: input string
        model: model instance (base or LoRA)
        tokenizer: tokenizer instance
        generation_kwargs: optional overrides for GENERATION_CONFIG. Defaults to None.
        device: optional device to run the model on (CPU/GPU). Defaults to None.

    Returns:
        model-generated string
    """
    formatted_prompt = format_prompt({"text": text, "label": ""})["input_text"]

    target_device = device if device is not None else next(model.parameters()).device

    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(target_device)

    gen_cfg = GENERATION_CONFIG.copy()
    if generation_kwargs:
        gen_cfg.update(generation_kwargs)
    
    outputs = model.generate(**inputs, **gen_cfg)

    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()