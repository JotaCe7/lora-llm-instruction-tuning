from typing import cast

from peft import PeftModel
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from src.config import MODEL_NAME

_model: PreTrainedModel | None = None
_lora_model: PreTrainedModel | None = None
_tokenizer: PreTrainedTokenizer | None = None


def load_base_model() -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    global _model, _tokenizer

    if _model is None or _tokenizer is None:
        _tokenizer = cast(
            PreTrainedTokenizer,
            AutoTokenizer.from_pretrained(MODEL_NAME),
        )
        _model = cast(
            PreTrainedModel,
            AutoModelForSeq2SeqLM.from_pretrained(
                MODEL_NAME,
                dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            ),
        )
        _model.eval()

    return _model, _tokenizer

def load_lora_model(adapter_path: str = "outputs/lora/final") -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    global _lora_model

    if _lora_model is None:
        base_model, _ = load_base_model()
        _lora_model = cast(PreTrainedModel, PeftModel.from_pretrained(base_model, adapter_path))
        _lora_model.eval()

    assert _tokenizer is not None
    return _lora_model, _tokenizer