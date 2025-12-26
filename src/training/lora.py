from peft import get_peft_model
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.config import LORA_CONFIG, MODEL_NAME

def load_lora_model():
    """
    Loads the base model and applies LORA adapters.
    Returns a PEFT-wrapped model ready for training and tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )

    model = get_peft_model(model, LORA_CONFIG)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.print_trainable_parameters()

    return model, tokenizer