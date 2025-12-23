import torch
from transformers import AutoModelForSeq2SeqLM
from peft import LoraConfig, get_peft_model, TaskType

MODEL_NAME = "google/flan-t5-base"

def load_lora_model():
    """
    Loads the base model and applies LORA adapters.
    Returns a PEFT-wrapped model ready for training.
    """
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        torch_dtype="auto"
    )

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q","v"],
    )

    model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()

    return model