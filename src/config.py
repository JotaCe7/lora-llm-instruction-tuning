from peft import LoraConfig, TaskType

MODEL_NAME = "google/flan-t5-small"

GENERATION_CONFIG = {
    "max_new_tokens": 8,
}

LORA_CONFIG = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q","v"],
    )