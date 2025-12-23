from transformers import AutoTokenizer

from src.config import MODEL_NAME

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(example, tokenizer):
    # Tokenize inpput (prompt)
    model_inputs = tokenizer(
        example["input_text"],
        truncation=True,
        padding="max_length",
        max_length=256,
    )

    # Tokenize output (label)
    labels = tokenizer(
        example["target_text"],
        truncation=True,
        padding="max_length",
        max_length=8,
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs