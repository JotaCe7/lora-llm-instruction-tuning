from transformers import AutoTokenizer

MODEL_NAME = "google/flan-t5-base"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(example):
    # Tokenize inpput (prompt)
    model_inputs = tokenizer(
        example["input_text"],
        truncation=True,
        padding="max_length",
        max_length=256,
    )

    # Tokenize output (label)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            example["input_text"],
            truncation=True,
            padding="max_length",
            max_length=8,
        )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs