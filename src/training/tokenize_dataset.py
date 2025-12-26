from transformers import BatchEncoding, PreTrainedTokenizer

def tokenize_function(example: dict, tokenizer: PreTrainedTokenizer) -> BatchEncoding:
    """
    Tokenizes input and target text for seq2seq training.

    Args:
        example: dict with keys "input_text" and "target_text"
        tokenizer: a HuggingFace tokenizer instance

    Returns:
        tokenized model inputs including labels
    """
    # Tokenize input (prompt)
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