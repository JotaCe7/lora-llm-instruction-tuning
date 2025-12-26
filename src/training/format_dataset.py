from .prompt import SYSTEM_PROMPT

def format_prompt(example: dict) -> dict[str, str | None]:
    """
    Formats a single dataset example into input/output strings for seq2seq models.

    Args:
        example: dict with keys "text" and "label"

    Returns:
        dict with:
            - "input_text": full prompt for model
            - "target_text": target label string
    """
    prompt = f"""{SYSTEM_PROMPT}

Customer message:
{example['text']}
"""
    return {
        "input_text": prompt,
        "target_text": example["label"]
    }