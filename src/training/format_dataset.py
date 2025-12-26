from .prompt import SYSTEM_PROMPT

def format_prompt(example: dict) -> dict:
    """
    Returns:
        {
            "input_text": str,
            "target_text": str | None
        }
    """
    prompt = f"""{SYSTEM_PROMPT}

Customer message:
{example['text']}
"""
    return {
        "input_text": prompt,
        "target_text": example["label"]
    }