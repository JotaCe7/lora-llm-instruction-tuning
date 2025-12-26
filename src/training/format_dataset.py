from .prompt import SYSTEM_PROMPT

def format_prompt(example):
    prompt = f"""{SYSTEM_PROMPT}

Customer message:
{example['text']}
"""
    return {
        "input_text": prompt,
        "target_text": example["label"]
    }