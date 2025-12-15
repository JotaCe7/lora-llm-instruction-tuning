import json

def load_json(path: str):
    with open(path) as f:
        return [json.loads(line) for line in f]
    
def format_prompt(example: dict) -> str:
    return (
        f"{example['instruction']}\n\n"
        f"Input:\n{example['input']}\n\n"
        f"Answer:"
    )