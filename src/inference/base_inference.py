import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

MODEL_NAME = "google/flan-t5-small"

def load_base_model() -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32 if torch.cuda.is_available() else torch.float16,
        device_map="auto",
    )

    model.eval()
    return model, tokenizer


def predict(prompt: str, max_new_tokens: int = 8) -> str:
    model, tokenizer = load_base_model()

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

if __name__ == "__main__":
    test_prompt = """You are a support ticket classifier.

Given a customer message, output exactly one of the following labels:
- billing
- technical
- account
- shipping
- other

Output ONLY the label.

Customer message:
I was charged twice for my subscription.
    """
    print(predict(test_prompt))
