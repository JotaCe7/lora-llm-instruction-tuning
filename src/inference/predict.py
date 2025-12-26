from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.training.format_dataset import format_prompt
from src.config import MODEL_NAME

model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def predict(text: str) -> str:
    prompt = format_prompt({"text": text, "label": ""})["input_text"]
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=5)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


if __name__ == "__main__":
    print(predict("I really love this new subscription."))