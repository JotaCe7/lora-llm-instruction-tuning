import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

from src.config import MODEL_NAME

TEST_PROMPTS = [
    "I was billed twice this month.",
    "My WiFi connection stopped working.",
    "I forgot my password and can’t log in.",
    "I want to change my subscription plan.",
]

def load_base_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    model.eval()
    return model, tokenizer

def load_lora_model(adapter_path="outputs/lora/final"):
    base_model, tokenizer = load_base_model()
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    return model, tokenizer

def predict(model, tokenizer, prompt, max_new_tokens=8):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

def main():
    base_model, base_tokenizer = load_base_model()
    lora_model, lora_tokenizer = load_lora_model()

    print("\n=== BASSE MODEL (Prompt-only) ===\n")
    for prompt in TEST_PROMPTS:
        out = predict(base_model, base_tokenizer, prompt)
        print(f"Input: {prompt}")
        print(f"Output: {out}\n")
    
    print("\n=== LORA MODEL (Fine-tuned)===\n")
    for prompt in TEST_PROMPTS:
        out = predict(lora_model, lora_tokenizer, prompt)
        print(f"Input: {prompt}")
        print(f"Output: {out}\n")
    
if __name__ == "__main__":
    main()