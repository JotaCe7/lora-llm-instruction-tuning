from src.inference.utils import load_lora_model, predict

if __name__ == "__main__":
    lora_model, tokenizer = load_lora_model()  
    text = "I was charged twice for my subscription."
    print(predict(text, lora_model, tokenizer))