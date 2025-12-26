from src.inference.utils import load_base_model, predict

if __name__ == "__main__":
    base_model, tokenizer = load_base_model()
    text = "I was charged twice for my subscription."
    print(predict(text, base_model, tokenizer)) 