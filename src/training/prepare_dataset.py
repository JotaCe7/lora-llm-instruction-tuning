from datasets import load_dataset

from src.training.format_dataset import format_prompt
from src.training.tokenize_dataset import tokenize_function

def main():
    dataset = load_dataset(
        "json",
        data_files={
            "train": "data/raw/train.jsonl",
            "validation": "data/raw/val.jsonl",
        },
    )

    dataset = dataset.map(format_prompt)
    print(dataset)
    dataset = dataset.map(
        tokenize_function,
        remove_columns=dataset["train"].column_names,
    )

    print(dataset)
    print(dataset["train"][0])

if __name__=="__main__":
    main()