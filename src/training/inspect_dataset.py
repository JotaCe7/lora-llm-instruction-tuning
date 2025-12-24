from typing import cast
from datasets import load_dataset, DatasetDict

from src.training.format_dataset import format_example

def main():
    dataset = cast(DatasetDict, load_dataset(
        "json",
        data_files={
            "train": "data/raw/train.jsonl",
            "validation": "data/raw/val.jsonl"
        }
    ))

    dataset = dataset.map(format_example)

    # Inspect a few formatted examples
    train_ds = dataset["train"]
    for i in range(3):
        print("-----")
        print(train_ds[i])

if __name__ == "__main__":
    main()
