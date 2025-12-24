from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
from datasets import load_dataset

from src.training.lora import load_lora_model
from src.training.format_dataset import format_example
from src.training.tokenize_dataset import tokenize_function
from src.config import MODEL_NAME

def main():
    model = load_lora_model()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    dataset = load_dataset(
        "json",
        data_files={
            "train": "data/raw/train.jsonl",
            "validation": "data/raw/val.jsonl"
        },
    )

    formatted = dataset.map(
        format_example,
        remove_columns=dataset["train"].column_names,
    )
    
    tokenized = formatted.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model
    )

    trainin_args = TrainingArguments(
        output_dir="outputs2/lora",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        num_train_epochs=10,
        fp16=True,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=trainin_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    model.save_pretrained("outputs2/lora/final")
    tokenizer.save_pretrained("outputs2/lora/final")

if __name__ == "__main__":
    main()

