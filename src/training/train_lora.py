from datasets import load_dataset
import torch
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)

from src.training.lora import load_lora_model
from src.training.format_dataset import format_prompt
from src.training.tokenize_dataset import tokenize_function

# Training constants
TRAIN_PATH = "data/raw/train.jsonl"
VAL_PATH = "data/raw/val.jsonl"
OUTPUT_DIR = "outputs/lora"
BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 2
LR = 2e-4
EPOCHS = 10

def main():
    model, tokenizer = load_lora_model()

    dataset = load_dataset(
        "json",
        data_files={"train": TRAIN_PATH, "validation": VAL_PATH},
    )

    formatted = dataset.map(
        format_prompt,
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
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=LR,
        num_train_epochs=EPOCHS,
        fp16=torch.cuda.is_available(),
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

    model.save_pretrained(f"{OUTPUT_DIR}/final")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")

if __name__ == "__main__":
    main()

