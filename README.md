# LoRA LLM Instruction Tuning

This project demonstrates **parameter-efficient fine-tuning (LoRA)** applied to a
sequence-to-sequence Large Language Model (FLAN-T5) for **instruction-based text classification**.

The goal is to show a **clear behavioral difference** between:
- a **base model** using prompt-only inference
- a **LoRA fine-tuned model** trained on a small, instruction-formatted dataset

---

## What This Project Demonstrates

- Instruction-style dataset formatting (`input_text` / `target_text`)
- LoRA fine-tuning with <1% trainable parameters
- Proper separation of concerns:
  - dataset formatting
  - tokenization
  - training
  - inference
- Behavioral comparison between:
  - base model (prompt-only)
  - LoRA-adapted model
- Real-world debugging around:
  - tokenization mismatches
  - prompt formatting
  - Trainer configuration
  - CPU vs CUDA environments

---

## Model & Task

- **Base model:** `google/flan-t5-small`
- **Task:** Customer support ticket classification
- **Labels:** `billing`, `technical`, `account`, `shipping`, `other`
- **Approach:** Instruction tuning using LoRA adapters

---

## Project Structure

```text
lora-llm-instruction-tuning/
├── data/
│   ├── raw/
│   │   ├── train.jsonl
│   │   └── val.jsonl
│
├── src/
│   ├── config.py
│
│   ├── training/
│   │   ├── dataset.py
│   │   ├── prepare_dataset.py
│   │   ├── format_dataset.py
│   │   ├── prompt.py
│   │   ├── tokenize_dataset.py
│   │   ├── inspect_dataset.py
│   │   ├── lora.py
│   │   ├── train_lora.py
│   │   └── test_lora.py
│
│   └── inference/
│       ├── base_inference.py
│       └── predict.py
│
│   └── evaluation/
│       └── compare_baseline_vs_lora.py
│
├── outputs/
│   └── lora/
│       └── final/
│
├── requirements.txt
└── README.md
````

---

## Dataset Format

Raw datasets are stored as JSONL files:

```
data/raw/train.jsonl
data/raw/val.jsonl

````

Each line follows this format:

```json
{"text": "I was charged twice for my subscription.", "label": "billing"}
{"text": "My internet has been down since yesterday.", "label": "technical"}
{"text": "I can't log into my account.", "label": "account"}
````

---

## Prompt Formatting

During preprocessing, each example is converted into an **instruction-style prompt**
defined in:

```
src/training/prompt.py
```

Example formatted prompt:

```
You are a support ticket classifier.

Given a customer message, output exactly one of the following labels:
- billing
- technical
- account
- shipping
- other

Output ONLY the label.

Customer message:
I was charged twice for my subscription.
```

This formatting is applied consistently during:

* training
* evaluation
* inference

⚠️ Note on Inference Functions

- `predict.py` applies instruction-style formatting automatically.
- `base_inference.predict()` expects a pre-formatted prompt. 
  Use `format_example()` from `src/training/format_dataset.py` 
  to convert raw text into the canonical instruction prompt before passing it to `base_inference.predict()`.


---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Recommended Setup

- **GPU (recommended):**
  - CUDA-enabled PyTorch
  - Faster training
  - fp16 enabled automatically

- **CPU (supported):**
  - Slower training
  - fp16 disabled
  - Suitable for testing and inference

---

## Hardware & Acceleration

This project supports both CPU and GPU execution.

- **Training** is recommended on GPU for speed and fp16 support
- **Inference and evaluation** can run on CPU or GPU

The default configuration enables fp16 training when a compatible GPU is available.
If no GPU is detected, the project will fall back to CPU execution.

### CPU-only Installation Note

If you want to run this project on CPU-only machines, do one of the following to avoid pip attempting to install the CUDA wheel pinned in requirements.txt.

**Option A - preferred:** (edit requirements then install):

* Edit requirements.txt and remove or comment the torch==2.5.1+cu121 line.

* Install dependencies:

```bash
pip install -r requirements.txt
```

* Install a CPU-only PyTorch wheel (if not already installed):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**Option B:** create a temporary requirements file without torch:

```bash
# Unix / WSL
grep -v '^torch' requirements.txt > requirements-no-torch.txt
pip install -r requirements-no-torch.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Windows PowerShell
Select-String -Path requirements.txt -Pattern '^torch' -NotMatch | Set-Content requirements-no-torch.txt
pip install -r requirements-no-torch.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
```


---

### 2. Inspect dataset formatting

To verify prompt construction and labels before training:

```bash
python -m src.training.inspect_dataset
```

---

### 3. Train LoRA adapter

```bash
python -m src.training.train_lora
```

This script:

* Loads the base model defined in `src/config.py`
* Applies LoRA adapters (see `src/training/lora.py`)
* Trains **only LoRA parameters** (~0.4% of total parameters)
* Saves the adapter to:

```
outputs/lora/final/
```

---

### 4. Compare base vs LoRA model

```bash
python -m src.evaluation.compare_baseline_vs_lora
```

This script compares:

* Base model (prompt-only)
* Base model + LoRA adapter

⚠️ Note on Evaluation

For a fair comparison, both the base model and the LoRA-adapted model must be
evaluated using the **same instruction-formatted prompts** used during training.

Earlier versions of the evaluation script used raw text prompts.
The current version applies the same formatting logic from `prompt.py`
to ensure a consistent and unbiased comparison.

---

## Example Output

### Base Model (Prompt-only)

```
Input: My WiFi connection stopped working.
Output: My WiFi connection stopped working.
```

The base model often echoes or paraphrases the input instead of performing classification.

---

### LoRA Fine-tuned Model

```
Input: My WiFi connection stopped working.
Output: technical
```

The LoRA-adapted model learns to output **only the correct label**, even for:

* longer inputs
* paraphrased messages
* instruction-heavy prompts

---

## Why LoRA?

* Trains <1% of total parameters
* Fast and memory-efficient
* No modification of base model weights
* Easy adapter reuse and swapping
* Suitable for limited datasets and hardware

---

## Limitations

* Very small dataset (demo-scale)
* No quantitative metrics (accuracy / F1) yet
* Results demonstrate **capability**, not production readiness

---

## Possible Extensions

* Increase dataset size for better generalization
* Add evaluation metrics (accuracy, F1-score)
* Support multi-label classification
* Expose inference via FastAPI
* Swap base models (T5-large, FLAN-T5, LLaMA adapters)

---