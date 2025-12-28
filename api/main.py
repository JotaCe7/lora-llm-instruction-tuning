import logging
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import PreTrainedModel, PreTrainedTokenizer

from src.inference.utils import load_base_model, load_lora_model, predict

base_model: Optional[PreTrainedModel] = None
base_tokenizer: Optional[PreTrainedTokenizer] = None
lora_model: Optional[PreTrainedModel] = None
lora_tokenizer: Optional[PreTrainedTokenizer] = None


app = FastAPI(
    title="LoRA LLM Instruction tuning API",
    description="API for customer support ticket classification using base FLAN-T5 and LoRA-adapted model.",
    version="1.0.0",
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
logger = logging.getLogger(__name__)


@app.on_event("startup")
def startup_event():
    global base_model, base_tokenizer, lora_model, lora_tokenizer

    # Load models once at startup
    logger.info("Loading base model...")
    base_model, base_tokenizer = load_base_model()
    logger.info("Loading LoRA model...")
    lora_model, lora_tokenizer = load_lora_model(adapter_path="outputs/lora/final")
    logger.info("Models loaded successfully")


class PredictionRequest(BaseModel):
    text: str = Field(..., example="I was billed twice this month.")
    max_new_tokens: Optional[int] = Field(
        None,
        description="Optional override for the maximum number of tokens to generate",
    )
    

class PredictionResponse(BaseModel):
    prediction: str = Field(..., example="billing")


@app.post("/predict/base", response_model=PredictionResponse, summary="Predict with base FLAN-T5 model")
def predict_base(request: PredictionRequest):
    """
    Generate a prediction using the base FLAN-T5 model.
    """
    logger.info("Base prediction request received")
    generation_kwargs = {}
    if request.max_new_tokens is not None:
        generation_kwargs["max_new_tokens"] = request.max_new_tokens
    output = predict(request.text, base_model, base_tokenizer, generation_kwargs=generation_kwargs)
    return {"prediction": output}


@app.post("/predict/lora", response_model=PredictionResponse, summary="Predict with LoRA-adapted model")
def predict_lora(request: PredictionRequest):
    """
    Generate a prediction using the LoRA-adapted model.
    """
    logger.info("LoRA prediction request received")
    generation_kwargs = {}
    if request.max_new_tokens:
        generation_kwargs["max_new_tokens"] = request.max_new_tokens
    output = predict(request.text, lora_model, lora_tokenizer, generation_kwargs=generation_kwargs)
    return {"prediction": output}


@app.get("/health", summary="Health check endpoint")
def health_check():
    """
    Simple health check endpoint to verify that the API is running.
    """
    return {"status": "ok"}

