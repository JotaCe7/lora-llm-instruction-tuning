import logging
from typing import Optional

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Histogram, generate_latest
from pydantic import BaseModel, Field
from transformers import PreTrainedModel, PreTrainedTokenizer

from src.inference.utils import load_base_model, load_lora_model, predict

base_model: Optional[PreTrainedModel] = None
base_tokenizer: Optional[PreTrainedTokenizer] = None
lora_model: Optional[PreTrainedModel] = None
lora_tokenizer: Optional[PreTrainedTokenizer] = None


REQUEST_COUNT = Counter(
    "api_request_count",
    "Total number of API requests",
    ["endpoint", "method", "status_code"],
)

REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds",
    "Latency of API requests in seconds",
    ["endpoint", "method"],
)

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
    status_code = "500"
    try:
        with REQUEST_LATENCY.labels(endpoint="/predict/base", method="POST").time():
            generation_kwargs = {}
            if request.max_new_tokens is not None:
                generation_kwargs["max_new_tokens"] = request.max_new_tokens
            output = predict(request.text, base_model, base_tokenizer, generation_kwargs=generation_kwargs)
            status_code = "200"
            return {"prediction": output}
    except Exception as e:
        logger.exception(f"Error during base prediction.")
        raise
    finally:
        REQUEST_COUNT.labels(endpoint="/predict/base", method="POST", status_code=status_code).inc()


@app.post("/predict/lora", response_model=PredictionResponse, summary="Predict with LoRA-adapted model")
def predict_lora(request: PredictionRequest):
    """
    Generate a prediction using the LoRA-adapted model.
    """
    logger.info("LoRA prediction request received")
    status_code = "500"
    try:
        with REQUEST_LATENCY.labels(endpoint="/predict/lora", method="POST").time():
            generation_kwargs = {}
            if request.max_new_tokens is not None:
                generation_kwargs["max_new_tokens"] = request.max_new_tokens
            output = predict(request.text, lora_model, lora_tokenizer, generation_kwargs=generation_kwargs)
            status_code = "200"
        return {"prediction": output}
    except Exception as e:
        logger.exception(f"Error during LoRA prediction.")
        raise
    finally:
        REQUEST_COUNT.labels(endpoint="/predict/lora", method="POST", status_code=status_code).inc()


@app.get("/metrics", summary="Prometheus metrics endpoint")
def metrics():
    """
    Endpoint to expose Prometheus metrics.
    """
    return Response(content=generate_latest(), media_type="text/plain")


@app.get("/health", summary="Health check endpoint")
def health_check():
    """
    Simple health check endpoint to verify that the API is running.
    """
    return {"status": "ok"}

