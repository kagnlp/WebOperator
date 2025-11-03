# openai_server.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Optional, Any

# import ChatCompletion, Choice, ChoiceLogprobs
from huggingface import HuggingFaceModel
from sentence_transformers import SentenceTransformer

app = FastAPI()
loaded_models = {}


class ChatCompletion(BaseModel):
    embeddings: List[List[float]]


# -------- Model Loader --------
def get_or_load_model(model_name: str) -> HuggingFaceModel:
    if model_name not in loaded_models:
        loaded_models[model_name] = SentenceTransformer(model_name)
    return loaded_models[model_name]


class ChatCompletionRequest(BaseModel):
    model: str
    documents: List[str]


# -------- Endpoint --------
@app.post("/api/v1/encode", response_model=ChatCompletion)
def chat_completions(request: ChatCompletionRequest):
    model = get_or_load_model(request.model)
    embeddings = model.encode(request.documents)
    return ChatCompletion(embeddings=embeddings.tolist())


# Usage: uvicorn st_server:app --host 0.0.0.0 --port 8000
# 10.141.10.17
