# openai_server.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import uvicorn
import uuid
import torch
import time
import asyncio

# import ChatCompletion, Choice, ChoiceLogprobs
from openai.types.chat.chat_completion import ChatCompletion, Choice, ChoiceLogprobs
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_token_logprob import ChatCompletionTokenLogprob, TopLogprob
from huggingface import HuggingFaceModel

app = FastAPI()
loaded_models = {}


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Dict[str, str]]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 512
    top_p: Optional[float] = 1.0
    top_k: Optional[int] = 50
    logprobs: Optional[bool] = False
    top_logprobs: Optional[int] = 10  # how many alternative tokens to return

import threading
processing_lock = threading.Lock()

# -------- Model Loader --------
def get_or_load_model(model_name: str) -> HuggingFaceModel:
    with processing_lock:
        if model_name not in loaded_models:
            loaded_models[model_name] = HuggingFaceModel(model_name=model_name)
            loaded_models[model_name].initialize()
        return loaded_models[model_name]

# -------- Endpoint --------
@app.post("/api/v1/chat/completions", response_model=ChatCompletion)
def chat_completions(request: ChatCompletionRequest):
    model: HuggingFaceModel = get_or_load_model(request.model)

    output, scores = model.chat(
        messages=request.messages,
        temperature=request.temperature,
        max_new_tokens=request.max_tokens,
        top_p=request.top_p,
        top_k=request.top_k,
    )

    def safe_logprob(logprob):
        if torch.isinf(logprob) or torch.isnan(logprob):
            # Very small but finite number
            return -1e10
        return logprob.item()

    top_log_probs = []
    full_tokens = model.from_text_to_tokens(output)
    for token_id, logits in zip(full_tokens, scores):
        token_text = model.from_token_to_text(token_id)
        probs = torch.tensor(logits)
        # print(f"Top k: {request.top_logprobs}, Vocab size: {len(probs)}")
        top_probs, top_indices = torch.topk(probs, request.top_logprobs)
        # top_logprobs = [
        #     TopLogprob(
        #         token=model.from_token_to_text(token),
        #         logprob=logprob.item()
        #     ) for token, logprob in zip(top_indices, top_probs)
        # ]

        # print(len(top_probs), end=",")
        # for token, logprob in zip(top_indices, top_probs):
        #     print(f"({model.from_token_to_text(token)}: {safe_logprob(logprob)})", end=", ")

        # print()

        top_logprobs = [
            TopLogprob(token=model.from_token_to_text(token), logprob=safe_logprob(logprob))
            for token, logprob in zip(top_indices, top_probs)
        ]

        top_log_probs.append(
            ChatCompletionTokenLogprob(
                token=token_text,
                logprob=safe_logprob(probs[token_id]),
                top_logprobs=top_logprobs,
            )
        )

    # print(f"Total logs: {len(top_log_probs)}")
    return ChatCompletion(
        id=f"chatcmpl-{uuid.uuid4()}",
        # current unix timestamp in seconds
        created=int(time.time()),
        model=request.model,
        object="chat.completion",
        choices=[
            Choice(
                index=0,
                message=ChatCompletionMessage(role="assistant", content=output),
                logprobs=ChoiceLogprobs(content=top_log_probs),
                finish_reason="stop",
            )
        ],
    )


# Usage: uvicorn hf_server:app --host 0.0.0.0 --port 5000
# python -m uvicorn hf_server:app --host 0.0.0.0 --port 5000