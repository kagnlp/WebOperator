# openai_server.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Optional, Any, Tuple
from web_retriever import WebRetriever
app = FastAPI()

class ChatCompletion(BaseModel):
    examples: List[Tuple[Dict[str, Any], float, str]]

class ChatCompletionRequest(BaseModel):
    model: str
    goal: str
    site: str
    obs: str
    top_k: int
    r_type: str

# -------- Endpoint --------
@app.post("/api/v1/search", response_model=ChatCompletion)
def chat_completions(request: ChatCompletionRequest):
    examples = WebRetriever.search(
        query={"goal": request.goal, "axtree_txt": request.obs},
        website=request.site,
        model_name=request.model,
        top_k=request.top_k,
        retriever_type=request.r_type
    )
    return ChatCompletion(examples=examples)


# Usage: uvicorn server:app --host 0.0.0.0 --port 8000
# 10.141.10.17
