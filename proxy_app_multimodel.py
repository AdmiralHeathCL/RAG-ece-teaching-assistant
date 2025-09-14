# proxy_tb_multimodel.py
import os, time, uuid, re, json, asyncio
from typing import List, Optional, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from langchain.chat_models import init_chat_model
from langchain.chains import RetrievalQA

from chatbot import self_query_retriever as tb_retriever, custom_prompt as tb_prompt


OPENROUTER_API_KEY  = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "")
if not OPENROUTER_API_KEY or not OPENROUTER_BASE_URL:
    raise RuntimeError("Set OPENROUTER_API_KEY and OPENROUTER_BASE_URL in your environment.")

TEST_MODELS: Dict[str, str] = {
    "gpt4o":           "openai/gpt-4o",
    "gpt41m":          "openai/gpt-4.1-mini",
    "qwen3_8b":        "qwen/qwen3-8b",
    "qwen3_14b":       "qwen/qwen3-14b",
    "qwen3_32b":       "qwen/qwen3-32b",
}

DEFAULT_ALIAS = "gpt41m"

APP_TITLE = "OpenAI-Compat RAG Proxy (Textbook TA, Multimodel)"
APP_VERSION = "2.0"

app = FastAPI(title=APP_TITLE, version=APP_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def make_llm(model_id: str):
    return init_chat_model(
        model_id,
        model_provider="openai",
        openai_api_base=OPENROUTER_BASE_URL,
        openai_api_key=OPENROUTER_API_KEY,
    )

_LLM_CACHE: Dict[str, object] = {}
_CHAIN_CACHE: Dict[str, RetrievalQA] = {}

def model_key(alias: str) -> str:
    return f"textbook_{alias}"

def get_llm(alias: str):
    if alias not in TEST_MODELS:
        raise HTTPException(status_code=400, detail=f"Unknown LLM alias '{alias}'")
    if alias not in _LLM_CACHE:
        _LLM_CACHE[alias] = make_llm(TEST_MODELS[alias])
    return _LLM_CACHE[alias]

def get_chain(alias: str) -> RetrievalQA:
    key = model_key(alias)
    if key not in _CHAIN_CACHE:
        llm = get_llm(alias)
        _CHAIN_CACHE[key] = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=tb_retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": tb_prompt}
        )
    return _CHAIN_CACHE[key]


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    owned_by: str = "local"

class ModelsResponse(BaseModel):
    object: str = "list"
    data: List[ModelInfo]

class ChatMsg(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMsg]
    stream: Optional[bool] = False
    temperature: float = 0.0
    top_p: float = 1.0
    n: int = 1

class Choice(BaseModel):
    index: int
    message: ChatMsg
    finish_reason: Optional[str]

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Choice]


def clean_thinks(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

def last_user_text(messages: List[ChatMsg]) -> str:
    users = [m.content for m in messages if m.role == "user"]
    if not users:
        raise HTTPException(status_code=400, detail="No user message provided")
    return users[-1].strip()

def resolve_to_alias(model_id: str) -> str:
    """
    Accepts:
      - 'textbook' -> DEFAULT_ALIAS
      - 'textbook_<alias>' -> <alias>
      - '<alias>' (for convenience) -> <alias>
    """
    if model_id == "textbook":
        return DEFAULT_ALIAS
    if model_id.startswith("textbook_"):
        alias = model_id.split("_", 1)[1]
        if alias in TEST_MODELS:
            return alias
        raise HTTPException(status_code=400, detail=f"Unknown model '{model_id}'")
    if model_id in TEST_MODELS:
        return model_id
    raise HTTPException(status_code=400, detail=f"Unknown model '{model_id}'")


@app.get("/v1/models", response_model=ModelsResponse)
def list_models():
    ids = ["textbook"] + [model_key(a) for a in TEST_MODELS.keys()]
    return ModelsResponse(data=[ModelInfo(id=m) for m in ids])

@app.get("/v1/models/{model_id}", response_model=ModelInfo)
def get_model(model_id: str):
    try:
        _ = resolve_to_alias(model_id)
        return ModelInfo(id=model_id)
    except HTTPException as e:
        if e.status_code == 400:
            raise HTTPException(status_code=404, detail="Model not found")
        raise

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(req: ChatCompletionRequest):
    alias = resolve_to_alias(req.model)
    chain = get_chain(alias)

    user_text = last_user_text(req.messages)

    try:
        out = await chain.acall({"query": user_text})
    except Exception as e:
        msg = str(e)
        if "APITimeoutError" in msg or "ConnectTimeout" in msg:
            raise HTTPException(status_code=503, detail="Embedding/LLM backend timeout") from e
        raise

    raw_answer = out.get("result", "")
    answer = clean_thinks(raw_answer)

    # Streaming SSE
    if req.stream:
        async def event_generator():
            for line in answer.splitlines(keepends=True):
                chunk = {
                    "id": f"chatcmpl-{uuid.uuid4().hex}",
                    "object": "chat.completion.chunk",
                    "model": req.model,
                    "choices":[{"delta":{"content":line},"index":0,"finish_reason":None}],
                }
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0.005)
            yield "data: [DONE]\n\n"
        return StreamingResponse(event_generator(), media_type="text/event-stream")

    # Non-streaming
    choice = Choice(index=0, message=ChatMsg(role="assistant", content=answer), finish_reason="stop")
    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex}",
        created=int(time.time()),
        model=req.model,
        choices=[choice],
    )