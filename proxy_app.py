import time, uuid, re, json, asyncio
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from chatbot import qa_ta

TEXTBOOK_MODEL_ID = "textbook"
MODEL_ALIASES = {
    TEXTBOOK_MODEL_ID
}

app = FastAPI(title="OpenAI-Compat RAG Teaching Assistant", version="1.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    owned_by: str = "local"

class ModelsResponse(BaseModel):
    object: str = "list"
    data: List[ModelInfo]

@app.get("/v1/models", response_model=ModelsResponse)
def list_models():
    return ModelsResponse(data=[ModelInfo(id=m) for m in sorted(MODEL_ALIASES)])

@app.get("/v1/models/{model_id}", response_model=ModelInfo)
def get_model(model_id: str):
    if model_id not in MODEL_ALIASES:
        raise HTTPException(status_code=404, detail="Model not found")
    return ModelInfo(id=model_id)

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

def resolve_model_to_textbook(model_id: str) -> str:
    if model_id in MODEL_ALIASES:
        return TEXTBOOK_MODEL_ID
    raise HTTPException(status_code=400, detail=f"Unknown model '{model_id}'")

async def run_chain(user_msg: str) -> str:
    try:
        out = await qa_ta.acall({"query": user_msg})
    except Exception as e:
        msg = str(e)
        if "APITimeoutError" in msg or "ConnectTimeout" in msg:
            raise HTTPException(status_code=503, detail="Embedding/LLM backend timeout") from e
        raise
    return out.get("result", "")

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    _ = resolve_model_to_textbook(req.model)
    user_text = last_user_text(req.messages)

    raw_answer = await run_chain(user_text)
    answer = clean_thinks(raw_answer)

    if req.stream:
        async def event_generator():
            for line in answer.splitlines(keepends=True):
                chunk = {
                    "id": f"chatcmpl-{uuid.uuid4().hex}",
                    "object": "chat.completion.chunk",
                    "model": req.model,
                    "choices": [
                        {"delta": {"content": line}, "index": 0, "finish_reason": None}
                    ],
                }
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0.005)
            yield "data: [DONE]\n\n"
        return StreamingResponse(event_generator(), media_type="text/event-stream")

    choice = Choice(index=0, message=ChatMsg(role="assistant", content=answer), finish_reason="stop")
    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex}",
        created=int(time.time()),
        model=req.model,
        choices=[choice],
    )

# [TESTING]
@app.get("/")
def root():
    return {"ok": True, "msg": "RAG proxy up", "endpoints": ["/v1/models", "/v1/chat/completions"]}