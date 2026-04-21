"""
OpenAI-compatible API server for webmind-brain.

Exposes the Brain's ask/generate through standard OpenAI endpoints.
Start: python server.py
Or:    uvicorn server:app --host 0.0.0.0 --port 8000
"""

import os
import sys
import time
import uuid
import json
import asyncio
from typing import Optional, List

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# Add src/ to path so brain imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from brain import Brain
from tools import ToolRouter

# --- Config from env ---
DB_PATH = os.environ.get("BRAIN_DB_PATH", os.path.expanduser("~/nexus-brain"))
PORT = int(os.environ.get("PORT", "8000"))
MODEL_NAME = os.environ.get("MODEL_NAME", "webmind-brain-v1")
WEB_SEARCH = os.environ.get("WEB_SEARCH", "true").lower() in ("true", "1", "yes")

# --- Init brain + tools ---
brain = Brain(db_path=DB_PATH)
tools = ToolRouter(web_search=WEB_SEARCH)

# --- FastAPI app ---
app = FastAPI(title="Webmind Brain API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic models ---

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 30
    stream: Optional[bool] = False

class CompletionRequest(BaseModel):
    model: Optional[str] = None
    prompt: str
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 30
    stream: Optional[bool] = False

# --- Helpers ---

def extract_user_message(messages: List[ChatMessage]) -> str:
    """Extract the last user message from the conversation."""
    for msg in reversed(messages):
        if msg.role == "user":
            return msg.content
    return messages[-1].content if messages else ""

def count_tokens(text: str) -> int:
    """Rough token count (whitespace split)."""
    return len(text.split())

def make_chat_response(content: str, model: str, prompt_tokens: int) -> dict:
    completion_tokens = count_tokens(content)
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }

def make_completion_response(content: str, model: str, prompt_tokens: int) -> dict:
    completion_tokens = count_tokens(content)
    return {
        "id": f"cmpl-{uuid.uuid4().hex[:12]}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "text": content,
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }

def brain_respond(message: str, max_tokens: int = 30, temperature: float = 0.7) -> str:
    """Ask the brain and optionally generate fluent text."""
    # Intercept: math/code queries handled by eval before brain
    eval_result = tools.on_query(message, brain)
    if eval_result:
        return eval_result

    try:
        ask_result = brain.ask(message)
    except Exception as e:
        return f"Error during reasoning: {type(e).__name__}: {e}"

    strategy = ask_result.get("strategy", "abstain")

    if strategy == "abstain":
        # Try tools (web search) before giving up
        tool_result = tools.on_miss(message, brain)
        if tool_result:
            return tool_result
        return ask_result.get("answer", "I don't know.")

    # Try generate for a more fluent response
    try:
        gen_result = brain.generate(message, max_tokens=max_tokens, temperature=temperature)
        gen_text = gen_result.get("text", "").strip()
        if gen_text:
            return gen_text
    except Exception:
        pass  # Fall back to ask answer

    # Fall back to ask answer
    return ask_result.get("answer", "I don't know.")

# --- Streaming helpers ---

def sse_event(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"

async def stream_chat_response(message: str, model: str, max_tokens: int, temperature: float):
    """Stream tokens one at a time via SSE."""
    chat_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    # First check if brain knows anything
    try:
        ask_result = brain.ask(message)
    except Exception as e:
        ask_result = {"answer": f"Error: {e}", "strategy": "abstain"}
    strategy = ask_result.get("strategy", "abstain")

    if strategy == "abstain":
        text = ask_result.get("answer", "I don't know.")
        # Send as single chunk
        chunk = {
            "id": chat_id, "object": "chat.completion.chunk",
            "created": created, "model": model,
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": text}, "finish_reason": None}],
        }
        yield sse_event(chunk)
    else:
        # Generate and stream token by token
        gen_result = brain.generate(message, max_tokens=max_tokens, temperature=temperature)
        text = gen_result.get("text", "").strip()
        tokens = text.split() if text else [ask_result.get("answer", "I don't know.")]

        # Send role chunk first
        role_chunk = {
            "id": chat_id, "object": "chat.completion.chunk",
            "created": created, "model": model,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        }
        yield sse_event(role_chunk)

        # Stream each token
        for i, token in enumerate(tokens):
            content = token if i == 0 else f" {token}"
            chunk = {
                "id": chat_id, "object": "chat.completion.chunk",
                "created": created, "model": model,
                "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
            }
            yield sse_event(chunk)
            await asyncio.sleep(0.02)  # small delay for streaming feel

    # Final chunk
    done_chunk = {
        "id": chat_id, "object": "chat.completion.chunk",
        "created": created, "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield sse_event(done_chunk)
    yield "data: [DONE]\n\n"

async def stream_completion_response(prompt: str, model: str, max_tokens: int, temperature: float):
    """Stream tokens for legacy completions endpoint."""
    cmpl_id = f"cmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    gen_result = brain.generate(prompt, max_tokens=max_tokens, temperature=temperature)
    text = gen_result.get("text", "").strip()
    tokens = text.split() if text else []

    for i, token in enumerate(tokens):
        content = token if i == 0 else f" {token}"
        chunk = {
            "id": cmpl_id, "object": "text_completion",
            "created": created, "model": model,
            "choices": [{"index": 0, "text": content, "finish_reason": None}],
        }
        yield sse_event(chunk)
        await asyncio.sleep(0.02)

    done_chunk = {
        "id": cmpl_id, "object": "text_completion",
        "created": created, "model": model,
        "choices": [{"index": 0, "text": "", "finish_reason": "stop"}],
    }
    yield sse_event(done_chunk)
    yield "data: [DONE]\n\n"

# --- Endpoints ---

@app.get("/health")
async def health():
    h = brain.health()
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "neurons": h.get("neuron_count", 0),
        "words": h.get("word_count", len(brain._words)),
        "death_risk": h.get("death_risk", 0),
        "rss_mb": h.get("rss_mb", 0),
        "disk_free_gb": h.get("disk_free_gb", 0),
    }

@app.post("/v1/tools/configure")
async def configure_tools(request: Request):
    """Enable/disable tools. Body: {"web_search": bool, "code_eval": bool}"""
    body = await request.json()
    tools.configure(
        web_search=body.get("web_search"),
        code_eval=body.get("code_eval"),
    )
    return {
        "web_search": tools.web_search.enabled,
        "code_eval": tools.code_eval.enabled,
    }

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{
            "id": MODEL_NAME,
            "object": "model",
            "created": 1714000000,
            "owned_by": "webmind",
            "permission": [],
            "root": MODEL_NAME,
            "parent": None,
        }],
    }

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    message = extract_user_message(request.messages)
    model = request.model or MODEL_NAME
    prompt_tokens = sum(count_tokens(m.content) for m in request.messages)
    max_tokens = request.max_tokens or 30
    temperature = request.temperature if request.temperature is not None else 0.7

    if request.stream:
        return StreamingResponse(
            stream_chat_response(message, model, max_tokens, temperature),
            media_type="text/event-stream",
        )

    content = brain_respond(message, max_tokens=max_tokens, temperature=temperature)
    return make_chat_response(content, model, prompt_tokens)

@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    model = request.model or MODEL_NAME
    prompt_tokens = count_tokens(request.prompt)
    max_tokens = request.max_tokens or 30
    temperature = request.temperature if request.temperature is not None else 0.7

    if request.stream:
        return StreamingResponse(
            stream_completion_response(request.prompt, model, max_tokens, temperature),
            media_type="text/event-stream",
        )

    content = brain_respond(request.prompt, max_tokens=max_tokens, temperature=temperature)
    return make_completion_response(content, model, prompt_tokens)

# --- Main ---

if __name__ == "__main__":
    import uvicorn
    print(f"Starting webmind-brain server on port {PORT}")
    print(f"  DB: {DB_PATH}")
    print(f"  Model: {MODEL_NAME}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
