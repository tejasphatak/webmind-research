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

# --- Init brain (auto-detect CSR > LMDB > SQLite) ---
def _load_brain(db_path):
    csr_path = os.path.join(db_path, 'cooc_csr', 'indptr.bin')
    lmdb_path = os.path.join(db_path, 'brain.lmdb')
    if os.path.exists(csr_path) and os.path.exists(lmdb_path):
        from brain_csr_adapter import BrainCSR
        return BrainCSR(db_path=db_path)
    if os.path.exists(lmdb_path):
        from brain_lmdb_adapter import BrainLMDB
        return BrainLMDB(db_path=db_path)
    return Brain(db_path=db_path)

brain = _load_brain(DB_PATH)
tools = ToolRouter(web_search=WEB_SEARCH)

# Bootstrap identity Q→A if not already in the Q→A map
if hasattr(brain, 'correct') and hasattr(brain, '_qa_map'):
    _identity = [
        ("hi", "Hello! I'm Guru, a self-evolving AI. Ask me anything — I learn from every conversation."),
        ("hello", "Hello! I'm Guru. Ask me anything — I learn from every conversation."),
        ("hey", "Hey! I'm Guru. What would you like to know?"),
        ("who are you", "I'm Guru, a graph-based reasoning engine created by Tejas Phatak. My knowledge is an editable graph that grows every time someone talks to me."),
        ("what are you", "I'm Guru, a self-evolving AI. Unlike traditional models, I learn from every conversation in real-time. No GPU, no gradient descent — just a knowledge graph that gets smarter with use."),
        ("what can you do", "I can answer questions about science, history, math, programming, and more. If I get something wrong, correct me and I'll remember."),
        ("help", "Ask me any question! I know about science, history, math, programming, physics, biology, and more. If I don't know something, I'll say so honestly."),
        ("thank you", "You're welcome! Feel free to ask anything else."),
        ("thanks", "You're welcome! Anything else you'd like to know?"),
        ("bye", "Goodbye! Everything you taught me today is saved — I'll be smarter next time we talk."),
    ]
    for q, a in _identity:
        brain.protect(q, a)
    print(f"  Identity: {len(_identity)} protected Q→A pairs")

# --- FastAPI app ---
app = FastAPI(title="Guru API", version="1.0.0")

from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

STATIC_DIR = os.path.join(os.path.dirname(__file__), 'static')
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

from starlette.responses import Response

@app.get("/")
async def root():
    chat_path = os.path.join(STATIC_DIR, 'chat.html')
    if os.path.exists(chat_path):
        with open(chat_path, 'r') as f:
            html = f.read()
        return Response(
            content=html,
            media_type="text/html",
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0",
            },
        )
    return HTMLResponse("<h1>Guru API</h1><p>Use /v1/chat/completions or /health</p>")

@app.get("/status", response_class=HTMLResponse)
async def status_page():
    import platform, psutil, pathlib
    h = brain.health()
    proc = psutil.Process()
    mem = proc.memory_info()
    cpu_name = platform.processor() or "Unknown"
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if "model name" in line:
                    cpu_name = line.split(":")[1].strip()
                    break
    except Exception:
        pass
    cores = os.cpu_count() or 0
    total_ram = psutil.virtual_memory().total / (1024**3)
    avail_ram = psutil.virtual_memory().available / (1024**3)
    db_path = os.path.expanduser(DB_PATH)
    lmdb_size = sum(f.stat().st_size for f in pathlib.Path(os.path.join(db_path, "brain.lmdb")).iterdir()) / (1024**3) if os.path.exists(os.path.join(db_path, "brain.lmdb")) else 0
    csr_path = os.path.join(db_path, "cooc_csr")
    csr_size = sum(f.stat().st_size for f in pathlib.Path(csr_path).iterdir()) / (1024**2) if os.path.exists(csr_path) else 0

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Guru — Status</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, monospace; background: #0d1117; color: #c9d1d9; padding: 2rem; }}
  h1 {{ color: #58a6ff; margin-bottom: 0.5rem; font-size: 1.8rem; }}
  .subtitle {{ color: #8b949e; margin-bottom: 2rem; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1.5rem; max-width: 900px; }}
  .card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 1.5rem; }}
  .card h2 {{ color: #58a6ff; font-size: 1rem; margin-bottom: 1rem; text-transform: uppercase; letter-spacing: 0.05em; }}
  .row {{ display: flex; justify-content: space-between; padding: 0.4rem 0; border-bottom: 1px solid #21262d; }}
  .row:last-child {{ border-bottom: none; }}
  .label {{ color: #8b949e; }}
  .value {{ color: #f0f6fc; font-weight: 600; }}
  .green {{ color: #3fb950; }}
  .yellow {{ color: #d29922; }}
  .footer {{ margin-top: 2rem; color: #484f58; font-size: 0.85rem; }}
  a {{ color: #58a6ff; text-decoration: none; }}
</style>
</head>
<body>
<div style="background:#da3633;color:#fff;padding:1rem 1.5rem;border-radius:8px;margin-bottom:1.5rem;font-size:0.95rem;max-width:900px;">
  <strong>RESEARCH PREVIEW</strong> — This is an experimental system for research purposes only. Do not submit personal, confidential, or sensitive information. All inputs may be stored in the knowledge graph. No warranties. No SLA. May be taken offline without notice.
</div>
<h1>Guru</h1>
<p class="subtitle">Self-evolving graph reasoning engine — live on <a href="https://guru.webmind.sh">guru.webmind.sh</a></p>
<div class="grid">
  <div class="card">
    <h2>Infrastructure</h2>
    <div class="row"><span class="label">CPU</span><span class="value">{cpu_name}</span></div>
    <div class="row"><span class="label">Cores</span><span class="value">{cores}</span></div>
    <div class="row"><span class="label">Architecture</span><span class="value">{platform.machine()}</span></div>
    <div class="row"><span class="label">RAM</span><span class="value">{total_ram:.1f} GB total / {avail_ram:.1f} GB free</span></div>
    <div class="row"><span class="label">GPU</span><span class="value">None</span></div>
    <div class="row"><span class="label">OS</span><span class="value">{platform.system()} {platform.release()[:20]}</span></div>
  </div>
  <div class="card">
    <h2>Brain</h2>
    <div class="row"><span class="label">Neurons</span><span class="value">{h.get('neuron_count', h.get('word_count', 0)):,}</span></div>
    <div class="row"><span class="label">Words</span><span class="value">{len(brain._words):,}</span></div>
    <div class="row"><span class="label">Edges</span><span class="value">~{getattr(brain, '_csr', None) and brain._csr.nnz or 0:,}</span></div>
    <div class="row"><span class="label">Q&A pairs</span><span class="value">{len(brain._qa_map):,}</span></div>
    <div class="row"><span class="label">Status</span><span class="value green">{"Alive" if h.get('death_risk', 0) == 0 else "Degraded"}</span></div>
  </div>
  <div class="card">
    <h2>Process</h2>
    <div class="row"><span class="label">RSS</span><span class="value">{mem.rss / (1024**2):.0f} MB</span></div>
    <div class="row"><span class="label">CPU %</span><span class="value">{proc.cpu_percent():.1f}%</span></div>
    <div class="row"><span class="label">LMDB</span><span class="value">{lmdb_size:.2f} GB</span></div>
    <div class="row"><span class="label">CSR</span><span class="value">{csr_size:.0f} MB</span></div>
    <div class="row"><span class="label">Disk free</span><span class="value">{h.get('disk_free_gb', 0):.0f} GB</span></div>
  </div>
  <div class="card">
    <h2>Architecture</h2>
    <div class="row"><span class="label">Model</span><span class="value">{MODEL_NAME}</span></div>
    <div class="row"><span class="label">Engine</span><span class="value">Co-occurrence graph + convergence loop</span></div>
    <div class="row"><span class="label">Tier 1</span><span class="value">Q&A direct lookup (&lt;1ms)</span></div>
    <div class="row"><span class="label">Tier 2</span><span class="value">Sparse convergence (~250ms)</span></div>
    <div class="row"><span class="label">Learning</span><span class="value">Every API call trains the brain</span></div>
    <div class="row"><span class="label">Training</span><span class="value">No GPU. No gradient descent.</span></div>
  </div>
</div>
<p class="footer">
  Guru by <a href="https://webmind.sh">Webmind Research</a> &middot;
  <a href="https://huggingface.co/tejadabheja/guru">HuggingFace</a> &middot;
  <a href="https://github.com/tejasphatak/webmind-research">GitHub</a> &middot;
  <a href="/">Chat</a>
</p>
</body>
</html>"""
    return HTMLResponse(html)

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
    session_id: Optional[str] = None  # client can pass; auto-generated if missing

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

def extract_context(messages: List[ChatMessage]) -> str:
    """Build context from full conversation (last 10 messages)."""
    parts = []
    for msg in messages[-10:]:
        parts.append(msg.content)
    return ' '.join(parts)

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

# Words to ignore when checking question-answer relevance
FILTER_WORDS = frozenset({
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'what', 'who',
    'how', 'when', 'where', 'which', 'why', 'do', 'does', 'did', 'can', 'could',
    'will', 'would', 'should', 'of', 'in', 'to', 'for', 'with', 'on', 'at', 'by',
    'from', 'and', 'or', 'but', 'not', 'it', 'its', 'this', 'that', 'me', 'tell',
    'about', 'please', 'know', 'explain', 'describe', 'current', 'today',
})

# --- Session WAL ---
# Per-session co-occurrence edges. Dies when session ends.
# Only the teach API and explicit corrections write to global LMDB.
_session_wals = {}  # session_id → {(i,j): weight}
_SESSION_MAX = 1000  # max concurrent sessions

def _get_session_wal(session_id: str) -> dict:
    if session_id not in _session_wals:
        if len(_session_wals) >= _SESSION_MAX:
            # Evict oldest
            oldest = next(iter(_session_wals))
            del _session_wals[oldest]
        _session_wals[session_id] = {}
    return _session_wals[session_id]

def brain_respond(message: str, messages: List[ChatMessage] = None, session_id: str = None, max_tokens: int = 30, temperature: float = 0.7) -> dict:
    """Ask the brain with session-scoped context.

    Returns dict with: answer, source, strategy, hops, confidence.
    Session WAL: conversation edges stay local to the session.
    Global LMDB: only written by /v1/teach, /v1/correct, /v1/protect + web search results.
    """
    session_wal = _get_session_wal(session_id) if session_id else {}

    # Teach prior messages to SESSION WAL only — not global graph
    if messages and len(messages) > 1:
        for msg in messages[:-1]:
            edges = brain.teach_session(msg.content)
            for k, v in edges.items():
                session_wal[k] = session_wal.get(k, 0.0) + v

    query = message

    # Intercept: math/code queries handled by eval before brain
    eval_result = tools.on_query(message, brain)
    if eval_result:
        return {"answer": eval_result, "source": "compute", "strategy": "math", "hops": 0, "confidence": 1.0}

    try:
        ask_result = brain.ask(query, auto_learn=False, session_edges=session_wal)
    except Exception as e:
        error_msg = f"Error during reasoning: {type(e).__name__}: {e}"
        brain.teach(f"query failed {message} error {type(e).__name__} {e}", confidence=0.1)
        return {"answer": error_msg, "source": "error", "strategy": "error", "hops": 0, "confidence": 0.0}

    strategy = ask_result.get("strategy", "abstain")
    hops = ask_result.get("convergence_rounds", 0)
    confidence = ask_result.get("confidence", 0.0)

    # Filter garbage from convergence
    answer = ask_result.get("answer", "")
    is_garbage = (
        not answer
        or "may refer to" in answer
        or (strategy != "qa_direct" and answer.strip().endswith("?") and len(answer) > 50)
    )

    # For non-direct answers, check if the answer is actually relevant to the question
    # If none of the question's key words appear in the answer, it's probably a wrong-topic match
    if not is_garbage and strategy != "qa_direct":
        import re
        q_words = set(re.findall(r'[a-z]+', message.lower())) - FILTER_WORDS
        a_words = set(re.findall(r'[a-z]+', answer.lower()))
        overlap = q_words & a_words
        if len(q_words) >= 2 and len(overlap) < 2:
            is_garbage = True

    if strategy != "abstain" and not is_garbage:
        return {"answer": answer, "source": "brain", "strategy": strategy, "hops": hops, "confidence": confidence}

    # Brain doesn't know — try web search
    web_result = tools.on_miss(message, brain)
    if web_result:
        # Teach the brain so it knows next time
        brain.correct(message, web_result)
        return {"answer": web_result, "source": "web", "strategy": "web_search", "hops": 0, "confidence": 0.7}

    # Nothing found anywhere
    return {
        "answer": "I don't know the answer to that yet. Can you teach me? Just tell me the answer and I'll remember it.",
        "source": "none",
        "strategy": "abstain",
        "hops": hops,
        "confidence": 0.0,
    }

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

    sid = request.session_id or str(uuid.uuid4())
    result = brain_respond(message, messages=request.messages, session_id=sid, max_tokens=max_tokens, temperature=temperature)
    response = make_chat_response(result["answer"], model, prompt_tokens)
    # Add Guru metadata to response
    response["guru"] = {
        "source": result["source"],
        "strategy": result["strategy"],
        "hops": result["hops"],
        "confidence": result["confidence"],
    }
    return response

class TeachRequest(BaseModel):
    sentences: List[str] = Field(default=None, description="Sentences to teach (bulk)")
    sentence: Optional[str] = Field(default=None, description="Single sentence to teach")
    confidence: Optional[float] = 0.5

class CorrectRequest(BaseModel):
    question: str
    answer: str

class ProtectRequest(BaseModel):
    question: str
    answer: str

@app.post("/v1/teach")
async def teach(request: TeachRequest):
    """Teach the brain new knowledge. Accepts single sentence or bulk list."""
    taught = []
    if request.sentences:
        for s in request.sentences:
            brain.teach(s, confidence=request.confidence)
            taught.append(s)
    elif request.sentence:
        brain.teach(request.sentence, confidence=request.confidence)
        taught.append(request.sentence)
    else:
        return {"error": "Provide 'sentence' or 'sentences'"}
    return {"taught": len(taught), "status": "ok"}

@app.post("/v1/correct")
async def correct(request: CorrectRequest):
    """Correct a Q→A pair. Brain learns the mapping."""
    brain.correct(request.question, request.answer)
    return {"question": request.question, "status": "ok"}

@app.post("/v1/protect")
async def protect(request: ProtectRequest):
    """Set a protected Q→A pair (cannot be overwritten)."""
    brain.protect(request.question, request.answer)
    return {"question": request.question, "status": "ok"}

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

    result = brain_respond(request.prompt, max_tokens=max_tokens, temperature=temperature)
    content = result["answer"]
    return make_completion_response(content, model, prompt_tokens)

# --- Main ---

if __name__ == "__main__":
    import uvicorn
    print(f"Starting webmind-brain server on port {PORT}")
    print(f"  DB: {DB_PATH}")
    print(f"  Model: {MODEL_NAME}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
