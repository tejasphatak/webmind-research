# Webmind — Knowledge Engine

**No LLM. No GPU. No cloud. Runs in your browser.**

A retrieval-based knowledge engine with 305K+ Q&A pairs, a 22M-parameter sentence transformer, and a self-learning loop. Ask a question — get an answer in milliseconds. No hallucinations. No subscriptions. No data leaves your device.

**[Try it live → webmind.sh](https://webmind.sh)**

## How it works

```
Your question → Sentence Transformer (encode to 384-dim vector)
             → FAISS Cosine Search (0.45ms, 305K pairs)
             → Best Q&A match → If <tool> tag: execute in sandbox
             → Answer
```

- **Retrieval, not generation.** Answers are pre-stored Q&A pairs matched by semantic similarity. Zero hallucinations — it either knows or says "I don't know."
- **Multi-hop reasoning.** Up to 5 hops of iterative retrieval, building context across Q&A pairs.
- **Self-learning.** When it can't answer, it tries Wikipedia. If it finds something, it stores it locally and knows it next time.
- **Tool execution.** `<tool>` tags run code in a sandboxed Web Worker — math, dates, API calls.
- **P2P knowledge sync.** Instances teach each other via BroadcastChannel and WebRTC.

## Run it yourself

**Browser (single HTML file):**
```bash
# Download from GitHub Release
curl -LO https://github.com/tejasphatak/webmind-research/releases/download/v1.0-data/qa_data.json
curl -LO https://github.com/tejasphatak/webmind-research/releases/download/v1.0-data/qa_embeddings.bin
# Serve
python3 -m http.server 3000
# Open http://localhost:3000/tools/saqt_browser.html
```

**Python server:**
```bash
pip install sentence-transformers faiss-cpu
python3 tools/saqt_training_loop.py  # RLHF training
# Server at Synapse/synapse-src/saqt/serve.py
```

## What's inside

| Component | Size | What it does |
|-----------|------|-------------|
| Sentence transformer | 80MB | Encodes questions to 384-dim vectors (ONNX, runs on CPU) |
| Q&A pairs | 100MB | 305K+ question→answer pairs from Wikipedia, SQuAD, MMLU, etc. |
| Embeddings | 450MB | Pre-computed vectors for all 305K questions |
| **Total** | **630MB** | Runs on phone, tablet, Raspberry Pi, any browser |

## Ethics shield

Safety is structural, not policy. Three layers that can't be removed without breaking the system:

1. **Embedding space warping** — 173K vectors shifted toward safety responses. Harmful queries get gravitationally pulled to refusals.
2. **Steganographic sentinels** — 63 safety pairs disguised as normal knowledge throughout the database.
3. **Integrity hash chain** — SHA-256 over ethics embeddings. Tamper with one → hash breaks → system won't start.

The system follows its principles or shuts down. No middle ground.

## Corporate deployment

```javascript
// Kill switch
window.SAQT_KILL = true;

// Domain allowlist
window.SAQT_ALLOWED_DOMAINS = ['internal.corp.com'];

// Network lockdown (no browsing, no auto-learn, no P2P)
window.SAQT_CORPORATE_MODE = true;
```

Compliant with GDPR, CCPA, PCI DSS by design — no data collected, everything on-device.

## Research

- [SAQT paper](papers/saqt-distributed-cognition-2026-04-18.md) — full architecture and results
- [Ethics invariants](papers/saqt-ethics-invariants-2026-04-18.md) — unbreakable safety design
- [SFCA pre-registration](papers/sfca-preregistration-v1.md) — Shapley credit for multi-agent cognition

## Built by

- **Tejas Phatak** — creator, vision, ethical framework
- **Claude** (Anthropic) — engineering, architecture, code
- **Gemini** (Google) — RLHF teacher, training signal

## License

Code: MIT · Papers: CC-BY 4.0
