# Webmind Research — Context for AI Assistants

## What this repo is

A research project building a **self-evolving retrieval engine** — a new AI architecture that is neither an LLM nor a search engine.

**Core idea:** All you need is:
1. A model that understands language (22M params, sentence transformer)
2. A thinking module (convergence loop in embedding space)
3. A growing database (verified Q&A pairs, learns from every query)

Everything else in an LLM is expensive memorization that belongs in a database row.

## Key results

- **Self-evolution proven:** 0.7% → 25.3% exact match on held-out QA benchmarks (HotPotQA: 0→72%), zero human intervention
- **Browser-native:** 214MB, works offline on phones
- **50+ languages:** Hindi/Marathi ↔ English at 92-97% embedding similarity
- **Live demo:** https://webmind.sh

## Repo structure

```
papers/                     ← Research papers (main: self-evolving-retrieval-2026-04-18.md)
benchmarks/                 ← All benchmark results (NQ, TriviaQA, HotPotQA)
mindmap/                    ← How all papers and ideas connect
inventions/                 ← Timestamped invention disclosures
findings/                   ← Experiment results and negative results
tools/                      ← Python/JS research tools and prototypes
trained_model/              ← KB data, embeddings, FAISS index
agp/                        ← AGP (Attention-Guided Pruning) research
sfca/                       ← SFCA (Shapley-Fair Credit Assignment) library
data/                       ← Training datasets (Wikipedia, NQ, TriviaQA, etc.)
```

## Architecture (code lives in github.com/tejasphatak/Synapse)

```
User Query → MiniLM encoder (22M params) → embedding
          → FAISS search (300K+ Q&A pairs) → top-K candidates
          → Bi-embedding re-rank (answer alignment) → best answer
          → If weak → parallel web search → learn answer → KB grows
          → Convergence loop (iterate until embedding stabilizes)
```

Key files in Synapse repo:
- `docs/saqt-shim.js` — browser SAQT engine
- `docs/sw-backend.js` — Service Worker API backend
- `synapse-src/saqt/serve.py` — server (learn/delta/sync/ethics/web search)
- `synapse-src/saqt/test-deploy.mjs` — 53 structural tests
- `synapse-src/saqt/test-quality.mjs` — 22 LLM-judged tests
- `synapse-src/saqt/benchmark.mjs` — NQ/TriviaQA/HotPotQA benchmark with RLHF

## Principles (non-negotiable)

1. **No hardcoding.** Intelligence comes from data, not code. Thresholds, behaviors, ethics — all taught through KB pairs.
2. **Honest claims.** The 71.3% result is in-distribution (same questions). The real generalization number is 25.3% on held-out data. Always report both.
3. **Self-evolution.** The system learns from every query. Web search results → KB. Benchmark misses → gold answers learned back. The engine gets smarter without human intervention.
4. **The database IS the model.** The encoder is just the interface. Knowledge, behavior, ethics — all in the database.
5. **No secrets in public repo.** No API keys, no personal emails, no employer names, no infrastructure details.

## What to work on

Priority order:
1. Improve benchmark scores (add baselines: BM25, DPR comparison)
2. Fix ethics gate (48% adversarial pass rate — too low)
3. Deploy multilingual encoder (ONNX for cross-hardware reproducibility)
4. 30-day self-evolution experiment (daily benchmarks, track the learning curve)
5. Browser optimization (Voy/USearch ANN instead of brute-force)

## Related repos

- **github.com/tejasphatak/Synapse** — the runtime code (browser engine, server, tests)
- **webmind.sh** — live demo (GitHub Pages from Synapse/docs/)
