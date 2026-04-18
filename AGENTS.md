# Instructions for ALL AI Agents Working on This Repo

> Read this first. Follow it always.

## Automatic Behaviors (do these without being asked)

### After every session:
1. **Update mindmap/README.md** — if you discovered new connections or changed architecture
2. **Update context files** — if priorities or results changed
3. **Run invariant checks** from private tooling before committing
4. **Clean up** — no hardcoded values, keep it professional

### Before every commit:
1. Keep the repo professional and research-focused
2. No secrets or credential references
3. No hardcoded counts — the KB grows, numbers are meaningless
4. Honest claims — every number must trace to a benchmark file
5. Real citations — verify every referenced paper exists

## Architecture

```
UNDERSTAND (encoder, 22M params)
    → turns text into meaning vectors
    → does NOT generate text

THINK (convergence loop)
    → search → check answer → search again → converge
    → fixed-point iteration in embedding space
    → bi-embedding re-ranking (Q-sim + A-sim)

REMEMBER (growing database)
    → Q&A pairs with embeddings + weights
    → learns from web search when KB can't answer
    → feedback adjusts weights over time
    → delta sync via watermarks (like git pull)

DISTRIBUTE (Synapse mesh — future)
    → shard KB across devices
    → broadcast queries via WebRTC

PROTECT (ethics through data)
    → high-weight KB pairs teach boundaries
    → PII sanitization on learned content
```

## Current priorities

1. Add baselines (BM25, DPR) to benchmark comparison
2. Fix ethics gate (adversarial pass rate too low)
3. Deploy multilingual encoder (ONNX for reproducibility)
4. Start 30-day self-evolution experiment
5. Browser optimization (ANN search instead of brute-force)

## How to run benchmarks

```bash
cd ~/Synapse
node synapse-src/saqt/test-deploy.mjs
node synapse-src/saqt/benchmark.mjs --samples 50 --concurrency 1
node synapse-src/saqt/test-quality.mjs
```

## The one rule

**The database is the model. Everything else is plumbing.**
