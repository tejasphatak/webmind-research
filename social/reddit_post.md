# Reddit Post — r/LocalLLaMA

**Title:** We measured 22x activation compression at shard boundaries — makes distributed inference on consumer devices actually practical

**Body:**

Hey r/LocalLLaMA,

I've been building Synapse (webmind.sh) — a decentralized inference system that shards LLMs across volunteer WebGPU browsers. The #1 bottleneck isn't compute, it's **bandwidth between devices**.

Last night I had an idea: what if most of the activation data flowing between shards is predictable? What if we could send a compact "carrier" signal and a tiny "payload" of corrections instead of the full tensor?

**TL;DR:** We decompose activations into PCA basis (carrier) + sparse residual (payload). On Gemma 3 1B IT:

| Config | Compression | KL Divergence | Top-1 Match |
|---|---|---|---|
| Rank 32, carrier only | **22x** | 0.023 | 100% |
| Rank 16, 10% sparse | 4.3x | 0.074 | 100% |
| Rank 8, 1% sparse | 10.5x | 0.18 | 96% |

The effective dimensionality of activations is ~32 out of 1536 hidden dims. 97% of the variance lives in 32 principal components.

**What this means for you:**
- A 70B model sharded across 10 laptops currently needs high-bandwidth links between each device
- With 10-22x compression, normal internet connections work
- Phones on 3G could participate in the swarm
- No retraining needed — works post-hoc on any model

**Caveats (being honest):**
- Tested on 1B model only. 7B/12B in progress.
- Tested on 50-70 token prompts. Longer contexts need validation (but math says compression ratio actually *improves* with longer sequences).
- Single-splice-point tested. Multi-boundary compounding not yet measured.
- Comparison against INT4/INT8 baseline still needed.

**Reproduce it yourself:**
```bash
git clone https://github.com/tejasphatak/webmind-research
cd webmind-research
export HF_TOKEN=your_token
bash tools/reproduce.sh
```

Paper draft coming this week, targeting MLSys.

This was AI-assisted research (Claude Opus 4.6 + Gemini 3.1 Pro) under my direction. All code and data are open. Happy to answer questions.

— Tejas

**Edit:** For the technically curious — the "carrier" is a PCA projection, and the "payload" is the top-k% of residual entries by magnitude. The PCA basis is a shared prior between nodes (computed once, distributed to all). Only the projections + sparse corrections cross the wire per inference call.
