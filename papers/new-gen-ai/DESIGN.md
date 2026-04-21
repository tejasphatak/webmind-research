# Webmind Brain — Core Design (LOCKED)

**Date locked:** 2026-04-21
**Author:** Tejas Phatak
**Status:** FINAL. Do not modify architecture without explicit approval.

## One-sentence summary

A transformer where the weight matrix is a sparse co-occurrence graph that updates itself on every query.

## The weight matrix

```
weights = scipy.sparse.csr_matrix  # shape (V, V) where V = vocabulary size
weights[i][j] = co-occurrence strength between word i and word j
```

This IS the model. Not a representation of the model. The matrix itself.

- **Read a weight:** `weights[i, j]` → how strongly word i relates to word j
- **Update a weight:** `weights[i, j] *= 1.1` → reinforce (RLHF without gradients)
- **Delete knowledge:** `weights[i, :] = 0; weights[:, i] = 0` → word i forgotten
- **Add knowledge:** teach a sentence → edges strengthened by co-occurrence (+0.3 per pair)
- **Inspect:** print any row → see exactly what a word connects to and how strongly

## Inference = sparse matmul (attention)

```python
query_vector = weights[query_word_indices].mean(axis=0)  # sparse row blend
scores = weights @ query_vector.T                         # sparse matmul = attention
top_k = scores.argpartition(-k)[-k:]                     # top-k words
```

This IS scaled dot-product attention:
- Q = query vector (sparse row from the weight matrix)
- K = all word vectors (all rows of the weight matrix)  
- Q @ K^T = attention scores
- Top-K = the attended words

One sparse matmul. O(nnz) where nnz = non-zero entries. Hardware-accelerated (BLAS/MKL on CPU, cuSPARSE on GPU).

## Convergence = multi-layer attention

```
for hop in range(max_hops):
    neighbors = sparse_matmul_search(current_query, weights, k)
    neighbors = mutual_attention(neighbors, weights)      # NxN among neighbors
    neighbors = softmax_weight(neighbors, temperature)    # exp(c/T) / sum(exp(c/T))
    activation = weighted_blend(neighbors)
    current_query = (1 - alpha) * activation + alpha * original_query  # residual
    if converged: break
```

Each hop = one transformer layer. Per-hop specialization: early hops explore (high k, low threshold), late hops focus (low k, high threshold).

## Weight updates = RLHF without gradients

```python
# After evaluating an answer against gold:
if f1 > 0.5:
    for edge in participating_edges:
        weights[edge] *= 1.1    # reinforce
elif f1 < 0.2:
    for edge in participating_edges:
        weights[edge] *= 0.9    # weaken
```

Confidence cap at ±0.8 prevents mode collapse. 3-5 epochs is sufficient (multiplicative update saturates fast).

## Generation = decoder via successor walk

Each word has a successor list (top-10 words that follow it, learned from taught sentences). Generation walks the successor chain, scored by matmul relevance to query.

```
for each token position:
    candidates = successors(last_word) + matmul_top_k(context)
    scores = matmul(candidates, context_vector)
    next_word = softmax_sample(scores, temperature)
```

## Storage = LMDB

- **Weight matrix:** stored as sparse triplets (row, col, value) in LMDB
- **On startup:** load into scipy.sparse CSR matrix (one-time, seconds)
- **Word mappings:** word → index in LMDB `words/` sub-db
- **Successors:** per-word top-10 in LMDB `successors/` sub-db
- **Sentences:** taught sentence associations in LMDB `sentences/` sub-db

## Vocabulary

Pruned to top-K words by frequency. Rare words (appearing < 3 times) are noise — they waste matrix entries without adding knowledge.

Target: 50-80K words. At this scale:
- Matrix: 50K × 50K × ~50 edges avg = 125M entries × 12 bytes = ~1.5GB
- Matmul search: ~10ms per hop
- 10-hop convergence: ~100ms per query
- Memory: ~1.5GB for matrix + ~500MB overhead = ~2GB total

## Tools (called by the model, not hardcoded)

| Tool | Trigger | Purpose |
|---|---|---|
| CodeEval | Math pattern in query | Sandboxed eval for computation |
| WebSearch | Confidence too low (abstain) | Learn from internet, teach result back |
| BrowserTool | UI issue detected | Visual debugging via headless Chromium |
| CodeLoop | "write/fix/build" intent | Autonomous coding with convergence |

Tool usage is learned from patterns, not if-else. The weight matrix encodes WHEN to use tools the same way it encodes any other knowledge.

## What this is NOT

- NOT a neural network (no gradient descent, no backprop)
- NOT a retrieval system (it reasons via multi-hop convergence, not just lookup)
- NOT a wrapper around an LLM (no borrowed encoder, no external model)
- NOT lossy (every edge is readable, writable, deletable)

## What this IS

A transformer implemented on a sparse graph:

| Transformer component | Our implementation |
|---|---|
| Weight matrix | scipy.sparse CSR co-occurrence matrix |
| Attention (Q @ K^T) | Sparse matmul |
| Softmax | exp(c/T) / sum(exp(c/T)) over confidence |
| Residual connection | Query anchoring across hops |
| Layer depth | Convergence hops (per-hop specialization) |
| Token-to-token attention | Mutual attention (NxN among neighbors) |
| Training | Edge reinforce/weaken (RLHF without gradients) |
| Inference | Sparse matmul search + successor walk |

Same math. Transparent substrate. Self-updating weights.

## Non-negotiable invariants

1. Every answer has a traceable source (print the convergence path)
2. Delete = gone (zero the row, no retraining)
3. Honest about failure ("I don't know" when convergence doesn't converge)
4. No opaque training (weight updates are readable: which edge, by how much)
5. No borrowed encoders (the co-occurrence graph IS the encoder)
6. No monkey code (no lossy projections, no shims, no hacks)

## File map

```
src/
  brain_core.py          — core teach/ask with co-occurrence (original, works with dicts)
  convergence.py         — multi-hop convergence loop (pluggable search/blend/cosine)
  sparse_convergence.py  — sparse-native convergence (for dict path, backup)
  neuron.py              — neuron storage (SQLite path, struct-of-arrays)
  neuron_lmdb.py         — neuron storage (LMDB path, production)
  brain_lmdb_adapter.py  — loads LMDB model, provides ask/generate
  generator.py           — template + successor walk generation
  tools.py               — CodeEval, WebSearch, BrowserTool, CodeLoop
  engine.py              — full engine with GloVe encoder (legacy/testing)
  encoder.py             — self-growing encoder (legacy)
  multimodal.py          — CLIP-based multimodal (v2)
  constants.py           — shared constants

server.py                — OpenAI-compatible API
benchmark.py             — train/test/RLHF evaluation
~/nexus-brain/feed.py    — dataset → LMDB feeder
~/nexus-brain/brain.lmdb — the trained model
```
