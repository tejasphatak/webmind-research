# Decompose → Distribute → Collaborate: Distributed-Native LLM Architecture

**Inventor:** Tejas Phatak
**Date:** 2026-04-17
**Status:** Architecture design + mathematical optimization analysis

## Core Idea

Decompose a large pretrained LLM (e.g., 70B) into N interdependent small expert models (~1B each, <500MB GPU RAM) that run on consumer devices. NO single expert produces correct output alone — they MUST collaborate, like neurons in a brain.

## Architecture

### Step 1: Decompose
- Structured model parallelism: each expert holds a SUBSET of attention heads + MLP slices
- Overlapping knowledge for resilience (erasure-coding inspired)
- Knowledge distillation from full model to train each expert's partial contribution

### Step 2: Distribute  
- Each device runs one complete expert (<500MB)
- P2P via WebRTC on LAN (2-10ms), DHT for discovery
- No centralized coordinator required

### Step 3: Collaborate
- Each expert produces partial hidden state
- Partial states compressed (carrier-payload, ~128 bytes) and sent to peers
- Peers aggregate → full hidden state → next token

## Mathematical Optimizations

### 1. Low-Rank Factorization of Expert Communication

The full model's hidden state h ∈ ℝ^d can be decomposed as:

```
h = Σᵢ Eᵢ(x)    where Eᵢ is expert i's contribution
```

If we factorize the communication as:

```
Eᵢ(x) = Uᵢ · zᵢ(x)    where Uᵢ ∈ ℝ^(d×k), zᵢ ∈ ℝ^k
```

Then each expert sends only zᵢ (k-dimensional, e.g., k=16 → 32 bytes), and the aggregator reconstructs:

```
h = Σᵢ Uᵢ · zᵢ = [U₁ | U₂ | ... | Uₙ] · [z₁; z₂; ...; zₙ]
```

The basis matrices Uᵢ are preloaded on all devices (constant, not per-token).
**This is carrier-payload compression applied at the expert level.**

### 2. Coding-Theoretic Resilience (Erasure Coding for Knowledge)

Model the N experts as an (N, K) erasure code over the knowledge space:
- Any K of N experts are sufficient to reconstruct the full output
- Each expert encodes redundant information across peers
- Loss of up to (N-K) experts → graceful degradation, not failure

Concretely: during distillation, each expert learns a linear combination of the full model's representations:

```
Expert_i = Σⱼ Gᵢⱼ · Component_j    where G is a generator matrix (N×K)
```

At inference, the aggregator solves a linear system with whatever K experts respond:

```
h = G⁻¹_available · [E₁(x); E₃(x); ...; Eₖ(x)]
```

This is Reed-Solomon coding applied to neural network activations.

### 3. Speculative Parallel Decoding

In standard autoregressive generation, tokens are produced one at a time. With N experts available:

```
Standard:  t₁ → t₂ → t₃ → t₄  (sequential)
Speculative: Expert₁ predicts t₂, Expert₂ predicts t₃, Expert₃ predicts t₄
             Then verify in one pass → accept or reject
```

Each expert can speculatively generate the NEXT token while waiting for peer contributions. If the speculation is correct (high probability for simple tokens), multiple tokens are produced per round-trip.

**Expected speedup: 2-4× for typical English text** (where next-token entropy is low).

### 4. Adaptive Expert Quantization (Entropy-Aware)

Not all experts need the same precision. For a given token:
- The "confident" expert (highest routing weight) runs at full precision
- Other experts can run at lower precision (int4/int2) since their contribution is smaller
- Saves compute on devices with less GPU capacity

```
Effective compute per token = full_precision(top_expert) + Σᵢ reduced_precision(other_experts)
```

This naturally assigns more work to more capable devices.

### 5. Information-Theoretic Minimum Communication

For N experts each holding d/N dimensions of the hidden state:
- Minimum bits to communicate = H(h | local_context) — the conditional entropy
- For modern transformers at middle layers, consecutive-token cosine similarity > 0.97
- This means H(h_t | h_{t-1}) is VERY low — only ~3% of the information changes per token
- **Delta encoding between tokens**: send only what changed, not the full state
- Expected compression: 10-30× on top of carrier-payload

### 6. Gossip Aggregation (Byzantine Fault Tolerance)

Instead of a single aggregation point:
- Experts gossip partial results to neighbors
- After O(log N) rounds, all experts converge on the same aggregated state
- If some experts are malicious or faulty (Byzantine), use majority voting on partial results
- Works with gossip protocols already designed for Synapse (DHT + P2P-local)

### 7. Topology-Aware Expert Placement

On a LAN with 30 tablets:
- Some tablets are physically closer (same table, same WiFi AP)
- Place interdependent experts on nearby devices
- Use network latency measurements to optimize the communication graph
- This is a graph partitioning problem: minimize cross-partition communication subject to memory constraints

Solvable via spectral clustering on the device latency matrix.

### 8. Progressive Expert Loading

For cold-start (first inference request):
- Don't load all experts at once
- Load the "core" experts first (highest routing weight across typical prompts)
- Other experts load lazily as needed
- First tokens use fewer experts (lower quality but fast) → quality improves as more experts come online

This gives near-instant first-token response with progressive quality improvement.

## Communication Budget Analysis

For a 70B model decomposed into 20 × 1B experts, K=16 summary dim:

| What | Size per token | With delta encoding |
|------|---------------|-------------------|
| Expert partial hidden state (raw) | 7168 bytes (d=3584, fp16) | — |
| Carrier-payload PCA-k=16 | 32 bytes | ~3 bytes |
| Gzip'd token probabilities | ~200 bytes | ~20 bytes |
| Routing decision | 8 bytes | 1 byte |
| Total per expert per token | ~240 bytes | ~24 bytes |
| Total for 3 active experts | ~720 bytes | ~72 bytes |
| Bandwidth at 50 tok/s | 36 KB/s | 3.6 KB/s |

**3.6 KB/s per device is achievable on ANY network, including 2G cellular.**

## Prior Art Integration

| Technique | Source | Role |
|-----------|--------|------|
| Sparse Upcycling | Google 2022 | Decomposition starting point |
| DeRS (base + delta) | 2025 | Efficient expert representation |
| Distributed MoA | Mitra 2024 | Gossip-based aggregation |
| Carrier-Payload PCA | Webmind 2026 | Inter-expert compression |
| Async Stale-Summary | Webmind 2026 | Zero-wait collaboration |
| Shadow Replicas | Webmind 2026 | Resilience via hot standby |

## What's Novel (nobody has combined these)

1. Decomposition of dense model into N COMPLETE interdependent experts
2. Erasure-coding-inspired redundancy for expert dropout resilience
3. Carrier-payload compression at expert communication level
4. Async gossip aggregation with delta encoding (3.6 KB/s per device)
5. Progressive expert loading for instant cold-start
6. Topology-aware placement on consumer device mesh
7. All on WebGPU, offline-first, no datacenter

## Minimum Viable Experiment

1. Take GPT-2 124M (12 layers, 768 hidden, 12 heads)
2. Decompose into 4 experts (3 heads each, MLP split 4 ways)
3. Distill from full GPT-2 teacher
4. Run on 4 processes on one machine, communicating via TCP (simulates LAN)
5. Test: quality vs monolithic, resilience (kill 1 expert), latency
6. Measure: bytes transferred per token, end-to-end tok/s
