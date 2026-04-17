# Asynchronous Learned-Summary Distributed Inference

**Inventor:** Tejas Phatak
**Date:** 2026-04-17
**Status:** Empirical validation in progress — cross-model staleness measured

## Core Idea

Design a transformer architecture where inter-device communication is **asynchronous and one-token-stale**. Each device runs a full local computation using its partition of the model, and incorporates **compressed summaries from neighboring devices that arrived during the previous token's computation**. No device ever waits for remote data.

This eliminates network latency from the critical path entirely. The model is trained with this constraint so it learns to compensate for stale remote context.

## Motivation (backward from target)

| Metric | Local A100 (70B int4) | Naive 100-phone pipeline | This architecture |
|--------|----------------------|--------------------------|-------------------|
| tok/s | ~50 | ~10 | target: ~40-50 |
| Latency per token | 20ms | 100ms (network-bound) | 20ms (compute-bound) |
| Network wait per token | 0 | 27ms × stages | **0** (async) |
| Min bandwidth per device | N/A | 600 KB/s | **11 KB/s** |

The 5× gap between local and distributed inference is almost entirely network latency between pipeline stages. Eliminating synchronous communication closes it.

## Architecture

### Distributed-native neuron

```
# Standard (synchronous, must wait for full x):
y = activation(W · x_full)

# Distributed-native (async, zero-wait):
y = activation(W_local · x_local + α · W_cross · summary_remote(t-1))
                 ↑ current token        ↑ previous token, arrived async
```

### Per-token computation flow

```
Token t on Device A:

1. COMPUTE (local, no wait):
   y_A(t) = LocalTransformer(x_A(t)) + W_cross · summary_B(t-1)
                                                    ↑ already in buffer

2. COMPRESS (local, trivial):
   summary_A(t) = W_compress @ y_A(t)     # [hidden/D] → [K]
                                            # K=16, learned projection

3. SEND (async, background):
   send summary_A(t) to neighbors         # arrives before token t+1
                                            # 16 × 2 bytes = 32 bytes

4. RECEIVE (async, background):
   buffer summary_B(t) from neighbors     # used at token t+1
```

Communication is fully overlapped with compute. The device never blocks.

### Why one-token-stale works

In autoregressive generation, the hidden state at token t and token t-1 differ by one position's contribution. Empirically, consecutive-token activations are ~95% correlated (measured on Carrier-Payload Qwen 32B data). The model, trained with this constraint, learns to:

1. Put primary reasoning capacity in **local computation** (95% of quality)
2. Use remote summaries for **context correction** (5% of quality)
3. Learn **what to compress** into K=16 summary dimensions (not post-hoc PCA — learned end-to-end)

### Topology

Expander graph: each device connects to log₂(D) neighbors.

- D=100 devices → 7 neighbors each
- Bandwidth: 16 fp16 × 7 neighbors × 50 tok/s = **11.2 KB/s** per device
- A phone on 3G (100 Kbps up) can sustain this
- Any device dropping doesn't partition the graph (expander property)

### Fault tolerance (built into architecture)

- Each device holds its shard + stale copy of one neighbor's shard
- If neighbor drops: use stale shard (degraded quality, not failure)
- Model trained with random-dropout of remote summaries (like dropout but at the device level) — learns to function with missing neighbors
- Background: remaining devices re-partition and re-balance

## Comparison to prior art

| Approach | Communication pattern | Latency cost | Bandwidth | Trained for distribution? |
|----------|----------------------|-------------|-----------|--------------------------|
| Pipeline parallelism | Sync, sequential | O(stages × RTT) | High (full activation) | No |
| Tensor parallelism | Sync, all-reduce | O(RTT) per layer | High (partial sums) | No |
| MoE (Mixtral, etc.) | Sync, routed | O(RTT) per layer | Medium (selected experts) | Partially (routing) |
| Petals (collaborative) | Sync, pipeline | O(stages × RTT) | High | No |
| **This work** | **Async, stale summary** | **0 (overlapped)** | **11 KB/s** | **Yes (end-to-end)** |

Key differentiator: every prior approach has synchronous inter-device communication on the critical path. This is the first (to our knowledge) that eliminates it entirely by making the model architecture aware of the communication delay.

## Training design

### Modified forward pass

```python
class DistributedNativeLayer(nn.Module):
    def __init__(self, hidden_dim, summary_dim=16, n_neighbors=7):
        self.local_attn = MultiHeadAttention(hidden_dim)
        self.local_ffn = FeedForward(hidden_dim)
        self.compressor = nn.Linear(hidden_dim, summary_dim)      # learned compression
        self.cross_proj = nn.Linear(summary_dim * n_neighbors, hidden_dim)
        self.alpha = nn.Parameter(torch.tensor(0.1))               # cross-contribution weight

    def forward(self, x_local, stale_summaries):
        # Local computation (standard transformer)
        local_out = self.local_ffn(self.local_attn(x_local))

        # Cross-device contribution (from previous token)
        if stale_summaries is not None:
            cross = self.cross_proj(torch.cat(stale_summaries, dim=-1))
            local_out = local_out + self.alpha * cross

        # Generate summary for neighbors (used at t+1)
        summary = self.compressor(local_out.detach())  # stop-gradient: no backprop across devices

        return local_out, summary
```

### Training strategy

1. **Phase 1 — Standard pretraining** on full model (single device). Establish baseline quality.
2. **Phase 2 — Distributed fine-tuning** with simulated async communication:
   - Partition model across simulated devices
   - Feed summaries from t-1 (shift by one position)
   - Random-drop summaries with p=0.1 (simulate device failure)
   - Train compressor + cross_proj + alpha end-to-end
3. **Phase 3 — Deploy** to real distributed devices. No further training needed.

Phase 2 can run on a single GPU by simulating the partitioning and delay — no actual distributed setup needed for training.

### Key hyperparameters to sweep

- Summary dimension K: [4, 8, 16, 32, 64]
- Cross-contribution weight α: learned vs fixed
- Number of neighbors: [3, 5, 7, log₂(D)]
- Staleness: 1-token vs 2-token vs variable

## Minimum viable experiment

**Model:** Gemma 3 1B (small enough to train phase 2 on one GPU)
**Partition:** Simulate 4 devices, each holding 7 layers
**Metrics:**
- Perplexity with stale summaries vs full synchronous vs no cross-device
- Quality vs K (summary dimension) curve
- Robustness: quality when 1 of 4 simulated devices "drops" (summaries zeroed)

**Compute:** ~1 day on a single A100, ~$50 on RunPod spot
**Success criterion:** perplexity within 5% of full synchronous model at K=16

## What this enables for Synapse

If validated, a 70B model runs across 100 volunteer phones at near-local inference speed:
- 40-50 tok/s sustained (vs 50 on a single A100)
- 270ms first-token latency (pipeline fill)
- 11 KB/s bandwidth per device (works on 3G)
- Any 5 devices can drop without interruption
- No datacenter, no cloud, no cost

This is the endgame architecture for Synapse.

## Empirical Validation: Staleness Experiment (2026-04-17)

### Experiment 1: Consecutive-token cosine similarity (Gemma 3 1B)

Measured cosine similarity between hidden states at position t and t-1, averaged across 4 prompts. Raw data: `findings/2026-04-17-staleness-experiment.json`.

| Depth range | Mean cosine sim | Verdict |
|---|---|---|
| 0-20% (early) | 0.659 | Dangerous — tokens not yet mixed |
| 20-40% | **0.979** | Excellent — safe for stale exchange |
| 40-60% (middle) | **0.966** | Excellent |
| 60-80% | **0.957** | Good |
| 80-100% (deep) | 0.868 | Degraded — tokens diverging for prediction |

Simulated async at device boundaries (K=16 compressed stale):
- Layer 6 boundary: 0.967 cosine sim
- Layer 12 boundary: **0.992** cosine sim
- Layer 18 boundary: 0.966 cosine sim

**Overall verdict: "Moderate support — stale activations need compression help"** (mean 0.877 across all layers; 0.97+ in the sweet spot).

### Experiment 2: Cross-model transfer (Gemma 3 1B vs GPT-2 Medium)

Raw data: `findings/2026-04-17-staleness-cross-model.json`.

| Depth bucket | Gemma 3 1B (26 layers) | GPT-2 Medium (24 layers) |
|---|---|---|
| 0-20% (early) | 0.659 | 0.765 |
| 20-40% | **0.979** | 0.848 |
| 40-60% (middle) | **0.966** | 0.830 |
| 60-80% | **0.957** | 0.802 |
| 80-100% (deep) | 0.868 | 0.835 |

**Critical finding: the staleness pattern does NOT fully transfer across architectures.**

- Gemma has a pronounced sweet spot (0.97-0.99) in the 20-60% depth range. GPT-2 has NO sweet spot — flat ~0.83 across all non-early layers.
- GPT-2's deep layers slightly improve (0.835 > 0.830), opposite of Gemma.
- The architectural principle (async is viable) transfers. The easy-mode assumption (just use stale activations without training) only works cleanly on modern architectures.

### Hypothesis: why modern architectures have the sweet spot

Modern architectures (Gemma, Llama) differ from GPT-2 in ways that promote smoother activation trajectories:
- **RoPE** (rotary position embedding): position info is multiplicative, not additive. Adjacent positions produce geometrically similar rotations → higher cosine similarity between consecutive tokens' activations.
- **SwiGLU** (gated FFN): the gating mechanism creates smoother activation surfaces than ReLU/GELU.
- **RMSNorm** (vs LayerNorm): normalizes magnitude without centering, preserving more directional consistency.
- **GQA** (grouped-query attention): shared KV heads across query groups create more uniform attention patterns.

**Testable prediction:** If this hypothesis is correct, replacing GPT-2's components with modern equivalents (one at a time) should monotonically increase the middle-layer cosine similarity toward Gemma's values.

### Implications for the async architecture

1. **Compressor must be per-model** — not just weights, but boundary placement strategy differs.
2. **Phase 2 training is essential for older architectures** — can't rely on pretrained activation structure alone.
3. **For Synapse (modern models only): async may work out-of-the-box** with just a learned compressor, no full retraining. This is the optimistic path.
4. **For universal async inference: need phase 2 fine-tuning** per model family. Cheap (~minutes, linear projection only) but not free.

## MVE Results (2026-04-17)

### v1: Frozen base model (Qwen 2.5 1.5B, 4 devices)
- Cross-device layers (9 params per config) train successfully — loss converges below baseline
- Alpha gradient: early boundaries ~0.5, deep ~0.2 (matches staleness data)
- **Eval PPL identical across all configs (16.50)** — cross-device contribution too small to measure with frozen base
- Detach vs no-detach: no difference (base model frozen, so gradient path distinction is moot)
- K=[4,8,16,32,64]: all equivalent
- Raw data: `findings/2026-04-17-async-stale-mve-results.json`

### v2: Unfrozen last 4 layers (420M trainable params)
- PPL moves: 16.52 → 13.41 (-18.8% improvement)
- **BUT: no_cross baseline (zero communication) also gets 13.41**
- Cross-device summaries add zero measurable quality over no-communication at this scale
- HellaSwag: 52.2% → 48.4-48.8% (fine-tuning domain shift, not async damage)
- Throughput simulation: sleep()-based, doesn't capture true async overlap benefit
- K=[4,16,64]: all equivalent (~13.40 ppl)
- Raw data: `findings/2026-04-17-async-stale-mve-v2-results.json`

**Sync throughput baseline (simulated latency per boundary × 3 boundaries):**

| Latency | tok/s | Slowdown |
|---------|-------|----------|
| 0ms | 40.7 | 1x |
| 10ms | 17.3 | 2.4x |
| 50ms | 5.3 | 7.7x |
| 100ms | 3.0 | 13.6x |
| 200ms | 1.6 | 25.4x |

### Interpretation
1.5B model with 4 partitions (7 layers each) has enough local capacity per device to function independently. Cross-device info is redundant at this scale. Need 7B+ model with 8-16 partitions to create genuine information gaps between devices.

### Gemini cross-validation (2× rounds)
- Round 1: correctly flagged identical PPL as eval bug → led to v2
- Round 2: correctly identified no_cross = async parity as scale issue, not architecture failure
- Suggested next: 7B+ model, more partitions, proper concurrent async simulation

## Next experiments (prioritized)

1. **7B model, 8+ partitions** — each device holds ~4 layers, genuinely needs cross-device context. Est. $10-20 on A100.
2. **Proper async throughput simulation** — concurrent threads with non-blocking send/recv, not sleep(). Shows the real speedup.
3. **Optimal boundary placement** — use staleness data to place boundaries at the "sweet spot" (20-60% depth) rather than uniform spacing.
4. **Performance optimization research** — identify additional techniques to maximize tok/s in distributed inference: speculative decoding at device level, adaptive compression based on token importance, pipeline bubble minimization, dynamic device recruitment/retirement.

## Open questions

1. Does one-token-stale degrade quality on long-range reasoning tasks? (Stale context may miss recent information that changes the generation direction)
2. Can the compressor be shared across layers or must each layer learn its own?
3. How does this interact with KV-cache? Each device has a local KV cache; the stale summaries don't contribute to cached keys/values.
4. Training phase 2 with stop-gradient on summaries — does this create dead gradients in the compressor? May need straight-through estimator.
5. Is there a theoretical lower bound on K for a given quality target? Information bottleneck theory may give this.
6. **NEW:** Is the Gemma sweet spot caused by RoPE, SwiGLU, RMSNorm, GQA, or their combination? Ablation needed.
7. **NEW:** At what model size / partition count does cross-device information become non-redundant?
8. **NEW:** Can we predict optimal boundary placement from the staleness profile without training?
9. **NEW:** What's the interaction between speculative decoding and async summaries? Could devices speculatively decode using stale summaries and verify later?
7. **NEW:** Does Llama 3 show the same sweet spot as Gemma? (Same architectural family — should confirm.)
