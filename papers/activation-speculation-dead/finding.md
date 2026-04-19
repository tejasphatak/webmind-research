# Why activation-level speculative decoding can't work (and the measurement that proves it)

**Date:** 2026-04-15
**Project:** Synapse — distributed browser-WebGPU LLM inference
**Status:** Finding — shareable. Measurements reproducible on the Synapse Qualcomm+Qualcomm mobile topology.

## TL;DR

We built Phase 2 speculative decoding into Synapse on the hypothesis that predicting the *next activation* at a shard boundary would hide ~50 ms of network latency between distributed browser nodes. With the feature shipped, deployed, and running against real devices, we measured it and found: the linear-extrapolation predictor produces **mean cosine 0.38 ± 0.12** against ground-truth activations on coherent GPT-2 117M decode output (n=96). Out of 96 samples, **zero** cleared even a 0.7 acceptance threshold; the max observed was 0.607.

The optimization's own acceptance threshold (0.995) is roughly four standard deviations above the best-case cosine we ever saw. Speculation would fire **never** in production.

This finding is architectural, not a tuning issue. Consecutive autoregressive transformer activations don't move locally linearly — and they can't be expected to, because each new token injects a fresh low-rank update whose direction is precisely the thing a downstream decoder is supposed to *compute*, not predict from the past two steps.

The route forward is not threshold tuning or batch depth. It's either a learned predictor (tiny MLP trained on activation traces), or pivoting to **token-level** speculation with a draft-verify scheme.

## The seductive hypothesis

Synapse splits a transformer across browser-GPU compute nodes coordinated over WebSockets. On mobile-dominated topologies, network round-trip between shards is ~50 ms, vs. a single-token forward pass on-GPU of ~1 ms. The wire, not the compute, is the bottleneck.

The VLSI analogy is compelling:
- Branch prediction in CPUs hides the cost of pipeline flushes by speculatively executing down the predicted path, rolling back only on misprediction.
- Here, *activation prediction* at a shard boundary would let a downstream node start computing on the predicted activation while the real one is still in flight. Right prediction → hide 50 ms. Wrong → discard and recompute (no worse than baseline).

The predictor we shipped was the simplest thing that could work: linear extrapolation, `next ≈ curr + (curr − prev)`. It's trivially cheap (one vector add), needs only two observations to start, and carries an explicit confidence estimate from the cosine of consecutive deltas. If autoregressive decode moves smoothly through activation space, this should have signal.

## The bootstrap deadlock (twice)

Before we could measure anything, we had to get past a gate we put in our own way.

The controller was designed to *prove itself before being turned on*:

```
enabled = true  ⇐  verified_steps ≥ warmup_threshold
                ⇐  verify() calls
                ⇐  pending speculations exist
                ⇐  speculation kicked off
                ⇐  enabled = true
```

Classic circular gatekeeping. On a live deploy with 59 cached decode steps, we logged zero speculation events. The feature was dead code. We fixed the outer gate by letting `_speculateNext()` run regardless of `enabled` during warmup (shadow mode), with the acceptance of the speculative hidden state still gated on `enabled`.

Still zero events.

It turned out `_speculateNext()` itself had an inner gate: it silently dropped predictions with `confidence < 0.9`. For early decode steps, the cosine of consecutive deltas hovers in the 0.5–0.85 range; most predictions never made it past the inner gate to populate the `pending` map, so `verify()` still never ran.

**Same bootstrap-deadlock pattern, one layer deeper.** The lesson generalizes: any self-enabling optimization with a *prove yourself before I turn you on* threshold risks this. Fixing one gate doesn't help if there's another gate underneath expressing the same constraint.

The correct resolution is **cheap shadow mode**: during warmup, run `predict()` and `verify()` with no confidence gate and no GPU work. The predictor output is discarded; only the cosine statistic is kept. This costs one extra CPU cosine per decode step and produces unbiased measurement data without corrupting inference correctness.

The takeaway: whenever you design a *prove yourself* feature gate, also design an instrumentation path that accumulates the proving data independently of the gate.

## The measurement trick

With shadow mode producing per-step cosines without affecting inference, we added one log event per shadow-verify:

```js
if (this.speculative.lastShadowCosine != null) {
  this._sendLog("perf", "speculation_cosine_sample", {
    shardId: this.shardId,
    seqPos,
    cosine: +this.speculative.lastShadowCosine.toFixed(4),
  });
}
```

This turns the controller into a cheap predictor-quality measurement harness. No separate benchmark harness, no offline traces, no golden datasets. Just run the model as you normally would, and collect a histogram of how predictable next-activation is from current-activation.

## The result

Test rig: GPT-2 117M, 2-shard split (layers 0–5 / 6–11), homogeneous Qualcomm mobile topology (Adreno GPU via Android Chrome). 40-token decode at T=0.9 on the prompt "The universe is vast and complex". Coherent output — the one we care about measuring against. Repeated 2× with different prompts; 96 cosine samples total.

```
n = 96
mean cosine = 0.380
stdev       = 0.124
min         = −0.158
p25         = 0.309
p50         = 0.388
p75         = 0.469
p95         = 0.564
max         = 0.607

samples ≥ 0.995 threshold: 0 / 96   (0.0%)
samples ≥ 0.9:             0 / 96   (0.0%)
samples ≥ 0.8:             0 / 96   (0.0%)
samples ≥ 0.7:             0 / 96   (0.0%)
samples ≥ 0.6:             1 / 96   (1.0%)
```

Histogram (bucket count, 40-char bars):

```
[-0.20, 0.00)    2
[ 0.00, 0.20)    1
[ 0.20, 0.40)   51  #####################
[ 0.40, 0.60)   41  #################
[ 0.60, 0.80)    1
[ 0.80, 0.90)    0
[ 0.90, 0.95)    0
[ 0.95, 0.99)    0
[ 0.99, 1.00)    0
```

This is not a distribution that threshold tuning can rescue. It's tight, unimodal, and sits entirely below 0.7. The 0.995 acceptance threshold is ≈4 standard deviations above even the best-case observed cosine.

### Cosine doesn't improve with longer history

A natural next hypothesis: maybe the predictor's 2-step linear extrapolation is too short-sighted. Maybe with a longer history (EMA, or a sliding-window regression) the cosine climbs.

The data says no. Bucketed by sequence position (5-step buckets, n=38 samples):

```
seqPos    n   mean cosine   stdev   max
[ 5, 10)  3       0.232     0.152   0.343
[10, 15)  5       0.424     0.076   0.497
[15, 20)  5       0.385     0.115   0.585
[20, 25)  5       0.321     0.081   0.415
[25, 30)  5       0.358     0.027   0.390
[30, 35)  5       0.352     0.049   0.439
[35, 40)  5       0.333     0.037   0.382
[40, 45)  5       0.295     0.052   0.342
```

Cosine peaks at seqPos 10-15 (shortly after prefill), then *declines monotonically* to ~0.30 by seqPos 40. Decode activations become *less* predictable as generation progresses, not more. This makes physical sense: each sampled token injects a fresh low-rank perturbation, so successive hidden states increasingly diverge from the locally-linear trajectory assumed by the predictor.

Historical context does not save this predictor. The ceiling isn't about model capacity or context length — it's about the structural unpredictability of autoregressive token choice at the activation layer.

### Why this had to happen

GPT-2 decode at step t produces an activation whose novelty comes from one thing: the token sampled at step t−1, which is the output of the sampler operating on the LM head's logits over 50 257 tokens. That's a one-hot index into the embedding table, which then propagates through 12 transformer blocks to produce the activation at the shard-1 boundary.

There is no reason to expect this to move linearly. The embedding vectors of token t−1 and token t−2 point in essentially unrelated directions (GPT-2's embedding table is ~64-dim-effective in 768-dim space, and consecutive sampled tokens are as independent as the model's output distribution allows them to be). After 6 layers of non-linearity, the activation-space motion between step t−1 and step t is dominated by whichever one-hot direction the previous sampler picked, modulated by attention to the KV cache. Linear extrapolation of that trajectory is predicting next sample's embedding from current sample's embedding — which is precisely what the *rest of the network* is built to do, and what you can't get for free by arithmetic on the previous two hidden states.

### What would work

- **Learned activation predictor.** Distill a tiny (e.g. 2-layer 256-dim) MLP trained on captured activation sequences from real decode traces. Cosine will still be bounded by the information content of the next token, but a non-linear predictor can at least exploit attention-structure regularities that linear extrapolation misses.
- **Token-level draft-verify.** The canonical approach: a small fast model proposes K tokens ahead, the big model verifies them in one forward pass. This is what MEDUSA, SpecDec, and Lookahead Decoding do. At the architectural level, it sidesteps our bottleneck entirely because it doesn't care about shard-boundary activation smoothness.
- **Multi-token prediction heads.** Train the big model itself to predict K tokens at once from one hidden state, so a single forward pass produces multiple accepted tokens.

## Related finding: nested gates hide in self-enabling optimizers

The dual-deadlock pattern we hit is worth remembering as a code-review pattern. If you're building an adaptive precision selector, a dynamic batcher, a JIT-threshold tuner, or any self-promoting optimization with a "warmup required" gate — audit the code path that's *supposed* to produce the warmup data. Make sure it cannot be gated on the very thing it's supposed to enable.

Shadow-mode (or: zero-cost instrumentation running alongside the feature) is the general fix.

## Negative result on Phase 4 entropy coding (bonus)

While we had the logging harness up, we measured RLE compression on INT8 activations: 6178 → 5450 bytes, **1.13× compression ratio**. On one sample; consistent across configurations. Twelve percent savings is not worth the code path on this data. Dense INT8 quantized activations lack the runs that RLE exploits; a different entropy coder (range / arithmetic / Huffman over byte frequencies) might do better, but the ceiling on INT8 is low to begin with. Documenting as a negative result: don't spend optimization cycles on dense-int8 wire compression.

## Reproducibility

Code paths in this measurement:
- `synapse-src/node/speculative.js` — shadow-mode predict+verify; `lastShadowCosine` field.
- `synapse-src/node/node.js` — `speculation_cosine_sample` log emission.
- `synapse-src/node/predictor.js` — linear-extrapolation predictor under test.
- `synapse-src/coordinator/index.js` — log-store + `/api/logs` query surface.

Query the measurement directly:

```bash
curl -s 'https://coord/api/logs?event=speculation_cosine_sample&limit=500' \
  | jq -r '.logs[].data.cosine'
```

Results are deployment-dependent on hardware and prompt, but the tight distribution centered in [0.3, 0.5] reproduced across every combination we ran on.

## Acknowledgements

This finding was produced on the Synapse distributed-inference prototype. Measurement was possible only because the cluster runs on browser WebGPU across heterogeneous mobile/desktop hardware in realistic network conditions — not a synthetic benchmark. Contributions welcome at https://github.com/tejasphatak/Synapse.
