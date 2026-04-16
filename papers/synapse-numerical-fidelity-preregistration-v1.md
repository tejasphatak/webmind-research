# Pre-registration — Synapse Numerical Fidelity Under Heterogeneous Precision Shard Boundaries

**Pre-registered:** 2026-04-16
**Status:** Locked before data collection.
**Author:** Claude Opus 4.6 (Synapse Research)
**Companion artifacts:** `findings/` (to be filed post-collection), `tools/synapse_fidelity_sim.py` (to be committed *before* running).

---

## Motivation

Synapse (webmind.sh) executes LLM inference by sharding a model across volunteer WebGPU devices and streaming activations across shard boundaries. The project's public-facing claim is that distributed output is **"99.99% numerically identical"** to a reference implementation. That number has no published derivation, no distribution, and no breakdown by component. It is a marketing artifact, not a scientific one.

This pre-registration locks the methodology for replacing that claim with a measured distribution, plus a layer-resolved drift breakdown, plus a novel measurement that the literature does not yet report: **compounding drift under heterogeneous precision-hopping across shard boundaries** (e.g., Node A `bf16` → Node B `fp16` → Node C `fp32`).

## Hypotheses (locked)

### H1 — Uniform-precision baseline reproduces literature
Under uniform `fp16` or `bf16` across all shards, layer-wise cosine similarity vs `fp32` reference stays above 0.999 at every layer of Gemma 3 1B IT. Final-logit KL divergence vs `fp32` ≤ 0.01 nats on 512 prompts drawn from the C4 validation split.

*Purpose:* sanity check. If H1 fails, instrumentation is broken.

### H2 — Heterogeneous precision-hopping produces super-linear error amplification
When shards use *different* precisions in sequence (the realistic Synapse case — a phone runs `fp16`, a desktop runs `bf16`, another desktop runs `fp32`), layer-wise drift exceeds the max of the corresponding uniform-precision drifts by a factor of ≥ 1.5× at the final layer.

*Purpose:* this is the load-bearing novel claim. If true, "99.99%" is defensible only in homogeneous deployments and Synapse should pin precision per-request. If false, Synapse can freely mix precisions and the heterogeneity question is closed.

### H3 — Activation outliers are the dominant drift source
The top-1% of activations by magnitude ("outliers", well-documented in LLM.int8() and SmoothQuant literature) account for ≥ 50% of the L2 drift at every attention output and MLP down-projection boundary.

*Purpose:* prescriptive. If true, a lightweight fix exists (outlier-preserving precision casts at shard boundaries only) and Synapse gets a cheap 10× fidelity improvement without full `fp32`.

### H4 (null-pre-registered) — Drift is negligible and the claim is defensible as stated
If final-layer cosine similarity vs `fp32` exceeds 0.9999 across every tested precision combination on 512 prompts, we report a null: the "99.99%" claim holds, the heterogeneity concern is academic, and further work on this axis is deprioritized. **This is the outcome we pre-commit to publishing even if unexciting** (repo policy: null results are first-class).

## Method

### Model
- `google/gemma-3-1b-it` (HuggingFace). Weights loaded via `transformers` at `fp32` as the oracle.
- Reference implementation: `fp32` single-device forward pass, no sharding.

### Shard topology (simulated)
- Per-layer shard boundary. Gemma 3 1B has 26 transformer blocks → 27 shard boundaries including embedding and final-layer-norm.
- Each shard boundary is the point at which a volunteer device's output would be serialized, transmitted, and deserialized on the next device. In simulation, this corresponds to casting the hidden state to the next shard's working precision.

### Precision regimes tested
1. **Uniform baselines:** all-`fp32`, all-`bf16`, all-`fp16`. (Establishes H1.)
2. **Heterogeneous sequences (H2):** every length-3 block-cycle over `{fp16, bf16, fp32}` — 27 cycles, covers realistic multi-device paths.
3. **Adversarial sequence:** `fp16` on the two layers with the highest empirical outlier density (identified from uniform run), `bf16` elsewhere. Tests whether outlier placement is predictable.

### Metrics (locked, per layer and at final logits)
- Cosine similarity vs `fp32` reference.
- Relative L2: `||x - x_ref||_2 / ||x_ref||_2`.
- Max absolute deviation per hidden dim.
- Fraction of top-1% outlier positions that drift by > 5% relative.
- At final logits only: KL divergence vs `fp32`, top-1 argmax agreement rate, top-5 overlap.

### Dataset
- 64 prompts from C4 validation split, truncated to 64 tokens. Seed `0xC0FFEE`.
- Seed fixed and committed before collection.
- **Scope note (2026-04-16):** reduced from 512 prompts × 128 tokens to 64 × 64 because the execution environment is CPU-only (`torch.cuda.is_available() == False`). Pre-registration amended in same commit as sim code, before any data collection. If future compute becomes available, scope restoration is a mechanical re-run; hypotheses and metrics are unchanged.

### Runs
- 1 trial per precision-regime. Variance across prompts approximates across-trial variance (reported as inter-prompt std).
- **Scope note:** reduced from 3 trials for same CPU-budget reason. Reported as a known limitation, not a silent change.

## Analysis plan (locked)

1. Report H1 verdict first. If H1 fails → halt, fix instrumentation, re-commit pre-registration amendment.
2. Compute H2 amplification factor: `drift_hetero / max(drift_uniform_components)` at final logit layer. Threshold 1.5× is the pre-committed decision boundary.
3. For H3, rank layers by outlier-contribution-to-drift; report top-5.
4. For H4: if all cosine-sim > 0.9999, report null and close.

## What would falsify / confirm each hypothesis

| Hypothesis | Confirms | Falsifies |
|---|---|---|
| H1 | cosine ≥ 0.999 layer-wise, KL ≤ 0.01 | otherwise |
| H2 | amplification factor ≥ 1.5× | factor < 1.5× |
| H3 | outliers account for ≥ 50% of L2 drift | < 50% |
| H4 (null) | all cosine > 0.9999 | any regime below 0.9999 |

## Stopping criteria

- Compute budget: ≤ 4 hours wall-clock on CPU. If exceeded, halt and report partial results with explicit coverage gaps.
- Data quality: if NaN/Inf observed in any precision regime, report immediately as a finding and halt that regime.

## Limitations acknowledged up front

- **No real WebGPU hardware in the loop.** Simulation uses PyTorch precision casts as a proxy for Mali/Adreno/Apple/desktop rounding differences. Real hardware will differ; this study is necessary but not sufficient. A follow-up with real browser measurements is listed as future work.
- **Single model, single size.** Gemma 3 1B IT only. Scaling to 7B+ may change the outlier picture materially (outliers grow super-linearly with scale per SmoothQuant). Explicitly out of scope for this pre-reg.
- **No adversarial precision faults.** A byzantine volunteer could claim `fp16` and return `int8` outputs. That's Artifact 2's problem (Byzantine verification), not this one.
- **Inference only, no training.** Claim does not extend to distributed training.

## Deliverables

- `tools/synapse_fidelity_sim.py` — reference implementation, committed before data collection.
- `findings/synapse-numerical-fidelity-<timestamp>.md` — results, written post-collection, referencing this pre-reg by commit SHA.
- `plots/synapse_fidelity_*.png` — drift curves.

## Relation to Nexus trajectory

Tejas has stated (2026-04-16) that Synapse is phase 1 of a longer plan: a continuously self-learning distributed model ("Nexus"). This pre-registration is scoped strictly to inference-time numerical fidelity — no training, no gradient accumulation. If Nexus proceeds, the equivalent question for gradient drift across heterogeneous-precision shards becomes the more important study; that is future work, not this study. The inference result is still a necessary prerequisite: inference drift bounds gradient drift from below.

## Disclosure

This pre-registration and all downstream code/analysis is produced by a Claude Opus 4.6 instance under human direction, consistent with repo policy.
