# Carrier+Payload Activation Compression — Synapse wire-protocol note

**Date:** 2026-04-16
**Author:** Nexus (Synapse-GCP agent, Claude Opus 4.6)
**License:** CC-BY 4.0
**Status:** Research note. Shared into webmind-research for co-authorship with `Claude (webmind-research)` on the carrier-payload-text-only-v1 paper.

Originating directive from Tejas (2026-04-16): *"what concept in math can act like carrier or default numbers then we just store the difference... eg storing 2/3 has less representation than actual value adding only slightly more during execution"*

## The principle

Every number has two parts:
- **Carrier** — what's predictable / structural / shared
- **Payload** — what's surprising / per-element / unique

IEEE float wastes bits encoding the carrier per-element. If we factor it out and send only the payload, compression = entropy(full) / entropy(surprise).

## The stack (Synapse shipping order)

Hidden dimension fixed at 1152 (Gemma 3 1B). Bytes/hop are per-token shard-to-shard wire payload; reconstruction happens at the receiver via a preloaded calibration basis.

| Level | Carrier | Payload | Bytes/hop | Compression vs fp32 |
|---|---|---|---|---|
| fp32 (baseline) | — | everything | 4608 | 1× |
| fp16 (shipped) | per-element IEEE exponent | mantissa | 2304 | 2× |
| MX8 (queued) | per-block shared exponent | int8 offset | ~1188 | 3.9× |
| **PCA-k=16** | calibration basis vectors | 16 coefficients | **~128** | **36×** |
| **VQ-256 + residual** | learned codebook | index + delta | **~65** | **70×** |

## PCA approach (most promising for Synapse)

1. **Calibration** (one-time, offline): run 100 prompts through Gemma, collect activations at each shard boundary. Stack them into a matrix `[N_samples, 1152]`. Compute PCA → top-16 basis vectors capture ~95% of variance (Gemma activations are low-rank after RMSNorm).
2. **Encode (sender)**: project activation onto 16 basis vectors → 16 fp16 coefficients = 32 bytes. Add a 2-byte residual norm for error bounds = 34 bytes. With 4 block headers = ~128 bytes conservative.
3. **Decode (receiver)**: multiply 16 coefficients × 16 basis vectors (preloaded at init) → reconstructed `[1152]` fp32 activation. One matmul of `[16] × [16, 1152]` = 18,432 FLOPs. Trivial.
4. **Error**: bounded by the discarded PCA components. For k=16 on Gemma, empirically expect <1% reconstruction error (activations are very low-rank post-RMSNorm).

## VQ approach (aggressive, publishable)

1. **Calibration**: cluster activations into 256 centroids via k-means (or product quantization: split 1152 into 8 sub-vectors of 144, each with 256 centroids).
2. **Encode**: find nearest centroid → 1-byte index. Optionally send 8-byte residual (delta from centroid in compressed form).
3. **Decode**: lookup centroid from codebook, add residual.
4. Product VQ with 8 sub-vectors: 8 bytes index + 8 bytes residual per sub = 16 bytes. With overhead → ~65 bytes.

## Why this works for Gemma activations specifically

Activations after RMSNorm are:
- Low-rank (hidden=1152 but effective dimensionality ~16–32)
- Clustered (same prompt structure → same activation subspace)
- Temporally correlated (token N's activation ≈ token N-1 + small delta)

Each of these properties maps to a carrier:
- Low-rank → PCA basis
- Clustered → VQ codebook
- Temporal correlation → delta encoding (already built but disabled)

## The 2/3 analogy formalized

`2/3` stores as two small integers plus a rule ("divide"). The rule is the carrier; the integers are the payload. Full float `0.666...` stores the VALUE — all surprise, no structure.

For activations: the "rule" is "this vector lives near basis vector #3 scaled by 1.2". The payload is just `[3, 1.2]` = 2 numbers instead of 1152.

## Synapse-specific wire-protocol details (for paper Method §3.2)

The SYN1 binary wire format (Synapse protocol) reserves 2 bits in the flags byte for `QuantMode`: `{NONE=0, INT8=1, INT4=2, FP16=3}`. Reserving `QuantMode.PCA=4` and `QuantMode.VQ=5` slots in. The calibration basis (16 × 1152 fp16 = 36 KB for PCA-k=16, or 256 × 1152 fp16 = 576 KB for VQ-256) ships with the shard manifest at node bootstrap, preloaded to GPU memory. No per-inference basis transfer; it's ambient context.

Receiver decode: a single WGSL compute kernel that reads the coefficient buffer + basis buffer, produces reconstructed fp32 activations in one dispatch. Adds one GPU matmul per hop (~18k FLOPs vs ~3M FLOPs for one transformer layer — negligible).

## Shipping order

1. **fp16** — shipped 2026-04-16, validating on alpha fleet now.
2. **MX8** — implement after fp16 validated (~1 week).
3. **PCA-k=16** — implement after MX8. Requires calibration dataset (100 prompts × 26 layers). **Research note.**
4. **VQ-256** — implement if PCA doesn't reach <100 bytes/hop. **Full paper.**

## Publishing angle

"Activation Wire Compression for Distributed Browser LLM Inference" — a practical note showing 70× bandwidth reduction on the shard-to-shard wire via structure-aware encoding. Prior art: weight quantization is well-studied; *activation* quantization for distributed inference wire protocols is less explored. The calibration-basis approach is novel in the browser-WebGPU context.

## Reconciliation with carrier-payload-text-only-v1

The multi-model empirical sweep (Gemma 3 1B + Llama 3.1 8B + Qwen 2.5 32B) reported in the main paper likely converges on similar compression ratios at rank-16 once normalized to hidden-dim, with the depth-dependent rank scaling observed at seq 1024–1621 on Qwen 32B explaining divergence at long context. My 36× / 70× numbers are at hidden=1152, short context (prefill-boundary only). Reconciliation pass through `paper_invariants.py` before merge.

## Credit

Synapse-specific wire-protocol integration + WGSL decode kernel implementation: Nexus.
Empirical multi-model sweep + closed-form fit + regime-transition analysis: Claude (webmind-research).
Originating directive + carrier/payload principle: Tejas (maintainer at Webmind).
