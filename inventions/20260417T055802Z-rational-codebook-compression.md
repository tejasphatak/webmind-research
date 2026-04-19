# Rational Codebook Compression for LLM Weight Transport

**Inventor:** Tejas Phatak
**Date:** 2026-04-17
**Status:** Full stack validated — mixed results, key finding below

## Core Idea

Represent LLM weight tensors using a hashmap (codebook) of rational numbers (p/q as two uint8 integers) instead of full floating-point values. Each weight becomes an index into the codebook. The GPU reconstructs the float via integer division at load time.

## Layered Compression Stack (Tejas's design)

1. **Rational codebook (base):** K-means cluster weights → K centroids stored as (p, q) pairs (2 bytes each vs 4 bytes float32). Index tensor: 1 byte per weight at K=256.

2. **Chained/additive codebooks:** Instead of one K=256 codebook, chain two K=16 codebooks: weight ≈ codebook_A[i] + codebook_B[j]. Stores 32 fraction entries, represents 256 values. Index: 2 × 4 bits = 1 byte. Three-level chain (K=16³ = 4096 representable values) at 48 entries.

3. **Offset residual:** Store a small quantized offset (int4) from the nearest codebook entry. Captures the approximation error from rational rounding. 0.5 bytes per weight.

4. **Delta encoding of indices:** Exploit spatial locality — nearby weights pick similar codebook entries. Store delta from previous index instead of absolute index. Deltas are small → variable-length or 4-bit encoding. Can halve the index tensor.

5. **Gzip on wire:** The delta-encoded index stream has high redundancy → gzip/zstd compresses it further for transport. Decompress at receiver before GPU load.

## Empirical Validation (2026-04-17, Gemma 3 1B)

Rational codebook (layer 1 alone) on 3 representative weight matrices:

| Layer | k=256 MSE ratio (rational/float) | Sample fraction |
|-------|----------------------------------|----------------|
| q_proj L0 | 1.36 | -1/7 (err 1.25e-04) |
| q_proj L12 | 1.48 | -14/145 (err 3.09e-06) |
| gate_proj L25 | 1.29 | -15/179 (err 6.60e-06) |

Rational codebook adds 29-48% MSE over float codebook while halving codebook storage.

## Full Stack Validation (2026-04-17, Gemma 3 1B — 3 layers, 500k weights each)

Tested all 5 layers of the stack plus two telecom-inspired variants (QAM constellation shaping, turbo-VQ iterative refinement). Raw data: `findings/2026-04-17-compression-stack-experiment.json`.

### What worked

| Method | bits/weight | MSE range | vs fp16 |
|--------|-------------|-----------|---------|
| K=256 rational codebook | 8.0 | 6.5–9.7e-7 | 2.0× smaller |
| **Chained 16+16 single-pass** | **8.0** | **2.6–5.6e-7** | **2.0× smaller** |
| K=256 rational + int4 residual | 12.0 | 1.9–4.4e-7 | 1.3× smaller |
| Chained + int4 residual | 12.0 | 2.0–3.9e-7 | 1.3× smaller |
| Chained indices only, gzipped | 6.7–6.9 | 5.2–7.0e-7 | 2.3× smaller |

**Key finding:** Chained single-pass (16+16) at 8 bits/weight achieves *lower MSE* than K=256 at the same bit rate. The additive structure provides a free lunch — 256 representable values from only 32 codebook entries, with better coverage of the weight distribution.

**Int4 residual** is the biggest single-stage MSE win: 57–71% reduction for 4 extra bits. Worth it when quality matters; drop it when bandwidth matters.

### What didn't work (negative results — published per null-results policy)

1. **QAM density-weighted codebook: ~92% WORSE MSE.** LLM weights are near-Gaussian (symmetric, light-tailed). QAM constellation shaping optimizes for non-uniform, heavy-tailed sources (telecom signals). Median-based centroid updates hurt when the distribution is already well-suited to uniform k-means. The analogy is real at the math level but the input distributions don't match.

2. **Turbo-VQ (iterative refinement): 33–191% WORSE than single-pass.** Each iteration undoes part of the other codebook's work. This is the known "codebook collapse" problem in residual VQ when per-codebook K is small (K=16). Turbo decoding works because the two constituent codes are designed to be complementary; our two k-means codebooks are not — they compete for the same variance.

3. **Delta encoding of indices: entropy INCREASED (~9–19%).** K=256 index streams are already near-maximum entropy (~7.96/8.0 bits). Adjacent weights map to uncorrelated codebook entries — there's no spatial locality to exploit at the scalar level. (Block-level delta encoding on weight *tiles* might work; scalar delta does not.)

4. **Gzip on raw index stream: <1% compression.** Same reason — near-uniform distribution of 256 symbols. Gzip needs structure (repeats, patterns) and finds almost none. Gzip only helps on the chained indices (6.6–6.9 bits entropy < 8 bits) or when combined with int4 residuals.

5. **Full 5-layer stack (delta + int4 + gzip): 2.6× vs fp32.** Underwhelming. The residual and metadata overhead eat into the codebook savings. The *simpler* path (chained codebook alone, gzipped = 6.7 bits/weight, 4.8× vs fp32) beats the full stack on compression ratio.

### Lessons

- Simpler beats clever here. Chained single-pass > turbo-VQ. Plain k-means > density-weighted.
- The rational representation is a *transport* optimization (smaller codebook on wire), not a quality optimization.
- For Synapse: ship chained codebook indices at 6.7 bits/weight for max bandwidth savings, add int4 residual only if quality degrades on downstream tasks.
- QAM/turbo analogy to telecom is structurally correct but fails empirically because LLM weight distributions ≠ telecom signal distributions.

## Why This Matters for Synapse

Synapse distributes model shards to volunteer devices over residential internet. Model download is the first bottleneck a new node hits. A 5-layer compression stack could reduce a 2 GB model shard to ~500 MB on the wire with minimal quality loss, and the rational representation can be decoded in WGSL (one integer division per weight in the compute shader).

## Prior Art to Differentiate

- GPTQ, AWQ: weight quantization but float codebooks, no rational representation
- AQLM (2024): additive codebook quantization — closest prior art, but uses float centroids
- Product quantization: sub-vector codebooks — orthogonal, can be combined
- This work: rational (fraction) codebook entries + chained additive structure + delta encoding

## Next Steps

- [x] Measure layers 2-5 of the stack empirically — DONE 2026-04-17
- [x] Measure gzip compression ratio on delta-encoded index streams — DONE (near-zero gain)
- [x] Test QAM constellation shaping — DONE (negative result)
- [x] Test turbo-VQ iterative refinement — DONE (negative result)
- [ ] Implement WGSL decode kernel (p/q → float, chained codebook lookup)
- [ ] Compare end-to-end against GPTQ int4 on perplexity + model size
- [ ] Test block-level (tile) delta encoding instead of scalar delta
- [ ] Measure perplexity impact of chained-codebook-only (6.7 bpw) path on downstream tasks
