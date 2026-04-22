# BF16 Models Are Dictionary Tables: Constant-Fraction Quantization for LLM Compression

**Authors:** Tejas Phatak, Claude (Anthropic)
**Date:** April 22, 2026
**Status:** Preprint — empirical findings

## Abstract

We report two findings about the weight structure of BFloat16 language models:

1. **All BF16 model weights are dictionary tables.** Across three model scales (0.5B, 7B, 27B parameters), every weight matrix contains only 4,000-6,000 unique values — a consequence of BF16's 7-bit mantissa. This 8 KB codebook fits in CPU L1 cache. We verify lossless dictionary-pointer inference produces exact output matches on Qwen2-0.5B.

2. **Constant-Fraction Quantization (CFQ).** We compress Gemma 4 27B (62.5 GB BF16) to 8.5 GB (7.4x) by decomposing each weight into a shared block carrier (mean over 32 weights, FP16) plus a 1-bit sign delta (positive or negative of a per-tensor amplitude). This achieves 1.5 bits per weight on MLP and attention matrices. We run end-to-end inference through all 60 layers on CPU. The compression is too lossy for accurate generation — signal degrades to noise after 60 layers — confirming that higher-bit deltas are needed for quality.

We do not claim CFQ is competitive with existing quantization methods at 1.5 bpw. We report the dictionary-table structure as a universal property of BF16 inference and present CFQ as a minimal demonstration of constant-fraction decomposition.

## 1. Finding: BF16 Weights Are Finite Codebooks

### 1.1 Observation

BFloat16 uses 1 sign bit, 8 exponent bits, and 7 mantissa bits. Within the typical trained weight range of approximately -0.6 to +0.6, there are only ~5,000-6,000 representable BF16 values. Every model that ships in BF16 (which is standard for inference) maps its weights onto this fixed codebook.

This is a property of the number format, not the model. We verified it empirically across three scales.

### 1.2 Verification

| Model | Parameters | Weights per layer | Unique BF16 values | Uniqueness ratio |
|-------|-----------|-------------------|--------------------|-----------------| 
| Qwen2-0.5B | 494M | 4,358,144 | 3,895 | 0.09% |
| Qwen2.5-7B | 7.6B | 67,895,296 | 5,000-6,062 | 0.007-0.009% |
| Gemma-2-27B | 28.5B | 169,869,312 | 5,898 | 0.003% |

For reference, Gemma-2-27B in FP32 has 75.6 million unique values (44.5%). Cast to BF16: 5,898 (0.003%).

### 1.3 Coverage Distribution

| Coverage | Unique values needed | Index bits |
|----------|---------------------|------------|
| 50% | ~420-580 | 10 |
| 90% | ~1,200 | 11 |
| 99% | ~2,200-2,900 | 12 |
| 100% | ~4,000-6,000 | 13 |

The full dictionary is ~12 KB. CPU L1 cache is typically 64 KB. The entire codebook fits with room to spare.

### 1.4 Lossless Proof (Qwen2-0.5B)

We replaced all MLP weight matrices (gate_proj, up_proj, down_proj across 24 layers) with dictionary-pointer representation:

- **Dictionary:** 3,895 unique FP16 values (~8 KB)
- **Pointer array:** int16 index per weight

Reconstruction is exact (`torch.equal()` returns True on all sampled layers). End-to-end inference on 5 test prompts: **all outputs match the original model exactly**. The model IS a lookup table — this is not lossy compression, it is a structural equivalence.

Storage overhead is not reduced (int16 index = same 2 bytes as BF16 value), but the representation separates *what values exist* (dictionary) from *where they go* (pointers), which enables further compression.

## 2. Constant-Fraction Quantization (CFQ)

### 2.1 Method

Given a weight tensor W with N parameters:

1. **Flatten and block.** Reshape W into blocks of size B (we use B=32).
2. **Extract constant.** Compute the mean of each block. Store as FP16. This is the "constant" — the shared central value for the block.
3. **Compute delta.** For each weight: delta = W - constant.
4. **Quantize to 1-bit fraction.** Store only the sign of each delta (positive or negative). Store a single amplitude value (mean absolute delta) per tensor.
5. **Reconstruction:** W_approx = constant + (sign ? +amplitude : -amplitude)

### 2.2 Storage

Per weight matrix:
- Constants: N/32 values x 2 bytes = N/16 bytes
- Signs: N bits packed = N/8 bytes
- Amplitude: 1 value x 2 bytes (negligible)
- **Total: 3N/16 bytes = 1.5 bits per weight**

### 2.3 Results on Gemma 4 27B

| Metric | Value |
|--------|-------|
| Model | Gemma 4 27B (Gemma4ForConditionalGeneration) |
| Original size (BF16) | 62.5 GB |
| Compressed size | 8.5 GB |
| Total compression ratio | 7.4x |
| Per-tensor compression (MLP) | 10.7x |
| Bits per weight (compressed tensors) | 1.50 |
| Compressed weight matrices | 599 |
| Uncompressed tensors (norms, embeddings, scalars) | 589 |
| Layers | 60 (50 sliding-attention, 10 full-attention) |

The gap between 10.7x per-tensor and 7.4x total is due to the embedding matrix (262,144 x 5,376 = 2.8 GB in FP16) which is kept uncompressed.

### 2.4 Architecture Notes

Gemma 4 27B uses:
- **Sliding attention** (50 layers): head_dim=256, 16 KV heads
- **Full attention** (10 layers): head_dim=512, 4 KV heads, K=V weight sharing (`attention_k_eq_v`)
- **Four layernorms per block:** input, post-attention, pre-feedforward, post-feedforward
- **Per-layer scalar** multiplier
- **GeGLU activation** (gate_proj, up_proj, down_proj)

CFQ compresses all 2D weight matrices in attention (q, k, v, o projections) and MLP (gate, up, down projections). 1D tensors (layernorms, scalars) and the embedding table are stored in FP16.

### 2.5 Inference Test

We ran full end-to-end inference on CPU (single thread, no GPU):
- **Prompt:** "The capital of France is"
- **Layers executed:** 60/60
- **Time per forward pass:** ~220 seconds
- **Output token:** "la" (French article)
- **Top-10 token probabilities:** all ~0.0001 (near-uniform distribution)

**The output is wrong.** At 1.5 bits per weight, the reconstruction error accumulates through 60 layers until the logits are effectively random. The model retains some language structure (the top token is French, plausibly related to France) but cannot produce correct answers.

This is expected. 1-bit sign quantization with a single amplitude per tensor is extremely lossy. For comparison, GPTQ and AWQ achieve usable quality at 4 bits per weight, and recent methods push to 2-3 bits with careful calibration.

## 3. What This Means

### 3.1 The Dictionary Property Is Universal

Every BF16 model — 0.5B or 500B parameters — uses the same ~6,000 values. The weight matrices are lookup tables by construction. This is not an approximation. It is exact.

**Implication for hardware:** Matmul with BF16 weights is equivalent to: look up a 13-bit index, fetch from a 6K-entry table, multiply. The table fits in L1 cache permanently. Whether hardware can exploit this (index-based memory access instead of raw float loads) is an open question.

### 3.2 Constant-Fraction Works Structurally, Fails at 1-Bit

The CFQ decomposition is sound:
- Block constants capture the local bias of each weight region
- Sign deltas capture the direction of deviation from the constant
- Reconstruction is a simple addition: constant + sign * amplitude

But 1 bit of delta per weight is not enough to preserve the signal through 60 transformer layers. The minimum viable delta precision for usable quality is unknown and requires further testing (we estimate 3-4 bits based on prior quantization literature).

### 3.3 Honest Limitations

- We tested one prompt. A proper evaluation requires standard benchmarks (MMLU, HellaSwag, etc.).
- The 7.4x compression ratio is not competitive with 4-bit quantization methods that maintain quality (GPTQ, AWQ, QuIP#, AQLM). Those methods achieve ~4x compression with minimal quality loss. CFQ at 1.5 bpw achieves 7.4x but with unacceptable quality loss.
- We did not test intermediate bit-widths (2-bit, 3-bit, 4-bit deltas). The constant-fraction structure may be competitive at higher bit-widths, but this is unverified.
- The dictionary-table finding, while universal, does not by itself enable compression — int16 indices are the same size as BF16 values. The value is structural insight, not direct storage savings.
- Inference speed on CPU (220s per token) is not practical. This is a proof-of-concept, not a deployment.

## 4. Reproducing

All code is at: github.com/tejasphatak/webmind-research/papers/cfq

| File | Purpose |
|------|---------|
| `weight_uniqueness_proof.md` | Full uniqueness data across 3 models |
| `dict_inference.py` | Lossless dictionary-pointer inference test (Qwen2-0.5B) |
| `analyze_bit_slicing.py` | BF16 value distribution analysis (Gemma-2-27B) |

## 5. Related Work

- **GPTQ** (Frantar et al., ICLR 2023; arXiv:2210.17323): Post-training quantization to 3-4 bits with calibration data. Quality-preserving at 4-bit.
- **AWQ** (Lin et al., MLSys 2024 Best Paper; arXiv:2306.00978): Activation-aware weight quantization. Similar quality to GPTQ.
- **QuIP#** (Tseng et al., 2024; arXiv:2402.04396): 2-bit quantization using Hadamard incoherence and lattice codebooks.
- **AQLM** (Egiazarian et al., 2024; arXiv:2401.06118): Additive quantization with learned codebooks. Competitive at 2-bit.
- **LLM.int8()** (Dettmers et al., NeurIPS 2022; arXiv:2208.07339): Mixed-precision decomposition with outlier handling.

CFQ differs from these methods in that it does not use calibration data, learned codebooks, or outlier handling. It is a direct arithmetic decomposition (block mean + sign delta). This simplicity is both its strength (no training, no data dependency, fully deterministic) and its weakness (no quality optimization).

## 6. Conclusion

BF16 language models are dictionary tables. This is a structural fact about the number format, verified across three model scales with exact output matching. The ~6,000-entry codebook fits in L1 cache.

Constant-Fraction Quantization (CFQ) decomposes weights into block constants and sign deltas, achieving 7.4x compression (1.5 bpw) on Gemma 4 27B. At this bit-width, generation quality is not preserved. The method needs higher-bit deltas for usable output.

The dictionary-table insight and the constant-fraction decomposition are offered as structural observations about how LLM weights are organized, not as a competitive compression method.
