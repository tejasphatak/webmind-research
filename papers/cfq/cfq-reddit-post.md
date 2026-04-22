# Reddit Post — r/MachineLearning

**Title:** [R] All BF16 LLM weights are dictionary tables (~6,000 unique values). Verified across 0.5B, 7B, 27B params.

**Body:**

Every BF16 language model — regardless of parameter count — stores its weights using only ~4,000-6,000 unique values per tensor. This isn't a compression technique. It's a structural property of the BFloat16 number format (7-bit mantissa constrains the representable values in the typical weight range).

**What we found:**

| Model | Params | Unique BF16 values per layer | Uniqueness ratio |
|-------|--------|------------------------------|-----------------|
| Qwen2-0.5B | 494M | 3,895 | 0.09% |
| Qwen2.5-7B | 7.6B | 5,000-6,062 | 0.007-0.009% |
| Gemma-2-27B | 28.5B | 5,898 | 0.003% |

The entire dictionary is ~8-12 KB. That fits in CPU L1 cache.

**Lossless proof:** We replaced all MLP weights in Qwen2-0.5B with dictionary + int16 index arrays. Reconstruction is bitwise exact (`torch.equal()` = True). 5/5 test prompts produce identical output to the original model. The model literally IS a lookup table.

**We also tested a compression method on top of this: Constant-Fraction Quantization (CFQ).** Decompose each weight into a block constant (mean of 32 weights) + 1-bit sign delta. Applied to Gemma 4 27B: 62.5 GB -> 8.5 GB (7.4x, 1.5 bits/weight). Ran all 60 layers on CPU. **The output is garbage** — 1-bit deltas are too lossy. Signal becomes noise after 60 layers. We report this as a negative result honestly.

**What this doesn't mean:**

- This is NOT a new competitive compression method. GPTQ/AWQ at 4-bit are far more useful.
- The dictionary property doesn't directly save storage (int16 index = same 2 bytes as BF16).
- CFQ at 1.5 bpw doesn't produce usable output. We need 3-4 bit deltas minimum.

**What this might mean:**

- Hardware could potentially exploit this: matmul becomes table lookup + multiply, with the table permanently resident in L1 cache.
- Every BF16 model shares the same ~6K codebook. The "knowledge" isn't in what values are used — it's in where they're placed (the index pattern).
- The constant-fraction decomposition separates the "what" (constant) from the "how much" (fraction) in each weight block. At higher bit-widths this structure might be useful.

Code and paper: github.com/tejasphatak/webmind-research/papers/cfq

Happy to answer questions. We tried not to oversell this — the dictionary finding is real and exact, the compression method needs work.

---

# Suggested subreddits

- r/MachineLearning (primary, [R] tag)
- r/LocalLLaMA (practical LLM compression audience)
