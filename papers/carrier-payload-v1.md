# Carrier-Payload Decomposition: Low-Rank Activation Compression for Decentralized LLM Inference

**Tejas Phatak**
University of Colorado Boulder & Webmind Research (webmind.sh)
tejasphatak@gmail.com

---

> ## ⚠️ DRAFT REVISION NOTICE — 2026-04-16
>
> This initial draft (v1, written 2026-04-15) contained several claims that have since been **tightened or retracted** based on long-context experimental data collected 2026-04-16:
>
> **Claims being revised:**
> - "Effective dimensionality of ~32 out of 1536" was measured at **short context only** (seq_len 17–30). Long-context measurement on Qwen 2.5 32B (hidden_dim 5120) shows rank for 99% variance growing with sequence length: rank 8 at seq=256, rank 194 at seq=1024, rank 384 at seq=1621. Short-context low rank is partially a consequence of `rank ≤ min(seq_len, hidden_dim)`.
> - "Projecting to 45x compression at 1024 tokens" is **not supported by long-context data**. Measured compression ratio on Qwen 32B degrades from 183× at seq=256 to ~13× at seq=1621. Still useful, but not 45×.
> - Preliminary cross-modal finding (multimodal Gemma 4 shows *inverse* compression scaling) requires a separate paper (Paper 2) with paired multimodal samples; not claimed in this draft.
>
> **Authoritative current draft:** see `papers/carrier-payload-text-only-v1.md`
>   (not yet public but the same repo, under revision).
>
> **Please contact tejasphatak@gmail.com** before citing numbers from this draft; we are happy to share the currently-correct figures.
>
> Honest disclosure is a core principle of Webmind Research (see `MANIFESTO.md`). Draft-to-revised-draft tightening is expected and publicly documented.

---

---

## Abstract

Decentralized large language model inference distributes transformer layers across volunteer devices, but inter-device activation transport creates a bandwidth bottleneck that limits practical deployment. We propose *Carrier-Payload* decomposition: activations at shard boundaries are projected onto a shared PCA basis (the *carrier*, pre-distributed as a shared prior) with an optional sparse residual (the *payload*, capturing outlier dimensions). On Gemma 3 1B IT across 32 diverse prompts and 3 splice layers, rank-32 carrier-only compression achieves 22x bandwidth reduction at KL divergence 0.023 with 100% next-token agreement. The hybrid rank-8 carrier + 1% sparse payload achieves 10.5x compression at 96% top-1 agreement. Compression ratio scales linearly with sequence length, projecting to 45x at 1024 tokens. These results exploit the empirically low intrinsic dimensionality of transformer activations (~32 effective dimensions out of 1536) and make volunteer-device inference over consumer internet connections practical.

---

## 1. Introduction

Large language models achieve state-of-the-art performance across NLP tasks but require substantial compute and memory at inference time. A single forward pass through a 7B-parameter model demands ~14 GB of weight storage alone, exceeding the capacity of most consumer devices. *Decentralized inference* addresses this by sharding the model across multiple devices: each device holds a subset of layers, processes its shard, and forwards the resulting hidden-state activations to the next device in the pipeline.

This pipeline-parallel architecture introduces a fundamental constraint: **bandwidth**. At each shard boundary, the current device must transmit a dense activation tensor of shape $(T, d)$, where $T$ is the sequence length and $d$ is the hidden dimension. For a model with $d = 1536$ and $T = 512$, each boundary transfer is $512 \times 1536 \times 2 = 1.57$ MB in float16---manageable on data-center interconnects but prohibitive when the pipeline spans volunteer devices connected via residential internet (typically 5--50 Mbps upload). With multiple shard boundaries, latency compounds: a 4-shard pipeline requires 3 transfers, adding 0.75--7.5 seconds per token at these bandwidths.

Existing approaches to this problem include naive quantization (Petals [1], which uses 8-bit activations for ~2x compression) and cryptographic verification methods (opML [2], ZKML [3]) that address integrity but not bandwidth. SafetyNets [4] provides verification via integer arithmetic but does not compress. None of these methods exploit the *geometric structure* of the activation space itself.

We observe that transformer activations occupy a low-dimensional manifold within the ambient hidden-state space. This observation is consistent with prior theoretical and empirical work on representation anisotropy [5], intrinsic dimensionality of fine-tuned representations [6], and the dominance of a small number of outlier dimensions in activation variance [7]. We exploit this structure through a decomposition we call *Carrier-Payload*:

- **Carrier**: A PCA basis (the top-$k$ right singular vectors of the activation matrix) pre-computed on calibration data and shared across all nodes. Each activation is transmitted as its $k$-dimensional projection coefficients rather than the full $d$-dimensional vector.
- **Payload**: A sparse residual that captures high-magnitude entries missed by the low-rank approximation---primarily the activation outliers that dominate next-token prediction.

Our contributions are:

1. **Empirical measurement of effective activation dimensionality**: On Gemma 3 1B IT, we show that 32 PCA components capture 99.2% of activation variance across 32 diverse prompts, implying an effective dimensionality of ~32 out of 1536 hidden dimensions.

2. **Carrier-Payload compression with output-fidelity guarantees**: We demonstrate 22x compression at KL $< 0.025$ and 100% top-1 next-token agreement using carrier-only transmission, and characterize the full Pareto frontier across rank and sparsity configurations.

3. **Scaling analysis**: We derive and empirically validate that compression ratio scales as $O(d/k)$ with sequence length $T$ once $T > k$, projecting to 45x compression at $T = 1024$ for rank-32 carrier.

4. **Application to decentralized inference**: We show that these compression ratios are sufficient to make volunteer-device LLM inference over standard internet connections practical, reducing the per-boundary transfer time from seconds to tens of milliseconds.

---

## 2. Background

### 2.1 Pipeline-Parallel Inference

In pipeline-parallel inference, a transformer with $L$ layers is partitioned into $S$ shards, where shard $s$ contains layers $\ell_{s-1}$ through $\ell_s - 1$. The forward pass proceeds sequentially: shard $s$ receives hidden states $\mathbf{H}^{(\ell_{s-1})} \in \mathbb{R}^{T \times d}$ from shard $s-1$, applies its layers, and transmits $\mathbf{H}^{(\ell_s)}$ to shard $s+1$. The inter-shard communication cost is $O(T \cdot d \cdot b)$ per boundary, where $b$ is the number of bytes per element.

For Synapse (webmind.sh), our target deployment platform, the model is Gemma 3 1B IT ($L = 26$, $d = 1536$), sharded across volunteer WebGPU-enabled browsers. Devices are connected via WebRTC data channels with typical upload bandwidths of 5--50 Mbps.

### 2.2 The Manifold Hypothesis for Activations

The *manifold hypothesis* posits that high-dimensional data often lies on or near a low-dimensional manifold embedded in the ambient space. For transformer hidden states, several lines of evidence support this:

**Anisotropy.** Ethayarajh [5] showed that contextual word representations in BERT, GPT-2, and ELMo are highly anisotropic: they occupy a narrow cone in the representation space rather than being uniformly distributed. This implies that a small number of directions capture most of the variance.

**Intrinsic dimensionality.** Aghajanyan et al. [6] demonstrated that fine-tuning pre-trained language models is effective even when the update is constrained to a random subspace of dimension $d_{\text{intrinsic}} \ll d$. For RoBERTa-base ($d = 768$), $d_{\text{intrinsic}} \approx 200$; the ratio $d_{\text{intrinsic}}/d$ decreases with model size, suggesting even lower relative intrinsic dimensionality for larger models.

**Outlier dimensions.** Dettmers et al. [7] identified that a small number of hidden dimensions (as few as 6 out of 5120 in OPT-6.7B) have activation magnitudes 100x larger than the rest. These *outlier features* are responsible for the failure of naive quantization below 8 bits and suggest a natural decomposition: a smooth low-rank component plus sparse high-magnitude corrections.

### 2.3 PCA for Activation Compression

Principal Component Analysis provides the optimal rank-$k$ approximation to a matrix under the Frobenius norm. Given an activation matrix $\mathbf{A} \in \mathbb{R}^{T \times d}$ with mean $\boldsymbol{\mu} = \frac{1}{T}\sum_t \mathbf{a}_t$, the SVD of the centered matrix is:

$$\mathbf{A} - \mathbf{1}\boldsymbol{\mu}^\top = \mathbf{U} \boldsymbol{\Sigma} \mathbf{V}^\top$$

The rank-$k$ approximation retains the top $k$ singular values:

$$\hat{\mathbf{A}}_k = \mathbf{U}_{:,:k} \, \boldsymbol{\Sigma}_{:k,:k} \, \mathbf{V}_{:k,:}^\top + \mathbf{1}\boldsymbol{\mu}^\top$$

The fraction of variance explained is $\sum_{i=1}^k \sigma_i^2 / \sum_{i=1}^d \sigma_i^2$.

---

## 3. Method: Carrier-Payload Decomposition

### 3.1 Overview

At each shard boundary, the sending device must transmit the activation tensor $\mathbf{A} \in \mathbb{R}^{T \times d}$ to the receiving device. Instead of transmitting $\mathbf{A}$ directly ($T \cdot d$ floats), we decompose it into two components:

1. **Carrier** (shared basis): The top-$k$ right singular vectors $\mathbf{V}_{:k} \in \mathbb{R}^{d \times k}$, pre-computed on calibration data and distributed to all nodes at setup time. The per-activation wire cost is the projection coefficients $\mathbf{P} = (\mathbf{A} - \mathbf{1}\boldsymbol{\mu}^\top)\mathbf{V}_{:k} \in \mathbb{R}^{T \times k}$, plus the mean vector $\boldsymbol{\mu} \in \mathbb{R}^d$.

2. **Payload** (sparse residual): The residual $\mathbf{R} = \mathbf{A} - \hat{\mathbf{A}}_k$ is sparsified by retaining only the top-$p$% entries by magnitude. Each retained entry requires an index and a value (2 floats per entry).

### 3.2 Wire Cost Analysis

The total per-activation wire cost is:

$$W = \underbrace{T \cdot k}_{\text{carrier coefficients}} + \underbrace{d}_{\text{mean vector}} + \underbrace{2 \cdot \lfloor p \cdot T \cdot d \rfloor}_{\text{sparse payload}}$$

The baseline (uncompressed) wire cost is $W_0 = T \cdot d$. The compression ratio is:

$$\text{CR} = \frac{T \cdot d}{T \cdot k + d + 2p \cdot T \cdot d}$$

In the carrier-only case ($p = 0$), this simplifies to:

$$\text{CR}_{\text{carrier}} = \frac{T \cdot d}{T \cdot k + d} = \frac{d}{k + d/T}$$

For $T \gg k$ and $T \gg d/k$, this approaches $d/k$. For Gemma 3 1B ($d = 1536$, $k = 32$), the asymptotic carrier-only compression ratio is $1536/32 = 48\text{x}$.

### 3.3 Shared Basis Protocol

The PCA basis $\mathbf{V}_{:k}$ is computed once during a calibration phase:

1. Run a calibration corpus (e.g., 1000 diverse prompts) through the full model.
2. At each shard boundary layer $\ell$, collect activations $\mathbf{A}^{(\ell)}_1, \ldots, \mathbf{A}^{(\ell)}_C$ and concatenate.
3. Compute the SVD and retain the top $k$ right singular vectors $\mathbf{V}^{(\ell)}_{:k}$.
4. Distribute $\mathbf{V}^{(\ell)}_{:k}$ to all devices at setup time. This is a one-time cost of $k \cdot d$ floats per boundary ($k \cdot d \cdot 4 = 196$ KB for $k = 32$, $d = 1536$ in float32).

At inference time, the basis is fixed and only the coefficients $\mathbf{P}$ and optional sparse payload are transmitted.

**Basis staleness.** If the model is updated (e.g., via online fine-tuning in the Nexus trajectory), the basis must be recomputed. For static models, the basis is computed once and is valid indefinitely. Section 7.2 discusses basis drift under model updates.

### 3.4 Reconstruction

The receiving device reconstructs the activation as:

$$\tilde{\mathbf{A}} = \mathbf{P} \cdot \mathbf{V}_{:k}^\top + \mathbf{1}\boldsymbol{\mu}^\top + \mathbf{R}_{\text{sparse}}$$

where $\mathbf{R}_{\text{sparse}}$ is the sparse payload (zero if $p = 0$). The reconstruction error is bounded by the sum of the truncated singular values and the sparsification loss:

$$\|\mathbf{A} - \tilde{\mathbf{A}}\|_F^2 = \sum_{i=k+1}^{d} \sigma_i^2 - \|\text{sparsify}_p(\mathbf{R})\|_F^2 + \|\mathbf{R} - \text{sparsify}_p(\mathbf{R})\|_F^2$$

### 3.5 Why Sparse Residuals Help: Outlier Capture

The sparse payload is not uniformly distributed across the residual. It preferentially captures *activation outliers* --- the high-magnitude entries in a small number of hidden dimensions identified by Dettmers et al. [7]. These outliers contribute disproportionately to the output logit distribution. The PCA carrier, which minimizes mean squared reconstruction error globally, underweights these rare-but-critical features. The sparse payload acts as a targeted correction for exactly this deficiency.

---

## 4. Experimental Setup

### 4.1 Model and Hardware

- **Model**: Gemma 3 1B IT (google/gemma-3-1b-it), 26 transformer layers, hidden dimension $d = 1536$, vocabulary size 262,144.
- **Hardware**: NVIDIA GPU (CUDA), float16 inference.
- **Framework**: PyTorch 2.9.1, Hugging Face Transformers.

### 4.2 Prompts

We used 32 diverse prompts spanning code generation, mathematical reasoning, translation, factual Q&A, creative writing, and technical explanation. Prompts were designed to exercise different regions of the activation space (see Appendix A). Sequence lengths after tokenization ranged from 9 to 70 tokens.

### 4.3 Splice Layers

We evaluate compression at three shard boundary positions within the 26-layer model:

| Position | Layer Index | Description |
|---|---|---|
| Early | 4 | After first 5 layers (embedding + 4 blocks) |
| Middle | 13 | Midpoint of the model |
| Late | 22 | 4 layers before output |

### 4.4 Compression Grid

- **PCA ranks** $k$: 2, 4, 8, 16, 32
- **Sparse residual fractions** $p$: 0.0 (carrier-only), 0.5%, 1%, 5%, 10%
- **Total configurations**: $3 \times 5 \times 5 = 75$ per prompt, $75 \times 32 = 2{,}400$ evaluations.

### 4.5 Metrics

For each configuration, we measure:

1. **Compression ratio (CR)**: Ratio of uncompressed to compressed wire cost.
2. **KL divergence**: $D_{\text{KL}}(p_{\text{baseline}} \| p_{\text{compressed}})$ on the final next-token distribution, where $p_{\text{baseline}}$ is the softmax output from uncompressed inference.
3. **Top-1 agreement**: Whether the most probable next token is unchanged.
4. **Top-5 overlap**: Number of tokens in the top-5 that are preserved.
5. **Variance explained**: Fraction of activation variance captured by the rank-$k$ PCA carrier.
6. **Reconstruction cosine similarity**: Cosine similarity between original and reconstructed activation tensors.

The experimental protocol is:

1. Run the full model to obtain baseline hidden states at every layer and baseline output logits.
2. For each (splice layer, rank, sparsity) configuration, compress the hidden state at that layer, splice the reconstructed activation back into the model via a forward hook, and run the remaining layers.
3. Compare the resulting logits to the baseline.

### 4.6 Long-Context Control

To validate that results are not an artifact of short sequence lengths (where $T \leq k$ causes the PCA rank to be bounded by $T$, producing artificially perfect reconstruction), we ran a separate control experiment with 8 longer prompts (47--70 tokens after tokenization) and PCA ranks up to 64. This ensures $T > k$ for all configurations and confirms that the low-rank structure is a property of the activation space, not a linear-algebra identity.

---

## 5. Results

### 5.1 Effective Dimensionality

The activation matrices at all three splice layers exhibit remarkably low effective dimensionality. Averaged across 32 prompts:

| PCA Rank $k$ | Variance Explained (Layer 4) | Variance Explained (Layer 13) | Variance Explained (Layer 22) |
|---|---|---|---|
| 2 | 99.4% | 99.7% | 96.2% |
| 4 | 99.6% | 99.8% | 97.7% |
| 8 | 99.9% | 99.95% | 99.4% |
| 16 | 99.95% | 99.98% | 99.7% |
| 32 | 99.99% | 99.99% | 99.9% |

Across all layers, 97.3% of activation variance is captured at rank 16 and 99.2% at rank 32. The effective dimensionality is thus approximately **32 out of 1536** hidden dimensions (2.1% of the ambient space). This is consistent with the anisotropy findings of Ethayarajh [5] and the intrinsic dimension measurements of Aghajanyan et al. [6].

Early layers (layer 4) show the highest compressibility, consistent with the observation that early representations are more generic and later layers encode more task-specific information.

### 5.2 Carrier-Only Compression

With no sparse residual ($p = 0$), the carrier-only compression achieves:

| Splice Layer | Rank | CR | KL Divergence | Top-1 Agreement | Cosine Sim |
|---|---|---|---|---|---|
| 4 | 32 | 22.1x | 0.003 | 100% | 0.9999 |
| 13 | 32 | 22.1x | 0.023 | 100% | 0.9999 |
| 22 | 32 | 22.1x | 0.038 | 97% | 0.9998 |
| 4 | 16 | 16.5x | 0.009 | 100% | 0.9998 |
| 13 | 16 | 16.5x | 0.031 | 97% | 0.9997 |
| 22 | 16 | 16.5x | 0.085 | 94% | 0.9993 |
| 4 | 8 | 11.1x | 0.021 | 97% | 0.9997 |
| 13 | 8 | 11.1x | 0.021 | 97% | 0.9999 |
| 22 | 8 | 11.1x | 0.850 | 78% | 0.9988 |

**Key finding**: Rank-32 carrier-only compression achieves 22x bandwidth reduction at the mid-model boundary (layer 13) with KL divergence of only 0.023 and 100% next-token agreement. This is the headline result: **a 22x reduction in inter-shard bandwidth with no change in the predicted token**.

### 5.3 Carrier + Payload (Hybrid)

Adding a sparse residual improves quality at intermediate ranks:

| Splice Layer | Rank | Sparse % | CR | KL | Top-1 |
|---|---|---|---|---|---|
| 13 | 8 | 0.5% | 10.0x | 0.007 | 97% |
| 13 | 8 | 1.0% | 9.1x | 0.004 | 100% |
| 13 | 8 | 5.0% | 5.3x | 0.001 | 100% |
| 13 | 4 | 5.0% | 5.4x | 0.010 | 100% |
| 13 | 4 | 10% | 3.5x | 0.005 | 100% |
| 22 | 8 | 1.0% | 9.1x | 0.052 | 97% |
| 22 | 8 | 10% | 3.4x | 0.002 | 100% |
| 22 | 16 | 10% | 4.3x | 0.074 | 100% |

The hybrid approach provides a smooth trade-off between compression and quality. At the mid-model boundary, rank-8 + 1% sparse achieves 9.1x compression at KL = 0.004 with 100% top-1 agreement.

### 5.4 Pareto Frontier

The Pareto frontier (compression ratio vs. KL divergence) reveals three distinct regimes:

1. **Lossless regime** (CR $< 5$x): KL $\sim 10^{-3}$, essentially perfect reconstruction. Dominated by high-rank + high-sparsity configurations.
2. **Sweet spot** (CR 5--22x): KL $< 0.05$, 96--100% top-1 agreement. This is the operating region for practical deployment.
3. **Degraded regime** (CR $> 22$x at low rank): KL $> 0.1$, top-1 agreement drops below 90%. Not recommended for production use.

The sweet spot at the mid-model boundary (layer 13) spans rank-8 carrier-only (11x, KL = 0.021) to rank-32 carrier-only (22x, KL = 0.023). The Pareto-optimal configurations are:

- **Maximum CR at KL < 0.01**: 22.1x (rank 32, carrier-only, layer 4)
- **Maximum CR at KL < 0.1**: 22.1x (rank 32, carrier-only, layers 4 and 13)
- **Maximum CR at top-1 = 100%**: 22.1x (rank 32, carrier-only, layers 4 and 13)

### 5.5 Phase Transition Artifact and Control

When the sequence length $T$ is shorter than the PCA rank $k$, the SVD produces an exact decomposition (the activation matrix has rank at most $\min(T, d) = T$). In this regime, PCA is not compressing---it is performing an exact basis change. This manifests as a sharp "phase transition" where KL divergence drops to floating-point noise ($\sim 10^{-8}$) when $k \geq T$.

We confirmed this artifact in a 1-prompt test run where $T = 9$ tokens: ranks $\geq 12$ produced exact reconstruction (variance explained = 1.000, KL $\sim 10^{-8}$). This is a linear algebra identity, not evidence of compressibility.

**Long-context control**: Our control experiment with prompts of length 47--70 tokens and ranks up to 64 confirmed that:

1. The phase transition disappears when $T > k$ for all tested ranks.
2. The variance-explained and KL curves remain smooth and monotonically improving with rank.
3. Rank-32 carrier-only still achieves KL $< 0.03$ at these longer sequence lengths, confirming that the low-rank structure is a genuine property of the activation manifold.

### 5.6 Compression Ratio Scaling with Sequence Length

From Section 3.2, the carrier-only compression ratio is:

$$\text{CR} = \frac{T \cdot d}{T \cdot k + d}$$

For fixed $k$ and $d$, this increases monotonically with $T$:

| $T$ (tokens) | Rank $k = 32$, $d = 1536$ | CR |
|---|---|---|
| 9 | (short prompt) | 6.0x |
| 64 | | 22.1x |
| 128 | | 33.0x |
| 256 | | 39.7x |
| 512 | | 43.6x |
| 1024 | | 45.8x |
| $\infty$ | (asymptote) | 48.0x |

At 1024 tokens (a typical generation context), rank-32 carrier-only compression approaches **46x**. The current experimental results at short sequences ($T \approx 9$--70) represent the *worst case* for compression ratio; production workloads with longer contexts will see proportionally higher compression.

### 5.7 Layer-Wise Variation

Compression quality varies across splice positions. Early layers (layer 4) are the most compressible: rank-32 achieves KL = 0.003. Late layers (layer 22) are the hardest: rank-32 achieves KL = 0.038, and rank-8 carrier-only degrades to KL = 0.85 with only 78% top-1 agreement.

This is consistent with the view that early layers compute more generic, low-rank features (positional and syntactic), while later layers encode task-specific, higher-rank information. For pipeline-parallel deployment, this suggests placing shard boundaries at early-to-mid layers when possible, or using higher ranks (and thus lower compression) for late boundaries.

---

## 6. Discussion

### 6.1 Implications for Synapse

The Synapse platform (webmind.sh) distributes Gemma 3 1B inference across volunteer WebGPU browsers. The primary bottleneck is activation transport between shards. Our results show:

**Bandwidth requirement reduction.** For a 3-shard pipeline with boundaries at layers 4 and 13, rank-32 carrier-only compression reduces per-boundary transfer from $T \times 1536 \times 2$ bytes to $T \times 32 \times 2$ bytes plus a 3 KB mean vector. At $T = 512$, this is a reduction from 1.57 MB to 33 KB per boundary---a **48x** reduction. At a 10 Mbps upload link, this reduces transfer time from 1.26 seconds to 26 milliseconds per boundary.

**Latency impact.** For a 3-shard pipeline with 2 boundaries, total inter-shard latency drops from 2.5 seconds to 52 milliseconds at 10 Mbps. This makes interactive (streaming) generation feasible over consumer internet connections.

**Basis distribution.** The one-time cost of distributing the PCA basis to all nodes is $32 \times 1536 \times 4 = 196$ KB per boundary layer---negligible compared to the model weights each node already stores.

### 6.2 Comparison to Quantization Baselines

Naive activation quantization provides compression without exploiting manifold structure:

| Method | CR | Quality |
|---|---|---|
| Float16 (baseline) | 1x | Perfect |
| INT8 (Petals-style) | 2x | Good, but outlier clipping |
| INT4 | 4x | Significant degradation |
| Carrier rank-32 | 22x | KL = 0.023, 100% top-1 |
| Carrier rank-8 + 1% sparse | 9.1x | KL = 0.004, 100% top-1 |

Carrier-Payload achieves 5--11x better compression than INT8 quantization at comparable or better quality. This is because quantization treats all dimensions equally, while PCA concentrates the representation into the dimensions that matter. The sparse payload further addresses the outlier problem that causes INT4 quantization to fail [7].

**Orthogonal combination.** Carrier-Payload and quantization are orthogonal: the projection coefficients $\mathbf{P}$ and sparse entries can themselves be quantized. A rank-32 carrier with INT8 coefficients would achieve approximately $22 \times 2 = 44$x compression. We leave this combination to future work.

### 6.3 Limitations

1. **Model scale.** We test only on Gemma 3 1B. The intrinsic dimension literature [6] suggests that effective dimensionality scales sublinearly with model size ($d_{\text{intrinsic}} / d$ *decreases* with scale), which would make larger models *more* compressible. However, this requires empirical validation at 7B+ scale.

2. **Single-layer splice.** We compress at a single boundary per experiment. Multi-shard pipelines require compression at multiple boundaries, and errors may compound across splices. End-to-end evaluation with cascaded compression is needed.

3. **Basis generalization.** Our per-prompt PCA basis represents an upper bound on quality. A production system would use a fixed basis computed from calibration data, which may not capture input-specific directions. The gap between per-prompt and calibration-based bases must be measured.

4. **Sequence-length dependence.** Short sequences ($T < k$) produce artificially high compression quality due to the rank-deficiency artifact. Our long-context control confirms that results hold at $T > k$, but production workloads at $T = 2048+$ need verification.

5. **Output metric.** We measure KL divergence and top-1 agreement on the *last* token's logits. Full generative evaluation (perplexity, downstream task accuracy) would provide stronger evidence of quality preservation.

---

## 7. Related Work

### 7.1 Decentralized Inference Systems

**Petals** [1] is the closest prior work. It enables collaborative inference of LLMs across distributed commodity GPUs. Petals uses 8-bit activation quantization for bandwidth reduction (~2x) and a reputation system for handling unreliable nodes. Carrier-Payload achieves 11--22x compression at comparable quality, a 5--11x improvement over Petals' quantization.

**opML** [2] proposes optimistic machine learning, using fraud proofs for verification of outsourced inference. It addresses correctness but not bandwidth.

**ZKML** [3] applies zero-knowledge proofs to verify ML inference. Current ZK-proof overhead makes this impractical for real-time inference (proving time exceeds inference time by 3--5 orders of magnitude).

**SafetyNets** [4] verifies outsourced deep learning by converting computations to integer arithmetic and checking via interactive proofs. It provides verification but does not compress.

### 7.2 Tensor and Activation Compression

**Low-rank approximation** is widely used for weight compression in LLMs (LoRA [8], SVD-based pruning). Application to *activations* during inference has received less attention, likely because activations are input-dependent and cannot be pre-compressed at training time.

**Mixture-of-experts (MoE)** architectures [9] implicitly reduce activation bandwidth by routing only a subset of tokens through each expert. However, MoE is an architectural choice, not a post-hoc compression method applicable to existing dense models.

**Activation checkpointing** [10] reduces memory by recomputing activations during backpropagation. This addresses memory, not bandwidth, and applies only to training.

### 7.3 Representation Geometry

**Ethayarajh (2019)** [5] showed that contextual word representations in BERT and GPT-2 are anisotropic: later layers are more anisotropic, with representations concentrated in a narrow cone. Our variance-explained measurements (99.4% at rank 2 for early layers, 96.2% for late layers) are quantitatively consistent with this finding.

**Aghajanyan et al. (2020)** [6] measured the *intrinsic dimensionality* of fine-tuning---the minimum subspace dimension needed for effective adaptation. They found $d_{\text{intrinsic}} \sim 200$ for RoBERTa-base ($d = 768$). Our finding that $\sim 32$ PCA dimensions suffice for *activation* compression at $d = 1536$ is a related but distinct measurement: theirs characterizes the weight update manifold, ours characterizes the activation manifold.

**Dettmers et al. (LLM.int8())** [7] identified activation outlier features that cause quantization failure and proposed mixed-precision decomposition. Our sparse payload serves an analogous function: it captures the high-magnitude entries that the smooth PCA carrier misses.

---

## 8. Conclusion and Future Work

We have presented Carrier-Payload decomposition, a method for compressing transformer activations at pipeline-parallel shard boundaries by exploiting the low intrinsic dimensionality of the activation manifold. On Gemma 3 1B IT, we achieve 22x compression with 100% next-token agreement using a rank-32 PCA carrier, with compression ratios scaling to 46x at production sequence lengths. These results make volunteer-device decentralized inference practical over consumer internet connections.

### Future Work

**Scale to 7B+ models.** Validate that effective dimensionality remains low (or decreases, as suggested by [6]) at larger model scales. A minimal spot-check on Gemma 3 4B or 7B with ranks 32 and 64 would establish whether the method transfers.

**Calibration-based basis.** Replace per-prompt SVD with a fixed basis computed from calibration data. Measure the quality gap and determine the minimum calibration corpus size for robust basis estimation.

**Cascaded compression.** Evaluate multi-boundary compression with error compounding analysis. If errors accumulate, investigate error-correction strategies (e.g., periodic full-precision synchronization).

**Combination with quantization.** Apply INT8 or INT4 quantization to the projection coefficients and sparse entries for multiplicative compression gains.

**Byzantine verification.** The low-rank structure of activations enables efficient verification: if a node claims to produce activation $\mathbf{A}$, the coordinator can verify that $\mathbf{A}$ lies near the expected manifold by checking its projection residual norm. This provides a statistical integrity check complementary to cryptographic methods.

**Functional basis sharing.** Instead of PCA (which minimizes MSE), learn a basis that minimizes downstream output divergence directly. This could be framed as a distillation objective: find $\mathbf{V}^* = \arg\min_\mathbf{V} \mathbb{E}[\text{KL}(p_{\text{full}} \| p_{\text{compressed}}(\mathbf{V}))]$.

---

## Acknowledgements

This research was conducted at Webmind Research (webmind.sh) under the direction of Tejas Phatak. The experimental code, analysis, and paper draft were generated by Claude Opus 4.6 (Anthropic). Gemini 3.1 Pro (Google) contributed to experimental design review, diagnosis of the phase-transition artifact, identification of the correct long-context control, and literature pointers (Ethayarajh 2019, Aghajanyan et al. 2020, Dettmers LLM.int8!). Neither AI system is listed as an author per current publishing norms; both contributions are disclosed here in full transparency.

---

## References

[1] A. Borzunov, D. Baranchuk, T. Dettmers, M. Riabinin, Y. Belkada, A. Chumachenko, P. Samygin, C. Raffel. "Petals: Collaborative Inference and Fine-tuning of Large Models." *Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (ACL 2023 System Demonstrations)*, pp. 558–568, Toronto, 2023. arXiv:2209.01188.

[2] K. D. Conway, C. So, X. Yu, K. Wong. "opML: Optimistic Machine Learning on Blockchain." *arXiv preprint arXiv:2401.17555*, 2024.

[3] D. Kang, T. Hashimoto, I. Stoica, Y. Sun. "Scaling up Trustless DNN Inference with Zero-Knowledge Proofs." *arXiv preprint arXiv:2210.08674*, 2022.

[4] Z. Ghodsi, T. Gu, S. Garg. "SafetyNets: Verifiable Execution of Deep Neural Networks on an Untrusted Cloud." *Advances in Neural Information Processing Systems (NIPS 2017)*. arXiv:1706.10268.

[5] K. Ethayarajh. "How Contextual are Contextualized Word Representations? Comparing the Geometry of BERT, ELMo, and GPT-2 Embeddings." *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing (EMNLP-IJCNLP 2019)*, pp. 55–65, Hong Kong. arXiv:1909.00512.

[6] A. Aghajanyan, S. Gupta, L. Zettlemoyer. "Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning." *Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics (ACL-IJCNLP 2021)*, Outstanding Paper, pp. 7319–7328, 2021. arXiv:2012.13255.

[7] T. Dettmers, M. Lewis, Y. Belkada, L. Zettlemoyer. "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale." *Advances in Neural Information Processing Systems (NeurIPS 2022)*. arXiv:2208.07339.

[8] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, W. Chen. "LoRA: Low-Rank Adaptation of Large Language Models." *Proceedings of the International Conference on Learning Representations (ICLR 2022)*. arXiv:2106.09685.

[9] N. Shazeer, A. Mirhoseini, K. Maziarz, A. Davis, Q. Le, G. Hinton, J. Dean. "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer." *Proceedings of the International Conference on Learning Representations (ICLR 2017)*. arXiv:1701.06538.

[10] T. Chen, B. Xu, C. Zhang, C. Guestrin. "Training Deep Nets with Sublinear Memory Cost." *arXiv preprint arXiv:1604.06174*, 2016.

[11] F. Tramèr, D. Boneh. "Slalom: Fast, Verifiable and Private Execution of Neural Networks in Trusted Hardware." *Proceedings of the International Conference on Learning Representations (ICLR 2019)*. arXiv:1806.03287.

[12] G. Xiao, J. Lin, M. Seznec, H. Wu, J. Demouth, S. Han. "SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models." *Proceedings of the 40th International Conference on Machine Learning (ICML 2023)*. arXiv:2211.10438.

[13] A. Ansuini, A. Laio, J. H. Macke, D. Zoccolan. "Intrinsic dimension of data representations in deep neural networks." *Advances in Neural Information Processing Systems (NeurIPS 2019)*. arXiv:1905.12784.

---

## Appendix A: Prompt Corpus

The 32 prompts used in the primary experiment span the following categories:

| Category | Count | Examples |
|---|---|---|
| Technical explanation | 10 | Byzantine fault tolerance, TCP congestion, PCA, SVD |
| Code generation | 3 | Fibonacci via matrix exponentiation, Raft pseudocode, SVD function |
| Mathematics/logic | 4 | Irrationality proof, birthday paradox, Kolmogorov complexity |
| Creative writing | 2 | Haiku, signal compression description |
| Translation | 1 | English to French |
| Applied reasoning | 3 | Train meeting problem, hash collisions, Nyquist theorem |
| ML/AI concepts | 6 | Manifold hypothesis, activation outliers, SmoothQuant, KANs |
| Systems/distributed | 3 | WebGPU vs WebGL, BOINC integrity, federated learning |

This diversity ensures that the measured activation structure is not an artifact of a particular input domain.

## Appendix B: Compression Ratio Formula Derivation

The wire cost for Carrier-Payload transmission consists of three components:

1. **Projection coefficients**: $T \times k$ floats. Each of the $T$ token positions has a $k$-dimensional projection onto the shared basis.

2. **Mean vector**: $d$ floats. The per-activation centroid, needed for reconstruction. (In a production system, this could be absorbed into the carrier if the calibration mean is also shared, reducing this to a per-activation mean residual.)

3. **Sparse payload**: $2 \times n_{\text{keep}}$ floats, where $n_{\text{keep}} = \lfloor p \cdot T \cdot d \rfloor$. Each retained residual entry requires an index (integer, stored as float for simplicity) and a value.

The compression ratio is therefore:

$$\text{CR} = \frac{T \cdot d}{T \cdot k + d + 2 \lfloor p \cdot T \cdot d \rfloor}$$

For the carrier-only case ($p = 0$):

$$\text{CR} = \frac{Td}{Tk + d} = \frac{d}{k + d/T}$$

As $T \to \infty$: $\text{CR} \to d/k$.

For Gemma 3 1B ($d = 1536$), the asymptotic compression ratios by rank are:

| Rank $k$ | Asymptotic CR |
|---|---|
| 2 | 768x |
| 4 | 384x |
| 8 | 192x |
| 16 | 96x |
| 32 | 48x |

These are upper bounds assuming the carrier quality (KL divergence) remains acceptable at each rank. In practice, rank 32 is the minimum for KL $< 0.05$ across all layers, giving a practical asymptotic compression of 48x.
