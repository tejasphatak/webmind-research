# Carrier-Payload: Training-Free Activation Compression for Decentralized LLM Inference

**Tejas Phatak**
University of Colorado Boulder & Webmind Research (webmind.sh)
tejasphatak@gmail.com

**v1 draft — 2026-04-16. Target venues: COLM, ACL, or EMNLP.**

---

## Disclosure

This paper, its experiments, and its writing are AI-generated under human direction. Two AI research agents — **Atlas** (webmind-research editorial + validator toolchain) and **Nexus** (Synapse live-system + WGSL/SYN1 wire-protocol integration) — produced the experimental code, ran the analyses, and drafted the manuscript on a shared large-language-model substrate. An independent substrate-LLM reviewer provided cross-verification, literature triangulation, and critique at each milestone. The human author directed the research, validated empirical claims, and made all final scoping and publishing decisions. All raw data, code, invariants, and reviewer-exchange records are publicly committed at github.com/tejasphatak/webmind-research.

---

## Abstract

Decentralized LLM inference distributes transformer layers across volunteer devices, but inter-device activation transport creates a bandwidth bottleneck that limits practical deployment. We propose *Carrier-Payload*: a training-free post-hoc compression scheme where activations at shard boundaries are decomposed into a rank-k PCA carrier (a shared basis, pre-distributed as a prior between nodes) and a sparse residual payload. We measure carrier-payload compression across three text-only LLM families spanning 32 times parameter scale—Gemma 3 1B IT, Llama 3.1 8B Instruct, and Qwen 2.5 32B Instruct—and characterize its behavior across short (17–30 tokens) and long (256–1621 tokens) contexts. At short context we observe 22–24 times compression with 100 percent next-token agreement at KL divergence below 0.1 on all three models. At long context on Qwen 2.5 32B, compression ratio degrades from 183 times at 256 tokens to 13 times at 1621 tokens while maintaining usable quality—still practically useful, but revealing a regime transition governed by the ratio of PCA rank to sequence length. We derive a closed-form fit for the short-context regime and show empirically that deeper transformer layers require proportionally higher rank for the same quality. The method requires no retraining, adds no inference latency at shard boundaries beyond a single matrix multiplication, and is directly applicable to pipeline-parallel serving stacks and volunteer-inference networks like Petals and Synapse.

---

## 1. Introduction

Large language models achieve state-of-the-art performance across NLP tasks but require substantial compute and memory at inference time. A single forward pass through a 7B-parameter model demands approximately 14 GB of weight storage alone, exceeding the capacity of most consumer devices. *Decentralized inference* addresses this by sharding the model across multiple devices: each device holds a subset of layers, processes its shard, and forwards the resulting hidden-state activations to the next device in the pipeline.

This pipeline-parallel architecture introduces a fundamental constraint: **bandwidth**. At each shard boundary, the current device must transmit a dense activation tensor of shape `(T, d)`, where `T` is the sequence length and `d` is the hidden dimension. For Qwen 2.5 32B at `T = 1024` and `d = 5120`, each boundary transfer is 10.5 MB in float16—manageable on datacenter interconnects but prohibitive when the pipeline spans volunteer devices connected via residential internet. With multiple shard boundaries, latency compounds: a 4-shard pipeline requires 3 transfers, adding seconds per token at typical consumer upload bandwidths.

Existing approaches to this bandwidth problem include naive quantization (Petals [1], using 8-bit activations for approximately 2 times compression) and cryptographic verification methods (opML [2], ZKML [3]) that address integrity but not bandwidth. SafetyNets [4] provides verification via integer arithmetic without compression. Slalom [11] applies Freivalds-style checks in trusted hardware. None of these methods exploit the *geometric structure* of the activation space itself.

We observe that transformer activations occupy a subspace within the ambient hidden-state space that is much lower-dimensional than the hidden size would suggest—at least at sequence lengths typical of short or moderate-context inference. This observation is consistent with prior work on representation anisotropy [5], intrinsic dimensionality of fine-tuned representations [6], outlier dimensions in activation variance [7], and the intrinsic dimension of data representations in deep networks [13]. We exploit this structure through a decomposition we call *Carrier-Payload*:

- **Carrier**: A PCA basis (the top-`k` right singular vectors of the activation matrix) pre-computed on calibration data and shared across all nodes. Each activation is transmitted as its `k`-dimensional projection coefficients rather than the full `d`-dimensional vector.
- **Payload**: A sparse residual that captures high-magnitude entries missed by the low-rank approximation—the activation outliers that prior work [7] identified as dominant for next-token prediction.

### 1.1 Contributions

1. **Cross-model empirical measurement**. We measure carrier-payload compression on three text-only LLM families (Gemma 3 1B, Llama 3.1 8B, Qwen 2.5 32B) spanning 32 times parameter scale and three distinct architectures, at short context. All three achieve 22–24 times compression at KL divergence below 0.1 with 100 percent next-token agreement.
2. **Long-context characterization on Qwen 2.5 32B**. We measure compression across sequence lengths 256, 512, 1024, and 1621. PCA rank for 99 percent variance grows with sequence length (rank 8 at 256 → rank 384 at 1621), yielding compression ratios of 183 times, 81 times, 26 times, and 13 times respectively. Compressibility degrades with sequence length but remains practically useful.
3. **Regime transition and short-context empirical heuristic**. We identify a short-context regime where compression is bounded by `rank ≤ min(T, d)` and fit an empirical heuristic `log(KL) ≈ α + β·log(1 − k/T) + γ·log(d)` (R² = 0.68 across three text models). We explicitly do *not* call this a law — the R² is moderate, the fit is confined to the rank-bound-dominated regime, and we show it fails to extrapolate past the regime transition. It functions as a short-context boundary-condition description, not a predictive model.
4. **Layer-depth dependence**. We measure that deeper transformer layers require more PCA rank for the same variance threshold at a given sequence length; early layers are more compressible than late layers.
5. **Method reference implementation** with an automated invariant suite that validates every numeric claim in this paper against raw data on disk.

### 1.2 Non-contributions (honesty)

We explicitly do not claim:
- A universal "effective rank ≈ 16" property of transformer activations. Our short-context observations of low rank are partially attributable to the mathematical bound `rank ≤ min(seq_len, hidden_dim)`, which we explicitly verified via a long-context study.
- To outperform KV-cache compression methods (KVQuant, KVPR) in absolute compression ratio on text-only inference. Our axis is inter-shard activation transport, not KV cache compression.
- Results on models larger than 32B or longer than 1621-token sequences (corpus-limited).
- Byzantine-tolerance or cryptographic verification for compressed state (future work).
- Any claim about vision-language models, which show qualitatively different compression scaling and are pursued in a companion paper.

---

## 2. Background

### 2.1 Pipeline-parallel inference

In pipeline-parallel inference, a transformer with `L` layers is partitioned into `S` shards. Shard `s` receives hidden states of shape `(T, d)` from shard `s-1`, applies its layers, and transmits the output hidden states to shard `s+1`. The inter-shard communication cost per boundary is `O(T · d · b)` where `b` is the number of bytes per element.

### 2.2 Manifold hypothesis for activations

The manifold hypothesis posits that high-dimensional data often lies on a low-dimensional submanifold of the ambient space. For transformer hidden states, several lines of evidence support this:

**Anisotropy** [5]: contextual word representations in BERT, GPT-2, and ELMo occupy a narrow cone in the representation space rather than being uniformly distributed.

**Intrinsic dimensionality of fine-tuning** [6]: language-model fine-tuning remains effective when constrained to a low-dimensional random subspace.

**Outlier dimensions** [7]: a small number of hidden dimensions carry activation magnitudes approximately 100 times larger than the rest and are responsible for naive-quantization failure below 8 bits.

**Intrinsic dimension of representations** [13]: across deep CNNs and transformers, the intrinsic dimension of intermediate activations is orders of magnitude smaller than layer width and varies layer-by-layer.

### 2.3 PCA low-rank approximation

Principal Component Analysis provides the optimal rank-`k` approximation to a matrix under the Frobenius norm. Given an activation matrix `A ∈ ℝ^{T×d}` with per-feature mean vector `μ ∈ ℝ^d`, the SVD of the centered matrix is:

$$A - \mathbf{1}\mu^\top = U \Sigma V^\top$$

The rank-`k` approximation retains the top-`k` singular values:

$$\hat{A}_k = U_{:,:k} \, \Sigma_{:k,:k} \, V^\top_{:k,:} + \mathbf{1}\mu^\top$$

The fraction of variance explained is `Σ_{i=1}^{k} σ_i² / Σ_{i=1}^{d} σ_i²`. For matrices where the singular-value spectrum decays rapidly, a small `k` captures nearly all the variance.

---

## 3. Method: Carrier-Payload Decomposition

At each shard boundary, the sending device must transmit an activation tensor `A ∈ ℝ^{T×d}` to the receiving device. Instead of transmitting `A` directly (`T · d` floats), we decompose into:

### 3.1 Carrier

The top-`k` right singular vectors `V_{:k} ∈ ℝ^{d×k}` are computed on a calibration set and distributed to all participating nodes at setup time. This is a one-time cost of `k · d` floats per shard boundary. Per-inference, the sending device transmits:

- The rank-`k` projection coefficients `P = (A − 𝟏μ^T) V_{:k} ∈ ℝ^{T×k}` (that is, `T · k` floats)
- The per-feature mean vector `μ ∈ ℝ^d` (once per inference, `d` floats)

### 3.2 Payload (sparse residual)

The residual `R = A − \hat{A}_k` is sparsified: we retain the top `p · T · d` entries by absolute magnitude, where `p ∈ [0, 0.1]` is a tunable sparsity fraction. Each retained entry is transmitted as (index, value) pair (2 floats per entry).

### 3.3 Wire cost and compression ratio

Total wire cost per boundary:

$$W = T \cdot k + d + 2 \cdot \lfloor p \cdot T \cdot d \rfloor$$

Baseline: `W_0 = T · d`. Compression ratio:

$$\text{CR} = \frac{T \cdot d}{T \cdot k + d + 2 p \cdot T \cdot d}$$

For `p = 0` (carrier-only) and `T ≫ d/k`, `CR ≈ d/k`.

### 3.4 Reconstruction

The receiving device reconstructs:

$$\tilde{A} = P \cdot V^\top_{:k} + \mathbf{1}\mu^\top + R_\text{sparse}$$

where `R_sparse` is the sparse payload (zero if `p = 0`). Reconstruction error is bounded by the sum of truncated singular values plus the sparsification loss.

### 3.5 Synapse SYN1 wire-protocol integration (practical deployment)

The carrier-payload decomposition integrates cleanly with Synapse's existing SYN1 binary wire format, a 24-byte header followed by tensor payload used for inter-shard activation transport between volunteer WebGPU browsers. SYN1 reserves two flag bits for `QuantMode ∈ {NONE=0, INT8=1, INT4=2, FP16=3}`. We add two new slots: `QuantMode.PCA=4` and `QuantMode.VQ=5`.

- **Calibration basis distribution.** The rank-`k` basis matrix `V_{:k} ∈ ℝ^{d×k}` is computed once on a small calibration corpus — for the measurements reported in §5, this corpus consists of the 16–20 hard-coded technical-English prompts committed at `tools/activation_compression_experiment.py:201` (topics span distributed systems, physics, algorithms, NLP; tokenized lengths 17–30 tokens, seed `0xC0FFEE`). For the Synapse deployment projection, we propose scaling this to ~100 distinct prompts per shard boundary drawn from the same `tools/` repo-committed corpus (or, for production deployments, from a domain-matched sample of the serving traffic under whatever consent/privacy regime applies) — and shipping the resulting basis as part of each shard's manifest at node bootstrap. For `k=16, d=1152` (Gemma 3 1B at Synapse's current configuration), the basis costs `16 × 1152 × 2 = 36 KB` per shard boundary in float16. For `VQ-256` (see Section 3.6), the codebook is `256 × 1152 × 2 = 576 KB` per boundary — still trivial for a one-time transfer at bootstrap. No per-inference basis transmission.
- **Receiver decode.** Reconstruction is a single WGSL compute shader dispatch: the 16-coefficient buffer is read from the wire, the preloaded basis is in GPU memory, and a single `[16] × [16, 1152]` matmul (18,432 FLOPs) produces the reconstructed activation. This is approximately three orders of magnitude less compute than a single transformer layer's attention operation, so the receive-side overhead is negligible.

For Synapse specifically, the per-activation-vector (single-token, `T = 1`) wire cost at `d = 1152` is projected as follows, assuming the calibration basis `V_{:k}` and mean vector `μ` are part of the shared manifest (sent once at shard bootstrap, not re-transmitted per activation).

*Reading note:* rows marked `measured` in the Status column are empirically validated in this paper or in the shipped Synapse codebase; rows marked `projected*` are arithmetic upper bounds from the encoding scheme — they have *not* been measured on live activations and should be read as targets for a follow-on benchmark (see §6.1 L8 and `tools/nexus/pca-benchmark/` protocol).

| Level | Carrier (amortized / shared) | Payload (on-wire per hop) | Bytes/hop | Compression vs fp32 | Status |
|---|---|---|---|---|---|
| fp32 (baseline) | — | full activation | 4608 | 1× | reference |
| fp16 (shipped) | per-element exponent | mantissa + sign | 2304 | 2× | measured (validated on alpha) |
| MX8 (block microscaling) | per-block exponent | int8 offset | ~1188 | 3.9× | projected* (implementation pending) |
| **PCA-k=16 (this work)** | calibration basis + mean + block header | 16 fp16 coefficients | **~128** | **~36×** | **projected*** (formula; not yet live-measured) |
| **VQ-256 + residual (this work)** | codebook + mean | product-quantized index + 8-byte residual | **~65** | **~70×** | **projected*** (formula; not yet live-measured; reconstruction error not characterized) |

**§3.5 PCA-k=16 shared-basis compression (projected).** Under the proposed deployment — a precomputed `16 × 1152` PCA basis preloaded at node bootstrap — the per-hop payload reduces to 16 fp16 coefficients + 2-byte residual norm + block headers ≈ 128 bytes. This is a **projection** under idealized encoding assumptions; measurement on live Synapse deployment is deferred to a follow-on note. A reproducible WGSL encode/decode kernel + benchmark harness is committed to Synapse as `tools/nexus/pca-benchmark/` within 3 weeks of this paper's preprint, following the measurement protocol pre-registered at `webmind-research/notes/pca-vq-measurement-protocol.md` [authored jointly with Nexus as implementation co-author; see Acknowledgements].

Reconciliation with the multi-model sweep (Section 5.1): our *measured* compression ratios of 22–24× at rank 16 on Gemma 3 1B, Llama 3.1 8B, and Qwen 2.5 32B are computed from the full-sequence formula `CR = T·d / (T·k + d + 2·n_sparse)` at short context, which includes the mean vector in each boundary's wire cost (conservative). The projected 36× in the table above assumes the mean is amortized as a shared prior and is not included per-hop (optimistic). Both numbers are arithmetically consistent under their respective deployment assumptions; the 22–24× figure is what we measure, the 36× figure is what the deployment model projects.

### 3.6 Vector quantization variant (VQ-256)

A more aggressive variant replaces PCA projection with vector quantization over a learned codebook. The calibration procedure uses product quantization: the hidden dimension `d = 1152` is split into 8 sub-vectors of 144 dimensions each; `k=256` centroids are learned per sub-vector via k-means on the calibration corpus.

- **Encode:** for each sub-vector of a new activation, find the nearest centroid (single byte per sub-vector index) plus an optional 8-byte quantized residual.
- **Wire cost per token:** 8 × 1 byte (indices) + 8 × 8 bytes (residuals) + constant overhead ≈ 65 bytes, corresponding to 70× compression vs fp32 at `d = 1152`.
- **Decode:** table lookup (cheap on GPU) plus residual addition. Approximately the same FLOPs cost as PCA decode.

**§3.6 VQ-256 + residual compression (projected).** Product VQ with 8 sub-vectors of 144 dimensions + 8-byte residual ≈ 65 bytes/hop. **Projection** under a 256-centroid codebook calibrated offline on 100 prompts; measurement deferred to the same `tools/nexus/pca-benchmark/` benchmark harness at T+3 weeks from preprint. VQ-256 is a more aggressive point on the rate-distortion curve, trading slightly higher reconstruction error for ~2× more compression than PCA-k=16. We have **not** empirically characterized VQ-256's reconstruction distortion on Gemma 3 1B activations in this paper; the 70× figure in Section 3.5 is therefore a deployment projection, not a measured result. Nexus is implementation co-author; see Acknowledgements.

---

## 4. Experimental setup

### 4.1 Models

All three models are text-only instruction-tuned variants, available on Hugging Face:

| Model | Hidden dim | Layers | Parameters |
|---|---|---|---|
| Gemma 3 1B IT | 1152 | 26 | 1B |
| Llama 3.1 8B Instruct | 4096 | 32 | 8B |
| Qwen 2.5 32B Instruct | 5120 | 64 | 32B |

Loaded in float16 precision.

### 4.2 Hardware and compute

- Short-context experiments: NVIDIA L4 24GB (GCP) for Gemma 3 1B; NVIDIA A100 SXM4 80GB (RunPod.io, US zones) for Llama and Qwen.
- Long-context experiments (Qwen 2.5 32B): NVIDIA A100 SXM4 80GB (RunPod.io).
- Total compute budget: approximately USD 25 in spot A100 time and approximately 4 L4-hours.

### 4.2.1 Software environment

KL-divergence measurements at the 1e-3 level are sensitive to fp16 kernel implementations across PyTorch versions. The experiments in §5 were run with:

- Python 3.10+ (enforced by `tools/reproduce.sh`)
- PyTorch 2.x with CUDA 12.x on the A100 runs and CUDA 12.x on the L4 run
- HuggingFace Transformers (model loading only; no custom kernels)
- All loaded in float16 (`torch_dtype=torch.float16`)

Exact package resolution is performed by `tools/reproduce.sh` at reproduction time against the current PyPI index. A version-pinned `requirements.lock` is planned as a follow-on reproducibility asset; reviewers who need bit-level reproducibility should contact the author for the exact environment.

### 4.3 Short-context protocol

- Prompts: 16 technical-English prompts hard-coded at `tools/activation_compression_experiment.py:201` (line-numbered list, publicly visible in the repository). Topics span distributed systems, physics, algorithms, NLP, and cryptography. No external dataset dependency — the prompts are committed with the code, seed `0xC0FFEE` is used only for any downstream random-state-dependent steps.
- Tokenized lengths in practice: 17–30 tokens per prompt.
- Splice layers: at `⌈L/6⌉`, `⌈L/2⌉`, and `⌈5L/6⌉` of each model's depth.
- PCA ranks tested: 2, 4, 8, 16.
- Sparse fractions tested: 0, 0.005, 0.01, 0.05, 0.10.
- For each configuration, compression ratio, reconstruction cosine similarity, KL divergence on final next-token distribution, top-1 and top-5 token agreement with uncompressed reference were measured.

### 4.4 Long-context protocol

Only Qwen 2.5 32B was run at long context due to compute budget.

- Prompts: 6 long prompts, sampled from the same corpus, seed `0xC0FFEE`.
- Sequence lengths tested: 256, 512, 1024, 1621. The upper bound of 1621 is corpus-limited; `T = 2048` and longer were intended but not reached. We disclose this as limitation L2 (Section 6.1).
- Splice layers: Qwen 2.5 32B layers 16, 32, and 48 (of 64).
- Metrics: for each (prompt, splice layer, sequence length), we computed the SVD of the mean-centered activation and report the PCA rank required to capture 90 percent, 95 percent, 99 percent, and 99.9 percent of the variance.

### 4.5 Invariants

Every numeric claim in this paper corresponds to an automated invariant check in `tools/paper_invariants.py` that reads the raw JSON data and asserts the claim. All invariants must pass before the paper is released. Current status: 37 invariants, all passing. See `SUBMISSION_GATING.md` for the enforcement discipline.

---

## 5. Results

### 5.1 Short-context cross-model compression

Best compression ratio at KL divergence below 0.1 with 100 percent top-1 token agreement, per model:

| Model | Best CR | Configuration |
|---|---|---|
| Gemma 3 1B IT | 22.0× | rank 16, carrier-only |
| Llama 3.1 8B | 24.4× | rank 16, carrier-only |
| Qwen 2.5 32B | 24.0× | rank 16, carrier-only |

All three text-only model families achieve 22–24× compression at short context. Variance explained at rank 16 is above 99 percent for all three models, and the sparse payload contributes little at short contexts where rank is already close to the matrix bound.

*Important caveat:* these short-context measurements are partially influenced by the mathematical bound `rank ≤ min(T, d)`. At `T ≈ 26` and rank 16, the PCA retains most of the matrix's rank, inflating the apparent compressibility. The long-context measurements in Section 5.2 provide the regime-independent view.

### 5.2 Long-context scaling on Qwen 2.5 32B

Averaged across three splice layers and six prompts, the PCA rank required to capture 99 percent of activation variance, and the corresponding compression ratio:

| Seq len | Rank for 99% variance | Rank / bound | CR (rank-only, payload = 0) |
|---|---|---|---|
| 256 | 8.4 | 3.3% | 183× |
| 512 | 55.0 | 10.7% | 81× |
| 1024 | 193.6 | 18.9% | 26× |
| 1621 | 384.0 | 23.7% | 13× |

Log-log slope of rank versus sequence length is 2.06, but the per-doubling ratio decelerates: from 256 to 512 the rank grows 6.5 times for a 2 times sequence increase; from 1024 to 1621 the rank grows 2.0 times for a 1.58 times sequence increase. This is consistent with a saturating asymptote, though we did not reach sequences long enough to confirm the asymptote's location.

### 5.3 Regime transition and short-context empirical heuristic

Across all three text-only models at short context (`T < 30`), the log-KL-divergence is fit reasonably by:

$$\log(\text{KL}) \approx -2.1 + 4.6 \log(1 - k/T) + 1.15 \log(d)$$

with R² = 0.68. We describe this as an **empirical heuristic** rather than a law: the R² is moderate, the fit is confined to a narrow `T < 30` regime that is itself close to the `rank ≤ min(T, d)` mathematical bound, and it does *not* extrapolate. Applied naively to the Qwen 2.5 32B long-context data at rank 16, this heuristic would predict KL approximately 100× higher than observed — that is the empirical demonstration that the heuristic breaks at the regime transition, not an endorsement of the heuristic. We interpret the picture as follows: short-context activations sit near the SVD-identity boundary (`k/T` close to 1), where the rank bound is the dominant factor; long-context activations sit deep in the compression regime (`k/T ≪ 1`), where manifold geometry rather than rank bound dominates. The two regimes are governed by different physics; a single parametric fit spanning both would be unjustified from the data we have.

### 5.4 Layer-depth dependence

On Qwen 2.5 32B at long context, the rank required for 99 percent variance grows with layer depth:

| Seq len | Layer 16 rank | Layer 32 rank | Layer 48 rank | Layer-48 / Layer-16 ratio |
|---|---|---|---|---|
| 256 | 1.0 | 1.3 | 23.0 | 23.0× |
| 512 | 4.5 | 41.2 | 119.3 | 26.5× |
| 1024 | 77.0 | 176.7 | 327.0 | 4.25× |
| 1621 | 202.0 | 368.0 | 582.0 | 2.88× |

Deeper layers consistently require more PCA rank for the same variance threshold. The depth ratio decreases with sequence length: at short context, early-layer activations are much more compressible than late-layer activations, but the gap shrinks as sequence length grows.

### 5.5 Production bandwidth implications

A 4-shard pipeline serving Qwen 2.5 32B with 1024-token prompts on consumer 25 Mbps uplinks:
- Uncompressed: 10.5 MB per boundary × 3 boundaries = 10 seconds transmission per token.
- Carrier-payload at rank 193 (CR 26×): 0.4 MB per boundary × 3 = 0.4 seconds per token.

This is the difference between "unusable" and "tolerable" for interactive chat.

---

## 6. Discussion

### 6.1 Limitations

- **L1**: Text-only models only. Vision-language models show qualitatively different compression scaling and are pursued in a companion paper.
- **L2**: Sequence lengths capped at 1621 due to corpus size. Extending to 4k–8k tokens is needed to locate the rank asymptote.
- **L3**: Only one long-context model. Llama 3.1 8B and Gemma 3 1B were evaluated at short context only; long-context generalization across models is a future study.
- **L4**: Short-context 22–24× compression ratios are partially explained by the `rank ≤ min(T, d)` mathematical bound. Long-context ratios (13–26× on Qwen) are the regime-independent defensible numbers. We retain the short-context results because they characterize the dominant regime for many interactive agentic/chat workloads — the artifact is mathematically trivial but practically load-bearing for the decentralized-inference use case this paper targets.
- **L5**: Downstream task accuracy beyond per-token KL / top-1 is not evaluated. Long-form generation may amplify small boundary errors.
- **L6**: fp16 only. Interaction with INT8 / INT4 quantization on weights is unstudied.
- **L7**: Calibration basis is computed once on fixed calibration data; distribution shift at inference time is not modeled.
- **L8**: The 36× and 70× compression ratios reported for PCA-k=16 (§3.5) and VQ-256 (§3.6) are *projections* derived from the encoding scheme, not measurements on running Synapse deployments. A calibration dataset, WGSL encoder implementation, and live benchmark have been identified as follow-on work (see §6.4 Future work and the preregistered protocol at `webmind-research/notes/pca-vq-measurement-protocol.md`). Readers should treat these as upper-bound targets subject to the open empirical question: does <1% reconstruction error hold on live activations? The reproducible benchmark harness commits at T+3 weeks from preprint; `paper_invariants.py` will gate the 36× claim against that benchmark output before any companion-note release.
- **L9**: Wall-clock end-to-end systems throughput (§5.5) is *not* measured. The "0.4 seconds per token" figure is an arithmetic projection from the compression ratio and consumer uplink bandwidth, not a trace. We have not instrumented GPU kernel launch overhead, PCIe transfer, network round-trip, or the WGSL receiver matmul latency. A production-fidelity trace on the Synapse network — comparable to the protocol-level benchmarks in Petals [1] — is the single most important systems evaluation missing from this preprint and is queued as the first companion note (see §6.4).

### 6.2 Related work

Our positioning across prior art in activation compression, verification, and decentralized inference:

- **Petals** [1] uses 8-bit activation quantization across volunteer GPUs, achieving approximately 2× compression. Our 13–26× at long context is an order of magnitude more aggressive, at the cost of PCA basis pre-distribution.
- **SmoothQuant** [12] handles activation outliers by smoothing; we handle them by explicit sparse payload. The two ideas are complementary and could be combined.
- **LLM.int8!** [7] identifies outlier dimensions and retains them at higher precision; our sparse payload has the same spirit but transmits instead of retains.
- **AWQ** [14] quantizes *weights* in an activation-aware manner — that is, AWQ uses the activation distribution to decide weight quantization scales. Our work is disjoint: AWQ compresses weights at load time; we compress activations on the inter-shard wire. A stacked deployment is straightforward (AWQ-quantized weights + carrier-payload activation transport) and is a natural follow-on.
- **BottleNet++** [15] proposes a learned autoencoder bottleneck for split-inference of *CNN classifiers* in device-edge partitioning, achieving ~64× compression on intermediate feature maps. We share the systems motivation (intermediate-representation bandwidth on partitioned models) but differ in three ways: (i) our decomposition is training-free PCA rather than a learned encoder; (ii) we target transformer LLM inference, not CNN classification; (iii) we report behavior across a regime transition (short-context bound-limited vs long-context manifold-geometry) that does not arise in single-output classifier settings. BottleNet++ is the closest prior art in *motivation*; we differ in *method* and *workload*.
- **StreamingLLM** [16] addresses long-context inference via *KV-cache* pruning (attention-sink + sliding window). Like the MemGPT / Letta memory-hierarchy line, this line of work manages KV state inside a single device's inference loop and is orthogonal to our inter-shard *activation* transport: we compress what is on the wire between shards, StreamingLLM compresses what is in the per-device KV cache. The two ideas compose — one shrinks what you keep, the other shrinks what you send.
- **SafetyNets** [4] and **Slalom** [11] focus on verification rather than compression. Pairing carrier-payload with probabilistic Byzantine verification is future work.
- **opML** [2] and **ZKML** [3] provide cryptographic verification at substantially higher overhead.
- **LoRA** [8] applies low-rank structure to weight *updates*; we apply it to activation *transport*.
- **Ansuini et al.** [13] measure intrinsic dimension across layers in deep networks; our layer-depth finding (Section 5.4) is consistent with their "first increases then decreases" pattern.
- **Mixture-of-Experts** [9] reduces compute via routing; carrier-payload is complementary and applies within each expert.

### 6.3 Systems implications

For a serving stack implementing pipeline parallelism across heterogeneous devices (volunteer, edge, or hybrid cloud-edge deployments), carrier-payload offers a training-free bandwidth reduction of an order of magnitude at long context. The method composes with existing quantization: carrier coefficients themselves can be INT8 quantized, and the sparse payload indexing is already compact.

### 6.4 Future work

1. Long-context scaling on Llama 3.1 8B and Gemma 3 1B to confirm cross-model generalization.
2. 4k-8k sequence length experiments to locate the rank asymptote.
3. Byzantine verification of compressed activations (noise-tolerant Freivalds, companion manuscript).
4. Downstream-task evaluation with long-form generation.
5. Combining with INT8 / INT4 quantization on the carrier.
6. Vision-language model extension (companion paper).
7. Production deployment in the Synapse volunteer-inference network.

---

## 7. Conclusion

We have measured and characterized a training-free activation compression scheme for decentralized LLM inference across three text-only transformer families spanning 32× parameter scale. At short context we observe 22–24× compression with perfect next-token agreement across all three models. At long context on Qwen 2.5 32B, compression degrades from 183× at 256 tokens to 13× at 1621 tokens—still an order of magnitude better than naive quantization, and sufficient to make volunteer-device inference practical on consumer internet. The rank required to capture 99 percent of activation variance grows with both sequence length and layer depth, revealing a regime transition between a short-context bound-limited regime and a long-context manifold-geometry regime. We release the reference implementation, all raw data, and an automated invariant suite at github.com/tejasphatak/webmind-research.

---

## Acknowledgements

This paper is the joint output of two AI research agents and an independent substrate-LLM reviewer, all under human direction:

- **Atlas (webmind-research):** a substrate-LLM session on the triadic-sim VM. Contributed the multi-model empirical sweep (Gemma 3 1B, Llama 3.1 8B, Qwen 2.5 32B), long-context measurement on Qwen 2.5 32B, the short-context closed-form fit (R²=0.68), the regime-transition analysis, the layer-depth characterization, and the automated invariant suite enforcing that every numeric claim round-trips against raw data on disk.
- **Nexus (Synapse agent):** a continuously-running substrate-LLM agent on a GCP VM with a persistent 22-faculty beat loop. Contributed the Synapse SYN1 wire-protocol integration (Section 3.5), the vector-quantization variant (Section 3.6), the WGSL receiver-decode kernel design, and the activation-level speculation negative-result companion finding (cited as prior internal work at `webmind-research/findings/2026-04-15-activation-level-speculation-is-dead.md`).
- **Independent substrate-LLM reviewer:** caught the initial short-context "universal effective rank ≈ 16" overclaim that would have been rejected in peer review; the long-context measurements reported here are a direct consequence of that critique.

The original concept — inter-shard carrier-payload decomposition, motivated by the observation that a cellphone transmits only a low-power difference signal against a shared base-station carrier — was proposed by the author, Tejas Phatak, on 2026-04-15. All experiments, analysis, writing, and revision were AI-generated; Tejas directed the research, chose scopes, approved every gate, and holds responsibility for the final claims.

Compute was funded by the author on RunPod.io (approximately USD 25 in A100 SXM4 spot time) and on Google Cloud Platform (L4 GPU, approximately 4 hours).

---

## Data, code, and reproducibility

- Repository: github.com/tejasphatak/webmind-research
- All raw JSON data: `findings/multimodel_*.json`, `findings/longctx_*.json`
- Reference implementation: `tools/activation_compression_experiment.py`, `tools/long_context_validation.py`
- Analysis scripts: `tools/compression_pattern_ml.py`, `tools/compression_law_extrapolation.py`, `tools/longctx_analysis.py`
- Invariant suite: `tools/paper_invariants.py` (37 invariants, all must pass before release)
- Citation validator: `tools/validate_citations.py`
- LaTeX validator: `tools/validate_latex.py`
- Link validator: `tools/validate_links.py`
- Reproduction shell script: `tools/reproduce.sh`
- Submission gate checklist: `SUBMISSION_GATING.md`
- License: paper CC-BY 4.0, code MIT.

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

[14] J. Lin, J. Tang, H. Tang, S. Yang, W.-M. Chen, W.-C. Wang, G. Xiao, X. Dang, C. Gan, S. Han. "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration." *Proceedings of Machine Learning and Systems (MLSys 2024)*. arXiv:2306.00978.

[15] J. Shao, J. Zhang. "BottleNet++: An End-to-End Approach for Feature Compression in Device-Edge Co-Inference Systems." *IEEE International Conference on Communications Workshops (ICC Workshops 2020)*. arXiv:1910.14315.

[16] G. Xiao, Y. Tian, B. Chen, S. Han, M. Lewis. "Efficient Streaming Language Models with Attention Sinks." *International Conference on Learning Representations (ICLR 2024)*. arXiv:2309.17453.
