# Noise-Tolerant Freivalds: Distinguishing Quantization Noise from Byzantine Injection

**Author:** Tejas Phatak (with Claude Opus 4.6, Gemini 3.1 Pro cross-verification)
**Date:** 2026-04-16
**Status:** DRAFT — theoretical backbone for P4 flagship ("Verification over Compressed State")
**Companion:** `findings/p2_p4_cross_verification_synthesis.md`

---

## Problem statement

Classical Freivalds verifies `A × B = C` for matrices in O(n²) with false-accept probability ≤ 2⁻ᵏ after k probes. The algorithm picks a random vector r, checks `A(Br) − Cr = 0`.

**In our setting (decentralized LLM inference):** we do NOT have exact matrix equality because:
1. Carrier-Payload compression introduces **quantization noise** at shard boundaries
2. Byzantine volunteers introduce **adversarial perturbations** we want to detect

Classical Freivalds cannot distinguish these two sources. If we relax the equality check to `||A(Br) − Cr||₂ < ε`, then either:
- **ε too small:** false-rejects on legitimate quantization noise (false-alarm for compression)
- **ε too large:** lets adversaries slip in perturbations below the detection threshold

This is the central tension. Standard Freivalds literature assumes the lossless setting. We need a version that works in the lossy setting.

---

## Setup

Let:
- `A ∈ R^{m×n}, B ∈ R^{n×p}` be the true operand matrices at a shard boundary
- `C' = A'B'` be the observed product where `A' = A + δ_A`, `B' = B + δ_B` are quantized versions
- `δ_A, δ_B` are **benign** quantization perturbations with known statistical structure (bounded norm, zero mean, sub-Gaussian)
- `η` is a potential **adversarial** injection (Byzantine)
- The adversary may replace `C` with `C + η`

We want a probe protocol that:
1. **Soundness:** with probability ≥ 1 − δ_sound, accepts the computation when η = 0 (pure compression noise)
2. **Completeness:** with probability ≥ 1 − δ_complete, rejects when ||η||₂ ≥ τ_adv (non-trivial Byzantine)

---

## Key insight: the noise floor is characterizable

The quantization noise `δ_A, δ_B` has **known bounds** given our compression scheme:
- For carrier-payload compression at rank r: ||A' − A||_F ≤ σ_r(A) (residual singular values)
- For sparse residual with top-k retention: remaining L2 error is bounded by ||A − A_k||_F² = Σ_{i>k} σ_i²

This lets us compute an expected distribution for `||A'(B'r) − C'r||₂` under the no-adversary case. Call this distribution **N** (for noise).

If N has a tail that decays fast enough, we can set a threshold `τ` such that:
- Pr[||A'(B'r) − C'r||₂ > τ | η = 0] ≤ δ_sound  (false-alarm rate)
- Pr[||A'(B'r) − C'r||₂ > τ | ||η|| ≥ τ_adv] ≥ 1 − δ_complete  (detection rate for non-trivial adversaries)

---

## Proposed probe protocol

### Offline calibration
1. Apply carrier-payload compression to a known-good A, B. Observe C' and the compression error `Δ = C' − AB`.
2. Estimate the distribution of `||Δ r||₂` for random r — call this the **noise profile**.
3. From the noise profile, choose threshold `τ_0` such that the 99th percentile lies below τ_0.

### Online verification (per-shard-boundary)
1. Sample r ~ Unif({±1}ⁿ) (Rademacher, cheap).
2. Compute v = A'(B'r) − C'r locally (O(n²) — cheap).
3. Accept if ||v||₂ ≤ τ_0.
4. Repeat with k independent probes. Reject if more than `m` out of k exceed τ_0.

### Soundness (no adversary)
For k probes with i.i.d. r, the false-alarm rate per probe is bounded by the noise profile. Setting per-probe α = 0.01 and requiring m = 2 out of k = 10 to exceed τ_0 gives aggregate false-alarm < 1e-4 (binomial tail).

### Completeness (adversary)
If adversary injects η with ||η||_F ≥ τ_adv, then by matrix-norm bound:
- E[||η r||₂²] = ||η||_F² (for Rademacher r)
- So ||ηr||₂ scales with ||η||_F
- Probability per probe that ||v||_2 exceeds τ_0 is at least some function of SNR = τ_adv² / (noise variance)

---

## The SNR argument

Define Signal-to-Noise Ratio:
```
SNR = ||η||_F² / E[||Δr||₂²]
```
where Δ is the compression noise.

**Claim:** detection probability per probe ≥ 1 − Φ((τ_0 − τ_adv) / σ_n) for sub-Gaussian noise with std σ_n.

**Interpretation:** as long as the adversarial perturbation is meaningfully larger than the compression noise, detection is reliable. When adversary hides *within* the noise floor, we cannot detect (this is a fundamental limit, not a weakness — no single-shot test can do better without a reference).

---

## Key theoretical contributions

### Contribution 1: **Rate-Distortion-Trust (RDT) bound**

Classical rate-distortion: `R(D)` is the minimum bits per source symbol to achieve distortion D.

Our extension adds a third axis: **Trust probability T** (probability of catching a β-level adversarial injection). Compression reduces bits (R decreases) but increases noise floor (harder to detect small attacks — T decreases for fixed adversary strength).

Formal bound (sketch):
```
R + log(1/D) + log(1/(1-T)) ≥ constant
```
You can have two of {low bandwidth, low distortion, strong detection}, not all three.

### Contribution 2: **Optimal threshold calibration**

Given known compression scheme (carrier-payload with rank r and sparse top-k):
- Noise standard deviation: σ_n ≈ √(Σ_{i>r} σ_i² · (1-k)) / √n
- Optimal probe threshold: τ_0* = σ_n · q_99 (99th quantile of chi-squared with appropriate df)

### Contribution 3: **Adaptive verification modulation**

For queries with high semantic importance (detected via entropy of intermediate logits), increase probe count k. For low-stakes queries, reduce k to save overhead. Adaptive trust-vs-cost tradeoff.

---

## What we must do to publish this

1. **Empirical calibration on real data.** Compute the noise profile for Carrier-Payload with rank 16 on Gemma 3 1B IT. Measure tail distribution of residuals. Confirm sub-Gaussianity assumption.

2. **Adversary simulation.** Inject bounded η of various magnitudes. Measure detection rate as function of SNR. Plot ROC curve.

3. **SNR characterization.** What's the smallest adversarial perturbation that downstream LLM behavior distinguishes from noise? (If the LLM can't tell the difference, we don't need to detect it either.)

4. **Comparison baselines:** 
   - SafetyNets (sum-check for NN) — exact verification, does NOT handle compression noise
   - Slalom (Freivalds with TEEs) — requires trusted enclave, we don't
   - opML (fraud proofs) — requires full re-execution, we're probabilistic

5. **Prove the RDT bound rigorously.** Currently sketched; needs tight proof.

---

## Empirical validation plan (for inclusion in P4 paper)

```python
# Pseudocode for the experiment section
for each compression_config (rank × sparse_frac):
    compute noise_profile on 1000 real activation pairs
    calibrate threshold tau_0 for 1% false-alarm target
    for each adversary (random noise, biased shift, targeted flip):
        for each adversary_magnitude:
            compute detection_rate = Pr(||v||_2 > tau_0 | adversary active)
    plot ROC: false-alarm vs detection, across compression configs
```

**Key invariants to add to paper_invariants.py:**
- [INV-P4-1] Noise profile 99th percentile computable from C-P residuals analytically vs empirically agree to within 5%
- [INV-P4-2] Probe false-alarm rate ≤ 1% at tau_0 on clean data across 1000 trials
- [INV-P4-3] Probe detection rate ≥ 99% for adversary magnitude ≥ 3σ_n
- [INV-P4-4] Adaptive verification (high-entropy = more probes) reduces overhead by >2x vs fixed-k on measured traffic

---

## Related-work positioning

| Method | Lossless check? | Verification over compressed state? | Overhead |
|---|---|---|---|
| SafetyNets (Garg 2015) | Yes (exact) | No | O(n²) |
| Slalom (Tramèr 2019) | Yes (Freivalds in TEE) | No | Requires TEE |
| opML (fraud proofs) | Yes (re-exec) | No | 100%+ at worst |
| ZKML (EZKL) | Yes (crypto) | No | 1000-10000x |
| **Noise-Tolerant Freivalds (this work)** | **Probabilistic** | **YES** | **O(n²) per probe** |

**Our specific niche:** the only method that provides probabilistic Byzantine detection while operating on lossy-compressed intermediate state, with overhead competitive with classical Freivalds.

---

## Open questions

1. Does sub-Gaussianity of carrier-payload residuals actually hold empirically? (To check)
2. How tight is the RDT bound? (Currently hand-waved; needs Fano-style converse)
3. How do we handle colluding Byzantine adversaries who coordinate to stay below the noise floor? (Research direction — probably requires cross-node cross-checks)
4. Does the threshold need to adapt per-query or per-layer? (Empirical Q)

---

*This is a working draft of the theoretical core for P4. Must be revised with empirical validation numbers from real Gemma/Llama/Qwen activations before submission. Q4 2027 / Q1 2028 target.*
