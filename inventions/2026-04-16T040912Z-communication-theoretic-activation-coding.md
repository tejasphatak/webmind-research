---
name: Communication-Theoretic Framework for Neural Activation Transport
description: Apply wireless-communications engineering (QAM, OFDM, JSCC, rate-distortion theory) to activation transport in decentralized LLM inference. Generalizes Carrier-Payload compression to a formal information-theoretic framework. To be absorbed into P4 flagship, not a separate paper.
type: project
status: CAPTURED — to be absorbed into P4 (Trust/Compress/Verify flagship)
---

**Inventor:** Tejas Phatak (Webmind Research)
**Recorded:** 2026-04-16
**Status:** Timestamped for priority. To be developed as the theoretical backbone of P4 flagship paper (Q1 2027).

---

## The Core Idea

Deep learning currently treats activation transport between distributed shards as a **naive serialization problem** — raw fp16 floats over TCP. Meanwhile, wireless communications has 70+ years of theory for exactly this: how to pack information efficiently onto a channel, how to survive noise, how to trade off rate vs distortion.

**Proposal:** Apply communication-theoretic framing to neural activation transport in decentralized inference. Specifically:

1. **Activations as a signal source** — treat the distribution of activations as a stochastic source with measurable entropy and manifold structure.
2. **Network links as a channel** — model internet paths as AWGN/packet-loss channels with measurable capacity.
3. **Task accuracy as the distortion metric** — rate-distortion theory optimized against downstream task performance, not reconstruction MSE.
4. **Modulation-style encoding** — pack multiple activation dimensions per "symbol" using constellation designs analogous to QAM/OFDM.
5. **Joint source-channel coding** — co-design compression and error resilience, matching Shannon's 1948 framework.

---

## Why This Matters

Current state:
- Carrier-Payload (our P1) achieves 10-22x compression empirically via PCA + sparse residual.
- No theoretical framework explains *why* this is possible, *what* the information-theoretic optimum is, or *how* to extend it.

With this framing:
- P1's 22x compression becomes an **instance** of a general rate-distortion curve.
- The intrinsic-rank finding (~32 dims of 1536) becomes a statement about **source entropy** of transformer activations.
- Byzantine verification becomes **error detection coding**.
- Adaptive precision (Synapse Phase 4 int4 toggle) becomes **adaptive modulation**.
- The whole system becomes a **coherent protocol** with provable bounds.

This elevates Synapse from a clever systems project to a **theoretical contribution** — which is what reviewers (and USCIS expert letter writers) notice.

---

## Prior Art (acknowledged)

### Classical information theory
- Shannon 1948 — source-channel coding separation
- Berger 1971 — Rate Distortion Theory
- Slepian-Wolf 1973, Wyner-Ziv 1976 — distributed source coding

### Deep Joint Source-Channel Coding (DJSCC)
- Bourtsoulatze, Gündüz 2019 — "Deep Joint Source-Channel Coding for Wireless Image Transmission"
- Kurka, Gündüz 2020 — DJSCC with attention
- Yang et al. 2022 — modulation-aware neural codecs

### Neural compression
- Ballé, Minnen et al. — learned image compression (Google)
- Habibian et al. 2019 — video neural compression
- Implicit neural representations (COIN, NeRF-compression)

**Critical observation:** Existing DJSCC work focuses on **images and video over wireless channels**. Almost no published work on:
- LLM activation transport specifically
- Task-accuracy-optimized distortion metrics for transformer inference
- Joint source-channel coding for pipeline-parallel inference across the public internet

---

## The Novel Contribution

A rigorous information-theoretic framework for distributed LLM inference, specifically:

1. **Source-entropy characterization of transformer activations** — measure H(activation) layer-by-layer, across models. Carrier-Payload's intrinsic-rank finding is the empirical version; generalize to full rate-distortion curves.

2. **Task-accuracy distortion function** — define D(X, X̂) as downstream KL divergence or task-score degradation, not MSE. Derive coding schemes optimal under this distortion.

3. **OFDM-style activation transport** — encode activation vectors as multi-carrier symbols. Parallels between subcarriers in OFDM and PCA components in Carrier-Payload. Formal analysis of why this is rate-optimal.

4. **Joint source-channel coding for decentralized inference** — co-design compression + error correction for the specific task: surviving volunteer-device dropout + poisoning + bit errors while preserving task accuracy.

5. **Capacity bound for decentralized LLM inference** — what is the information-theoretic minimum bandwidth for task-accurate inference? Carrier-Payload's 22x is a lower bound on what's achievable; the theoretical upper bound is unknown.

---

## How This Absorbs Into P4 Flagship

Original P4 pitch was "Trust, Compress, Verify" covering three independent contributions.

Revised P4 pitch: **"A Communication-Theoretic Protocol for Decentralized LLM Inference"**

- **Section 1: Source Coding.** Carrier-Payload + rate-distortion theory for activation compression.
- **Section 2: Channel Coding.** Byzantine verification as probabilistic error detection (Freivalds).
- **Section 3: Joint Coding.** Co-designed compression + verification + adaptive precision.
- **Section 4: Capacity Bounds.** Theoretical upper/lower bounds on decentralized LLM inference bandwidth.

This is a **much stronger paper** — unifies three empirical contributions under one theoretical framework.

Tradeoff: pushes P4 from Q1 2027 to Q2 2027. Worth it for the quality jump.

---

## What Is NOT Being Claimed

- Not claiming Shannon's theorems are novel (they're 70 years old)
- Not claiming Deep JSCC is novel (exists for images/video)
- Not claiming the analogy "wireless comm ↔ neural transport" is novel (has been noted in passing)
- Not claiming any single constituent technique is novel

**What IS being claimed:** the *specific synthesis* — applying DJSCC + rate-distortion + modulation theory to **decentralized LLM inference** — and deriving task-accuracy-optimal coding schemes for this setting.

---

## Thread 2 — Frequency-coded neural computation (SEPARATE, LOWER PRIORITY)

Brief secondary idea: neurons in biological brains communicate via frequency-coded signals (theta, alpha, beta, gamma bands). Artificial neurons do not. There may be value in designing artificial activations that use multiple frequency channels for different feature classes, enabling resonance-based computation.

This intersects with:
- Coupled oscillator computing (Ising machines)
- Neuromorphic systems
- Hopfield networks with oscillatory dynamics

**Status:** Too speculative for current roadmap. Separate invention claim if ever pursued. Not part of P4.

---

## Thread 3 — Physical-layer photonic transport (PARKED)

Using actual lasers / wavelength-division multiplexing / photonic chips for inter-device activation transport. Real field (Lightmatter, Lightelligence, MIT Shen group) but requires photonic hardware. Links to the analog-CIM parked project.

**Status:** Parked with the analog-CIM invention. Not part of the current roadmap.

---

## Actionable Next Steps (post-P1)

1. **Literature deep-dive** (July-Aug 2026, after P1 ships):
   - Read Bourtsoulatze 2019, Kurka 2020, Yang 2022 end-to-end
   - Re-read Carrier-Payload in light of DJSCC framing
   - Survey recent NeurIPS/ICLR papers on task-aware compression

2. **Theoretical development** (Sept-Dec 2026, alongside P3):
   - Derive rate-distortion bound for transformer activations (for specific architectures)
   - Formalize task-accuracy distortion function
   - Design OFDM-analogous encoding scheme

3. **Empirical validation** (Jan-Mar 2027, during P4 drafting):
   - Measure source entropy across models (Gemma, Llama, Qwen, Gemma 4)
   - Empirical rate-distortion curves
   - Demonstrate capacity-approaching scheme

4. **Integration into P4** (Q2 2027):
   - Unified protocol description
   - Theoretical framework section
   - Experiments showing each component

---

## Priority Claim

This document records the specific synthesis of **communication-theoretic framing for decentralized LLM inference, including OFDM-style activation coding, task-accuracy rate-distortion theory, and joint source-channel coding for transformer activations** as of 2026-04-16.

Constituent ideas are pre-existing; the specific application and synthesis are recorded here for future development.

---

## Honest Note

This is a theoretically ambitious direction. Full execution requires:
- Comfort with rate-distortion theory (doable with study)
- Comfort with channel coding (doable)
- Careful related-work survey (6+ weeks)
- Writing for a more theoretical audience than the systems P1 paper

**Risk:** the theory piece requires a level of mathematical formalism that can slow a paper down significantly. If P4 starts lagging in Q2 2027, we should be willing to descope this absorption and keep P4 as the original "Trust/Compress/Verify" systems paper.

---

*Captured 2026-04-16 04:09 UTC during active research phase. To be developed per ROADMAP. No further immediate action; move forward with P1 shipping and multi-model validation.*
