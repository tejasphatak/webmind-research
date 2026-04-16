# P2 + P4 Cross-Verification Synthesis

**Date:** 2026-04-16
**Inputs:** `gemini_responses/p2_multi_faculty.md`, `gemini_responses/p4_multi_faculty.md`

---

## P2 — "Portable ML on WebGPU"

### What survives
- **Measurement paper framing** — not novel systems paper. Value = dataset + failure taxonomy.
- **Intel iGPU failure root-cause** — publishable if properly diagnosed (TDR? OOM? maxBufferSize?)
- **Cross-vendor fidelity matrix** — first to systematically quantify WGSL-relaxation-induced drift across consumer WebGPU implementations
- **WGSL FP relaxation as the mechanism** — explains why the drift exists, not just that it does

### What gets killed
- **Heterogeneous speedup narrative** (2.34x Qualcomm+Nvidia) — belongs in P3, not P2. Gemini flagged split personality.
- "First cross-platform distributed inference" headline — it's still cool, but not this paper's story

### Required measurements beyond what's already collected
1. Layer-by-layer error accumulation at Layer 1, 6, 12 of GPT-2
2. KL divergence on final token distributions (not just "same text out")
3. WGSL precision toggle: f32 vs f16 — does drift disappear at f32?
4. Intel crash threshold: batch-size and seq-len sweep to find exact failure boundary

### Verdict
- **Venue:** ISPASS workshop or IEEE Micro short paper
- **Timeline:** July–September 2026 (write-up from existing data + new measurements)
- **Novelty:** systematic WGSL-fidelity measurement matrix on consumer hardware — a real gap

### Invariants needed for P2 (to add to paper_invariants.py)
- [INV-P2-1] Measurements include all claimed device classes (Apple, PowerVR, Adreno, Nvidia, Intel)
- [INV-P2-2] Reported KL divergence per layer per device pair is computable from raw tensor dumps
- [INV-P2-3] Intel failure is reproducible and has documented exact error code
- [INV-P2-4] WGSL f32 vs f16 comparison runs same model, same prompt, different shader variant

---

## P4 — "Communication-Theoretic Protocol for Decentralized LLM Inference"

### What survives (genuinely novel synthesis)
1. **Byzantine Joint Source-Channel Coding (B-JSCC)** — modeling LLM inference as comm channel where *channel coding* is Freivalds-style interactive proofs instead of error-correcting codes. GAP FOUND in prior art.
2. **Noise-Tolerant Freivalds for Lossy Pipelines** — mathematical adaptation that distinguishes benign quantization noise (from P1's compression) from malicious Byzantine injection. Solves false-slashing in decentralized networks.
3. **Rate-Distortion-Trust (RDT) bound** — 3-axis extension of classic rate-distortion theory: Bandwidth × Accuracy × Verification-Probability.
4. **Adaptive Verification Modulation** — dynamically adjust Freivalds probability based on token entropy / importance.

### Constituent prior art (must cite, must differentiate from)
- SafetyNets (Garg et al. 2015) — sum-check protocols for NN verification
- Slalom (Tramèr et al. 2019) — Freivalds for outsourced matrix multiplication in TEEs
- BottleNet++ (Shao et al. 2019) — info-bottleneck for split computing
- Deep JSCC for images (Bourtsoulatze 2019) — different media, same framework
- opML (Ora, Hyperbolic) — fraud proofs, full re-execution required
- ZKML (EZKL, Modulus) — 1000x+ prover overhead
- FedML / Deep Gradient Compression — training not inference

### The actual novel claim (after cross-verification)
> "Verification over Compressed State (VoCS): running probabilistic Byzantine checks on carrier-payload compressed activations without false-positive triggering from quantization noise, while maintaining adversarial soundness. This is the missing piece between lossy compression (P1) and cryptographic verification (existing Freivalds/Slalom)."

### Timeline — PUSH BACK
- **Original:** Q2 2027
- **Revised:** Q4 2027 / Q1 2028
- **Why:** P4 depends on P1 long-context validation + P3 residual-transport empirics. Can't write the unified theory until both are empirically grounded. Premature P4 = handwaving.

### Venue target
- IEEE TIT (Trans Info Theory) — for the theory
- NSDI — for the systems framing
- NeurIPS — if positioned as ML systems

### Invariants needed for P4
- [INV-P4-1] Noise-tolerant Freivalds demonstrated: false-accept rate < ε on quantization noise, > δ on adversarial injection, empirically measured
- [INV-P4-2] RDT-bound derivation matches simulation within tolerance on synthetic + real traces
- [INV-P4-3] Every cited prior-art paper has a specific sentence in Related Work differentiating P4's contribution

---

## Revised 4-paper roadmap (post-cross-verification)

| Paper | Title | Timeline | Venue target | Status |
|---|---|---|---|---|
| P1 | Carrier-Payload (narrowed) | Apr–May 2026 arXiv | MLSys / workshop | Data in hand, invariants passing, long-ctx pending |
| P2 | WebGPU Fidelity Measurement | Jul–Sep 2026 | ISPASS / IEEE Micro | Data partially collected, need new measurements |
| P3 | Predictive Residual Transport (pivoted) | Sep 2026 – Feb 2027 | MLSys workshop / NSDI | predictor.js exists; need to instrument & measure |
| P4 | B-JSCC Flagship | Q4 2027 – Q1 2028 | IEEE TIT / NSDI / NeurIPS | Depends on P1+P3 being solid |

**Net change after cross-verification:**
- P1 still strong, narrower claim, clearer differentiation from BottleNet++/Pluralis
- P2 scope tightened (kill heterogeneous angle, focus on fidelity)
- P3 pivoted from "Distributed Speculative Decoding" (dead) to "Predictive Residual Transport" (alive)
- P4 pushed back, narrower novel claim ("Verification over Compressed State"), depends on P1+P3

**4 papers is still achievable** but on a slower, honest, defensible timeline.

---

## What I learned from this cross-verification exercise

1. **AI brainstorming generates ideas fast but reliability is low.** Half of our "novel" ideas had been published already. Without the cross-verifier, we would have claimed novelty on well-known work. Review-1 would have destroyed us.

2. **Primary-source validation > AI summaries.** Gemini correctly flagged BottleNet++, SafetyNets, etc., but its descriptions were sometimes inaccurate. I had to WebFetch the actual arxiv abstracts to confirm.

3. **Convergent independent ideas signal likely prior art.** When Claude and Gemini independently propose the same thing, the field has probably proposed it too. Extra literature caution needed.

4. **Novelty shrinks under scrutiny; that's OK.** The paper becomes smaller but more defensible. Narrow-but-solid > broad-and-handwavy.

5. **Invariants catch hallucinations.** My claim that "ranks [2,4,8,16,32,64] were tested" was wrong (only [2,4,8,16]). Invariant check caught it. Every paper claim should be an invariant.

— Claude Opus 4.6, with Gemini 3.1 Pro cross-verification
