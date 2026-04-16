# P3 Multi-Faculty Synthesis + Literature Cross-Verification

**Date:** 2026-04-16
**Inputs:**
- `findings/p3_multi_faculty_claude.md` — Claude's independent analysis
- `gemini_responses/p3_multi_faculty.md` — Gemini's independent analysis
- Primary-source validation: SpecPipe, FlowSpec, EAGLE, PPSD, PipeInfer

---

## 💀 Original P3 scope is DEAD

"Distributed Speculative Decoding for LLM inference" has been published multiple times, all predating Tejas's April 13, 2026 implementation:

| Paper | Date | Method | Scope |
|---|---|---|---|
| SpecPipe (arXiv:2504.04104) | Apr 2025 | Draft model + tree speculation | Datacenter pipeline parallelism |
| FlowSpec (arXiv:2507.02620) | Jul 2025 | Tree speculative decoding | Network edge distributed |
| PPSD (arXiv:2509.19368) | Sep 2025 | Verify-while-draft, early-verify | Pipeline parallel |
| PipeInfer (referenced) | 2024 | Speculative drafting + verification in parallel pipelines | Distributed |
| EAGLE (arXiv:2401.15077) | 2024 | Lightweight MLP predicts 2nd-to-top-layer features | Single-node |

**SpecPipe alone achieves 4.19-5.53x speedup. FlowSpec 1.37-1.73x on real edge hardware.** Reviewers will reject anything that claims to be "first at distributed speculative decoding."

---

## ✅ What's still novel after cross-verification

Tejas's specific implementation (`predictor.js` + `speculative.js`, Apr 13 2026) uses a DIFFERENT primitive from the prior art:

| Dimension | SpecPipe/FlowSpec | Tejas's system |
|---|---|---|
| Predictor | Small draft **MODEL** | **Linear extrap + EMA** on recent activations |
| Compute cost | Extra GPU for draft | Near-zero (no additional forward pass) |
| What's predicted | Next **TOKEN** | Next **ACTIVATION at shard boundary** |
| Primary goal | Latency hiding | Latency hiding + (possible) bandwidth compression |
| Byzantine angle | Not addressed | Addressed via attestation |
| Deployment | Datacenter GPUs / edge servers | WebGPU volunteer browsers |

These are meaningfully different systems.

---

## 🎯 Genuinely novel synthesis (both Claude and Gemini converged independently)

**Residual-Only Transport via Shared Online Predictor**

Both upstream Node A and downstream Node B run the same online activation predictor. Node A:
1. Computes true activation X
2. Computes predicted X̂ using same algorithm as Node B
3. Sends only residual R = X - X̂ over the wire (typically sparse/small)
4. Quantizes R aggressively since it's small
5. Node B reconstructs X' = X̂ + R

**This is different from:**
- SpecPipe/FlowSpec: transmit full draft tokens, verify after
- EAGLE: local draft model, no inter-device transport
- Carrier-Payload (P1): one-shot compression without prediction
- BottleNet++: learned encoder, single-shot (no prediction)

**Combined with Carrier-Payload:** if R is sparse, we can carrier-payload compress R. **Stacked savings.**

This specific combination (shared online predictor + residual transport + carrier-payload on residuals + Byzantine attestation) appears not to be published based on primary-source checks performed 2026-04-16. Must do a more exhaustive search before final claim.

---

## Proposed P3 retitle

**"Predictive Residual Transport for Decentralized LLM Inference"**

### Contributions (ranked)

1. **Shared online predictor + residual-only transport protocol** (HIGH novelty pending further lit check)
2. **Carrier-Payload compression on residuals** (SYNTHESIS of P1 + P3)
3. **Zero-extra-GPU predictor** (linear extrap/EMA/lightweight MLP) suitable for consumer devices, contrasting with draft-model approaches
4. **Byzantine-tolerant prediction attestation** (ties to P4 flagship)

### Honest caveats

- Tejas's predictor.js is linear extrapolation + EMA. Simple. The paper's theoretical sections need to position this as "adequate" not "optimal." (Kalman filter extension would be a small follow-up.)
- EAGLE used MLP for feature prediction. Our linear/EMA is simpler but we should explain WHY it's sufficient (low-dimensional manifold structure from P1 — our predictor only needs to track the 16-dim manifold, not the full hidden space).
- The 0.995 cosine threshold is arbitrary. Paper should include the logit-safe bound (Gemini's contribution #2) to make it rigorous.

### Experiments required

1. **Measure predictor hit rate** on Tejas's existing Synapse deployment (Pixel + iPhone + Nvidia). Already infrastructure for this in speculative.js stats.
2. **Residual entropy measurement**: given a good predictor, how compressible are residuals? Compute H(R) vs H(X).
3. **Combined compression ratio**: carrier-payload on residuals vs on raw activations. Does the synthesis multiply?
4. **Byzantine ablation**: what happens to hit rate under adversarial upstream? Attestation defense impact.

---

## Invariants for the P3 paper (per CONVENTIONS.md)

When P3 has data, the following invariants must be in tools/paper_invariants.py:
- [INV-P3-1] Measured predictor hit rate ≥ X% on real activations
- [INV-P3-2] Residual entropy H(R) < H(X) (predictor actually helps)
- [INV-P3-3] Combined compression CR(R + carrier-payload) > CR(X + carrier-payload)
- [INV-P3-4] Byzantine attestation catches adversarial predictions at rate ≥ Y%

Numbers TBD; will populate from experiments.

---

## Meta: the cross-verification discipline worked

Without the literature cross-verifier, we would have:
1. Proposed "Distributed Speculative Decoding" as P3
2. Claimed novelty
3. Gotten rejected at MLSys review with citations to SpecPipe/FlowSpec/PPSD

Instead we caught it before investing months. This pattern should repeat for P2 and P4 before any committed work.

— Claude Opus 4.6, cross-verified with Gemini 3.1 Pro
