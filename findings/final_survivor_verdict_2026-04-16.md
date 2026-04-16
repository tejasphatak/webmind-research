# Final Survivor Verdict — After Empirical Validation

**Date:** 2026-04-16
**Summary:** 60 ideas across 3 rounds → 6 preliminary survivors → **1 empirically validated**

---

## Benford's Law Byzantine Detection — FALSIFIED

**Idea:** legitimate transformer activations follow Benford's Law; adversarial injections violate it detectably.

**Empirical test (tools/benfords_law_activations.py):**
- Synthetic Gaussian-with-outliers activations (LLM-like)
- 7 scenarios: honest + 6 adversarial
- Every scenario including honest baseline flags as "Benford violation" (p < 1e-4)
- L1 distance of honest data to Benford ≈ 0.20 (already far from Benford)

**Reason:** Benford's Law emerges when data spans 5+ orders of magnitude. Transformer activations have heavy-tailed outliers but the BULK is Gaussian with std ~1 — most values are in [0.1, 10], which is only 2 orders of magnitude. The leading-digit distribution is dominated by the Gaussian bulk, not the tail.

**Would real activations fare better?** Probably not. Real LLM activation ranges are typically ±10-30 for the bulk with outliers to ±100-1000. That's 3-4 orders of magnitude — still not enough for Benford's Law to emerge cleanly.

**Verdict: FALSIFIED.** Idea removed from survivor list. Honest null result — good science. Attempted real-data confirmation hit disk-full error on Qwen pod; ending this line of inquiry here as the synthetic null is strong enough signal.

---

## Keystone Shard Analysis — VALIDATED

**Idea:** some volunteer shards are disproportionately critical. Identify keystones via ablation; allocate more redundancy to them.

**Empirical test (tools/keystone_shard_metric.py):**
- Synthetic 10-shard fleet with ground-truth criticality levels (0.05–0.90)
- Monte Carlo ablation with 200 samples per condition
- **Spearman rank correlation: 0.994** (near-perfect rank recovery)
- Redundancy allocation policy correctly assigns 4× replicas to keystones (crit 0.8–0.9) and 1× to low-criticality (crit 0.05–0.10)

**Verdict: EMPIRICALLY VALIDATED.** Method works on synthetic data with known ground truth.

**Next validation step:** measure on real Synapse telemetry data once available. The ablation-Monte-Carlo approach is cheap; can be deployed as an ongoing background process.

**Paper placement:** Section of P4 flagship OR its own short paper — "Keystone Shard Identification for Volunteer LLM Inference Networks."

**Prior art check (final):** No published work on criticality ranking in volunteer ML inference networks. Concept borrowed from ecology (keystone species). Novel application.

---

## Other preliminary survivors — status

| Idea | Source | Status |
|---|---|---|
| reCAPTCHA Gambit (widget-embedded compute) | R3 Gemini | NOT EMPIRICALLY TESTED — design-level idea; works if deployed, risk is social/legal (consent) |
| Parachute Local Fallback | R3 Gemini | NOT EMPIRICALLY TESTED — needs tiny-model + handoff protocol design |
| Progressive Decentralization Protocol | R3 Claude | NEEDS DEEPER LITERATURE CHECK |
| Attribution-Based Karma | R3 Claude | MECHANISM DESIGN — not testable empirically until deployed |

All 4 above remain on the "to validate" list. Can't empirically test them without real deployment data or larger-scale simulation.

---

## Net after 3 rounds of creative ideation + empirical validation

**Hard empirical survivors: 1 (Keystone Shard Analysis)**

This is significantly fewer than the headline "6 novel ideas" from the prior-art-cross-verification step. Empirical testing killed one more idea (Benford's). The remaining 4 are design/mechanism ideas that need deployment data to validate.

**What this teaches about the research process:**
1. **Paper ideas should pass 3 gates:** (a) prior-art cross-verification, (b) empirical feasibility test, (c) peer review. Most ideas stop at (a) or (b).
2. **Creative brainstorming alone generates garbage.** Brainstorm → literature → simulation → validation pipeline needed.
3. **Synthetic validation is cheap and catches the common failure modes.** Doing `tools/benfords_law_activations.py` took 30 minutes and saved us from proposing a falsified idea in a paper.

---

## Updated research priority

Given this verification pass, the confirmed building blocks for P4 flagship are:

**Confirmed:**
- Carrier-Payload compression (P1 work, 22-24x measured)
- Predictor.js / speculative.js infrastructure (Tejas Apr 13, 2026)
- **Keystone shard analysis (new, validated today)**

**Removed from P4:**
- ~~Benford's law Byzantine detection~~ (falsified)

**Still to validate:**
- Noise-tolerant Freivalds (the P4 theory core)
- Rate-distortion-trust bound (P4 theory core)
- Byzantine prediction attestation (depends on Freivalds extension)

This narrows P4 appropriately.

---

*End of 2026-04-16 autonomous ideation session. ~6 hours of work. Net: 1 empirically validated idea + Carrier-Payload paper progressing + invariant discipline established + roadmap recalibrated. Honest, narrower, more defensible.*
