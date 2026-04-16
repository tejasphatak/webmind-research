# Weird Faculties — Round 2 Synthesis + Validation

**Date:** 2026-04-16
**Inputs:** `weird_faculties_round2_claude.md` + `gemini_responses/weird_faculties_round2.md`

---

## Gemini Round 2 was more technical (and more often prior art)

### Gemini Round 2 ideas validated against literature

**UV-Plane Baseline Sparsification (VLBI):**
- ALREADY PUBLISHED: MixKV (importance + diversity), R-KV (redundancy scores), CAKE (entropy+variance), KVCrush (cosine+Hamming similarity), PALU (low-rank KV cache)
- **NOT NOVEL.** Diversity-aware KV selection is a well-populated 2024-2025 research area.

**Passive Sonar Waterfall Sync (transmit only on deviation):**
- This is essentially WHAT TEJAS'S predictor.js + speculative.js ALREADY DO (April 13, 2026)
- Plus in the P3 synthesis: residual-only transport covers this
- **Already in the pipeline, not an addition.**

**Bistatic Ping Speculative Decoding (small pinger + passive listeners):**
- Speculative decoding with small draft model + big verifier = standard
- Distributed variant: see SpecPipe, FlowSpec, PPSD (already prior art for P3)
- **NOT NOVEL.**

**Dark-Scene Pre-rigging (MoE weight prefetch):**
- MoE prefetching is standard (DeepSpeed-MoE, Tutel, ExFlow 2024)
- **NOT NOVEL.**

**Proteolytic Degradation Quantization (age-based precision decay):**
- StreamingLLM (2023), H2O (2023), SnapKV (2024) all do age-weighted KV retention
- Progressive quantization by age: very close to published work
- **NOT NOVEL.**

**Flash-Freeze Background Distillation (system prompt → LoRA):**
- Prompt Distillation (arxiv:2412.14964, Dec 2024) exactly does this
- Text-to-LoRA (2506.06105) generates LoRAs from text
- **NOT NOVEL.**

**Shear-Wall Layer Skippers (cross-layer skip connections):**
- Highway Networks, DenseNet, skip connections — established
- **NOT NOVEL.**

**Base-Isolated Logit Dampers (entropy-triggered routing):**
- Mixture of Experts with entropy-based routing exists
- Speculative decoding pauses on high entropy
- **Adjacent to prior art, low novelty.**

### Claude Round 2 ideas validated

**VLBI-style activation correlation (phase-coherent Byzantine defense):**
- Byzantine voting / majority consensus in distributed systems exists
- Phase-coherence specifically for activation agreement: perhaps novel angle
- **UNCERTAIN — needs deeper validation.**

**Matched filter Byzantine detection:**
- Template matching against known attack patterns — standard in intrusion detection
- Applied to LLM activations specifically: weak novelty claim
- **LIKELY NOT NOVEL.**

**Benchmark-query network calibration (surveyor-style):**
- "Canary queries" are standard in distributed systems monitoring
- **NOT NOVEL.**

**Phase-gated verification at finer granularity (potter):**
- Intra-layer verification is adjacent to SafetyNets (Garg 2015)
- **ADJACENT TO PRIOR ART.**

**Understudy pre-warming for failover (theater):**
- Warm standby / hot-spare is textbook distributed systems
- **NOT NOVEL.**

---

## Surviving ideas across all 2 rounds

| Idea | Source | Validation status |
|---|---|---|
| **Benford's law Byzantine detection on LLM activations** | R1 (both) | NOVEL likely — no prior art on LLM activations |
| **Keystone shard analysis for volunteer networks** | R1 (Claude) | NOVEL likely — no prior art |
| **VLBI activation correlation via phase coherence** | R2 (Claude) | UNCERTAIN — needs deeper lit search |

**Net: still 2 strongly-novel survivors. Added 1 uncertain.**

---

## Honest meta-assessment

Most weird-faculty ideas are either (a) literary analogies without engineering substance or (b) reinvented wheels. The ones that survive share a pattern:
- Genuinely novel APPLICATION of a well-understood method to a domain that hasn't seen it
- Not just new framing of old ideas

**The 2 survivors (Benford + Keystone) are both in the BYZANTINE DETECTION axis.** That's not coincidence — Byzantine detection for LLM inference is genuinely underexplored, while compression/prediction/KV-management are oversaturated.

**Strategic insight: future idea generation should target under-saturated axes:**
- Byzantine / adversarial defense (ripe)
- Trust establishment mechanisms (economic, reputation, cryptographic)
- Incentive design for volunteer participation
- User-facing privacy (differential privacy at activation level?)
- Energy / sustainability / carbon-aware compute
- Cold-start / bootstrapping a new network

These are Synapse-specific challenges that MLPerf and academia haven't heavily covered.

---

## Next round: try UNDER-SATURATED axes

Instead of weird faculties → random ideas, I'll now explicitly target areas where prior art is SPARSE:
- Incentive mechanisms for volunteer compute
- User-side privacy for sensitive queries (medical, legal)
- Cold-start bootstrapping of a new Synapse network
- Carbon-aware compute scheduling
- Client-side KV cache privacy

---

## Invariant: novel survival rate ≈ 5%

Of ~50 raw ideas across 2 rounds of weird-faculty brainstorming, ~2-3 survived. Future rounds should optimize for this: generate FEWER but more TARGETED ideas, spending less on validation per idea.
