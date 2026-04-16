# Round 3 Synthesis + Final Survivor List

**Date:** 2026-04-16
**Inputs:** `under_saturated_axes_round3_claude.md`, `gemini_responses/under_saturated_axes_round3.md`
**Mode:** Autonomous idea generation + validation

---

## Round 3 validation

### KILLED by prior art (most ideas)

| Idea | Source | Killed by |
|---|---|---|
| Edge-Sandwich Sharding | Gemini R3 | *Split Learning for LLMs* (2501.05965 2025), *Privacy-Aware Split Inference* (2602.16760), Fission (eprint 2025/653) |
| Coded Distributed Inference | Gemini R3 | IBM COIN (ISIT 2024), Lagrange Coded Computing (2019), Private Coded Matmul (2019) |
| Activation Chaffing | Gemini R3 | Equivalent to DP forward passes (well-published) |
| Follow-the-Sun Steering | Gemini R3 | Google's Carbon Intelligent Computing, general carbon-aware ML scheduling |
| Reciprocal Priority Queuing | Gemini R3 | BitTorrent tit-for-tat textbook |
| Prompt Splintering | Claude R3 | Fission (multi-party LLM inference), split inference papers |
| Homomorphic Activation Obfuscation | Claude R3 | Adjacent to full homomorphic encryption — heavy prior art |
| Shard-Hot-Migration | Claude R3 | KV cache snapshot + warm standby is standard |

### SURVIVORS (narrow novelty, require deeper validation)

1. **reCAPTCHA Gambit** (Gemini R3) — Bloggers embed a Synapse widget; their visitors' browsers silently process Synapse inference tasks while reading the page. The webmaster gets free AI; Synapse gets viral compute.
   - Prior art: CoinHive-style crypto mining (deprecated, privacy-hostile). For AI compute, specifically consent-based and bounded — likely novel application.
   - **Worth exploring** as a paper on "opt-in ambient compute."

2. **Parachute Local Fallback** (Gemini R3) — Every client has a tiny local model cached. Mid-query network failure hands KV state to local model; user gets slightly degraded output but no crash.
   - Prior art: Edge-cloud split computing has fallback modes. The specific "hand off mid-generation KV cache to local tiny model" is nuanced.
   - **Worth exploring** for graceful degradation paper.

3. **Progressive Decentralization Protocol with Readiness Score** (Claude R3) — Formal framework for gradual transition from centralized to decentralized compute, with a measurable "decentralization readiness" score.
   - Prior art: general federated learning transitions exist, but formal readiness score is not standard.
   - **Worth deeper lit search.**

4. **Attribution-Based Karma with user feedback** (Claude R3) — If a volunteer's compute contributed to an upvoted answer, volunteer earns karma. Public leaderboard. Social incentive.
   - Prior art: reddit karma, SOverflow reputation. Applied to LLM compute — narrow novelty.
   - **Worth including as part of a mechanism-design paper for decentralized ML incentives.**

---

## Final survivor list across ALL 3 rounds (validated)

| Idea | Source | Status | Potential placement |
|---|---|---|---|
| **Benford's law Byzantine detection on LLM activations** | R1 | NOVEL likely | P4 flagship — Byzantine defense section |
| **Keystone shard analysis for volunteer networks** | R1 | NOVEL likely | P4 flagship — Criticality-aware allocation |
| **reCAPTCHA Gambit (widget-embedded compute)** | R3 Gemini | Narrow novel, new application | Design paper / blog / Synapse deployment strategy |
| **Parachute Local Fallback** | R3 Gemini | Narrow novel | Operational / reliability paper |
| **Progressive Decentralization Protocol** | R3 Claude | Uncertain, deeper check needed | Systems paper |
| **Attribution-Based Karma** | R3 Claude | Narrow novel | Mechanism-design paper |

**Total: 6 narrow-to-strong novel ideas from ~60 raw ideas. ~10% aggregate novelty rate after validation.**

---

## Strategic conclusions from 3 rounds of autonomous ideation

1. **Most "novel" ideas are reinventions.** The cross-verifier is essential; without it, we'd have claimed novelty on maybe 30% of our ideas, all of which would have been destroyed in peer review.

2. **Targeting under-saturated axes helps but ceiling is still low.** Even there, the obvious angles are published. The novel gaps are narrow and specific.

3. **Multi-faculty creativity scales sub-linearly.** 10 faculties didn't give 10x the novel ideas compared to 5. Convergence dominates.

4. **The Byzantine/trust axis is the richest vein.** Almost all final survivors relate to trust, verification, or decentralized incentive design. Compression / prediction / KV-cache are picked over.

5. **Deep engineering domains (VLBI, submarine warfare) generate MORE SPECIFIC but MORE-LIKELY-PUBLISHED ideas.** Breadth of weird-faculty analogies doesn't overcome the fact that signal processing / distributed systems have already industrialized the best tricks.

6. **Synapse's true novelty is OPERATIONAL, not ALGORITHMIC.** The algorithms (speculative decoding, compression, sharding) all have prior art. What Synapse UNIQUELY owns:
   - Volunteer WebGPU browsers as compute substrate
   - Heterogeneous consumer hardware (iPhone + Android + Nvidia + Intel coexisting)
   - Cross-platform numerical fidelity measurement (P2 — real contribution)
   - The specific deployment design (coordinator + volunteers + public good ethos)

**Strategic recommendation: Synapse papers should lead with OPERATIONAL contributions (real measurements on real consumer hardware) and treat algorithmic contributions as incremental. This plays to our strengths and the gap in the literature.**

---

## What to do with the 6 survivors

1. **Benford's law + Keystone shards** → add as sections of P4 flagship (Byzantine defense in depth)
2. **reCAPTCHA Gambit + Attribution Karma** → combine into a "deployment strategy" design note, committed to repo as an invention claim
3. **Parachute Local Fallback** → add as a reliability measurement to P2 or P3
4. **Progressive Decentralization Protocol** → needs more literature work before claim

---

## Ending autonomous session

Three rounds done. ~60 ideas generated, 6 survive preliminary validation. All committed to repo with explicit timestamps and prior-art citations.

Next useful autonomous actions:
- Design an empirical test for Benford's law on Gemma/Llama activations (cheap, 20 min on existing multimodel data)
- Write up the Keystone shard metric formally
- Timestamp the reCAPTCHA-Gambit idea as an invention claim
- Monitor long-context experiments on RunPod (still running)

Doing these now.
