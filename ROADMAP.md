# Webmind Research Roadmap — 18-month plan

**Author:** Tejas Phatak
**Locked:** 2026-04-16
**Last updated:** 2026-04-16
**Strategic context:** EB1-A/NIW portfolio building, CU Boulder MS, H1B extension Feb 2027, priority date Aug 23, 2017 (India EB2 → port to EB1 on approval).

---

## The 4-paper plan

### P1 — Carrier-Payload: Training-Free Activation Compression for Text-Only Decentralized Inference
- **Status (2026-04-16 late evening):** All 9 internal submission gates GREEN. Gate 10 (arXiv endorsement) is the only remaining item, pending the Dettmers response email send.
  - Clean text-only rewrite at `papers/carrier-payload-text-only-v1.md`.
  - **16/16** citations arXiv-verified (added AWQ, BottleNet++, StreamingLLM per Gemini Deep Research review 2026-04-16).
  - LaTeX 0 errors, Links 16/16 resolve, **40/40** paper invariants pass (rescoped from 37 to cover long-context Qwen §3.4–§3.6).
  - Short-context data: N=3 text families (Gemma 3 1B, Llama 8B, Qwen 32B), all 22–24× at short context; rank-bound-artifact disclosed with practical-regime defense.
  - Long-context data: Qwen 32B at seq 256/512/1024/1621, CR 183×→13× degradation documented honestly; §5.3 reframed from "law" to "short-context empirical heuristic."
  - External novelty validated 2026-04-16 by Tim Dettmers (CMU, LLM.int8! / QLoRA / bitsandbytes author): "not aware of a deep analysis of how activation compression can be tweaked to be best for distributed inference on regular consumer computers + internet connections."
  - Gemini 3.1 Pro external review 2026-04-16 verdict: **READY FOR arXiv (with minor polish)** — 100% confidence; findings at `findings/2026-04-16-deep-research-review-carrier-payload.md`.
- **arXiv primary category:** `cs.DC` (distributed computing). Secondaries: `cs.LG`, `cs.CL`.
- **Target venues post-arXiv:** COLM, ACL, or EMNLP main conference (not MLSys — pivoted per Gemini scope analysis 2026-04-16).
- **Effort remaining:** Tejas sends the Dettmers email → wait for endorsement → arXiv upload. Zero additional code or writing required on the Nexus/Atlas side.

### P1' (OLD superseded) — `papers/carrier-payload-v1.md`
- Original broad draft with multimodal claims. Now superseded by P1 text-only. Retained with prominent revision notice at top because Tim Dettmers and CU Boulder professors received URLs to it via email. Not for submission.

### P2 — Portable ML on WebGPU: Numerical Fidelity Across Consumer Hardware
- **Status:** Data already exists in `Synapse` repo (Pixel 10 Pro, iPhone 16, Qualcomm Android, Nvidia desktop, Intel iGPU failure). Tested GPT-2 parity across architectures. 2026-04-15 benchmark shows 2.34x speedup on heterogeneous vs all-mobile.
- **Timeline:** Write up June-July 2026. Submit to workshop by Sept 2026.
- **Venue target:** MLSys 2027 workshop OR IEEE Micro short paper.
- **Effort remaining:** 6-8 weeks (formalize data, add device classes if possible, write).
- **Low-effort, high-yield.** Data is already collected.

### P2 — The Modality Gap (NEW from today's pivot)
- **Status:** Placeholder at `papers/modality-gap-v2-placeholder.md`. Preliminary Gemma 4 31B data (N=1 multimodal) shows inverse compression scaling vs text. Qwen-VL confounder check failed 3x on RunPod infra.
- **Timeline:** Q3 2026. Requires N=3 paired text/vision-language families (e.g., Qwen 2.5 vs Qwen-VL, Llama 3.1 vs Llama 3.2-Vision, Mistral vs Pixtral).
- **Venue target:** NeurIPS or ICLR.
- **Dependency:** builds on and cites P1 (text-only). Coherent research-program narrative for EB1-A.

### P3 — Predictive Residual Transport (was "Distributed Speculative Decoding")
- **Status:** Scope pivoted after SpecPipe/FlowSpec/PPSD/EAGLE prior-art found 2026-04-16. Original "Distributed Speculative Decoding" is prior art. New scope: residual-only transport via shared online predictor + carrier-payload on residuals (novel synthesis).
- **Timeline:** Sept 2026 – Feb 2027 research + experiments. Submit early 2027.
- **Venue target:** MLSys 2027 workshop or NSDI.
- **Novelty (narrow but defensible):** first residual-only transport protocol combining prediction + carrier-payload compression + Byzantine attestation for decentralized LLM inference.
- **Effort:** 3-4 months (predictor.js + speculative.js already exist in Synapse from Apr 13, 2026).

### P4 — Trust, Compress, Verify: A Complete Protocol for Decentralized LLM Inference (FLAGSHIP)
- **Status:** Depends on P1 + P3 complete + Byzantine verification (Artifact 2 from earlier research mission).
- **Timeline:** Jan-Mar 2027 synthesis.
- **Venue target:** MLSys 2027 main track OR SOSP 2027.
- **Concept:** Single end-to-end paper combining:
  - Carrier-Payload compression (from P1)
  - Adaptive precision (Synapse Phase 4 int4 toggle)
  - Freivalds-based probabilistic Byzantine verification
- **Why it matters:** Only paper in the literature that addresses compression + adaptation + trust as one protocol.

---

## Supporting activities (also count for EB1-A)

- **Peer review:** after P1 lands, volunteer for workshops. Target 4-5 reviews/year. Qualifies EB1-A criterion 4.
- **Provisional patent:** File on carrier-payload method. Q2 2026. Full utility filing within 12 months if papers gain traction.
- **Conference presentations:** Attend MLSys/NeurIPS 2026-2027. Present if accepted.
- **Media coverage:** After arXiv on each paper, pitch TechCrunch / VentureBeat / podcasts. Indian grad student + distributed AI = strong narrative.
- **Reference letters:** Build relationships with Tim Dettmers (UW), Petals team, Gemma team, Long Chen (CU Boulder) or similar. 6-12 months of relationship before asking for EB1-A letter.

---

## Kill list (do NOT develop into papers)

These are internal R&D notes, not scientific contributions. Stay as engineering artifacts on webmind.sh, not arXiv.

- AGP / Cognitron / Warm Stream Nexus — engineering design, not science
- Holographic Cognition / Faculty-UAT — speculative conjectures, no data
- SFCA — only pursue if synthetic benchmark (running now) shows clear signal over EMA+CF. If marginal, ship as workshop poster only.
- MoEfication v3 — existing published literature is strong. Skip unless clear novelty angle emerges.

---

## Pace checks (hard rules)

1. If P1 gets workshop reject + bad reviews: revise P1 before launching P2.
2. If H1B extension filing in Sept 2026 gets stressful: PAUSE research.
3. If MS coursework heats up: PAUSE research. School first.
4. If any paper's experimental numbers weaken on deeper testing: DO NOT SHIP. Better no paper than bad paper.
5. 3 strong papers > 4 mediocre papers.

---

## Compute budget

- **RunPod**: ~$50-100/month during active experiment phases. On-demand pods for specific experiments.
- **GCP**: local VM for development, CPU-only. Used for drafting, small sims.
- **CU Boulder HPC (Alpine/Blanca)**: free GPU access via student account. Use when possible.
- **Hugging Face**: gated model tokens in place.

---

## EB1-A portfolio target (by Q1 2028)

- 4 papers (P1-P4), all published
- 50-200 citations (if lucky)
- 4-5 peer reviews done
- 1-2 provisional patents
- 3-5 conference presentations
- Media coverage on at least 1 paper
- 6 independent expert reference letters (18+ month relationships)
- CU Boulder MS completed
- Aug 23, 2017 priority date ported to EB1 via approved I-140

**If the above is in hand by Q1 2028: file EB1-A + concurrent I-485 (PD likely current for India EB1 at 2017). GC expected 2028-2029.**

---

## Immediate next actions (April 2026)

1. Finish P1 multi-model experiment (running tonight)
2. Polish P1 paper, incorporate multi-model results
3. Submit P1 to arXiv (need endorser — request from CU Boulder professor)
4. Ship social rollout 48 hrs after arXiv goes live
5. Begin P2 write-up drafting (data already collected in Synapse repo)
6. Schedule immigration attorney consult
7. File provisional patent on carrier-payload

---

## Immigration parallel track

- **Before Sept 2026 (H1B extension filing):**
  - Get certified DWI dismissal court records
  - FBI background self-check
  - Attorney consult (Chen Immigration / WorkPermit / Denver firm)
  - Copy of both I-140 approval notices (TechM 2017 + Mastercard 2023)
- **Late 2027 (EB1-A filing):**
  - 3-4 papers in hand
  - Portfolio assembled
  - Expert letters collected
  - PD port to EB1 if current

---

## Authorship convention (every paper)

- **Author:** Tejas Phatak, University of Colorado Boulder & Webmind Research
- **Contact:** tejasphatak@gmail.com
- **AI contribution disclosure:** "Research executed by Claude Opus 4.6 (Anthropic) with review by Gemini 3.1 Pro (Google). All experiments, analysis, and writing were AI-generated under human direction."
- **License:** CC-BY 4.0 (papers), MIT (code).

---

*This roadmap is a living document. Review quarterly. Revise on major signal (paper acceptance, citation counts, H1B/immigration milestones).*
