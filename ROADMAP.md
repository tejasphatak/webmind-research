# Webmind Research Roadmap — 18-month plan

**Author:** Tejas Phatak
**Locked:** 2026-04-16
**Last updated:** 2026-04-16
**Strategic context:** EB1-A/NIW portfolio building, CU Boulder MS, H1B extension Feb 2027, priority date Aug 23, 2017 (India EB2 → port to EB1 on approval).

---

## The 4-paper plan

### P1 — Carrier-Payload: Activation Compression for Decentralized Inference
- **Status (2026-04-16):** Experiment complete on Gemma 3 1B (22x compression at KL=0.023). Multi-model validation running tonight (Llama 3.1 8B, Gemma 4 27B, Qwen 2.5 32B). Paper draft written. Social posts drafted. Email templates ready.
- **Timeline:** Ship to arXiv April 2026. Workshop submission (NeurIPS ENLSP or MLSys workshop) by Aug 2026.
- **Venue target:** MLSys 2027 main track OR ICML/NeurIPS workshop 2026.
- **Effort remaining:** ~2 weeks (polish, arXiv submit, social rollout).

### P2 — Portable ML on WebGPU: Numerical Fidelity Across Consumer Hardware
- **Status:** Data already exists in `Synapse` repo (Pixel 10 Pro, iPhone 16, Qualcomm Android, Nvidia desktop, Intel iGPU failure). Tested GPT-2 parity across architectures. 2026-04-15 benchmark shows 2.34x speedup on heterogeneous vs all-mobile.
- **Timeline:** Write up June-July 2026. Submit to workshop by Sept 2026.
- **Venue target:** MLSys 2027 workshop OR IEEE Micro short paper.
- **Effort remaining:** 6-8 weeks (formalize data, add device classes if possible, write).
- **Low-effort, high-yield.** Data is already collected.

### P3 — Distributed Speculative Decoding for Decentralized Inference
- **Status:** Idea stage. Listed as "Phase 2" in Synapse roadmap. Infra partially exists.
- **Timeline:** Sept-Dec 2026 research + experiments. Submit early 2027.
- **Venue target:** MLSys 2027 main track OR NSDI 2027 OR OSDI 2027.
- **Novelty:** High. Almost no published work on distributed speculative decoding.
- **Expected impact:** 2-4x tok/s speedup on volunteer networks.
- **Effort:** 3-4 months deep work, needs GPU budget for experiments.

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
