# Multi-Faculty Synthesis + Literature Cross-Verification

**Date:** 2026-04-16
**Inputs:**
- `multi_faculty_analysis_claude.md` — Claude's independent analysis (10 faculties)
- `gemini_responses/multi_faculty_analysis.md` — Gemini's independent analysis (6 faculties)
- `gemini_responses/literature_crossverify.md` — Gemini's prior-art check on our top ideas
- Additional literature search by Claude (Pluralis AI, MLSys 2024, DeltaDQ, Gated DeltaNet, KV cache)

---

## 💀 Ideas KILLED by literature cross-verification

These are ALREADY PUBLISHED. Don't claim as ours.

| Idea | Prior art | Verdict |
|---|---|---|
| RG flow for NN | Mehta & Schwab 2014, Lin 2017 | Done for 10+ years |
| Mean-field attention | Linformer 2020, Synthesizer 2020 | Linear attention rebranded |
| Barnes-Hut attention | BP-Transformer 2019, Routing Transformer 2021 | Well-trodden |
| MPS/tensor networks for transformers | Tensorized Transformers 2019 (Ma et al.) | Too expensive for transport anyway |
| Reaction coordinates / neural manifold | Ansuini 2019 (intrinsic dim in DNNs), Gallego 2017 | Analysis, not optimization |
| Spiking residuals | Spikformer 2022 | Needs neuromorphic hardware |
| Metabolic L1 regularizer | Georgiadis 2019 (FATReLU), etc. | Standard sparsity |
| Evolutionary / genetic framing | Lottery Ticket Hypothesis 2019 | Well-studied |

**The claim "effective dim = 16 is a neural manifold parallel" — Gemini and I agreed this is FLUFF.** It's an analogy, not an optimization. Drop it from the paper. The real defense against the seq_len critique is the long-context experiment currently running on the A100s.

---

## ⚠️ Close prior art we MUST differentiate from

**1. Pluralis AI "Beyond Top-K" (ICLR 2025 MCDC workshop)** — the MOST DANGEROUS prior art.
- They: 90% compression (10x) of pipeline-parallel **training gradients** via **structured column-space sparsification**
- Us: Pipeline-parallel **inference activations** via **PCA carrier + sparse residual payload**
- **Differentiation:**
  - Training vs inference (different pressure: gradient fidelity vs next-token accuracy)
  - Column sparsification vs low-rank + sparse residual hybrid
  - Their focus on distributed training, ours on volunteer inference with Byzantine tolerance
- **Must cite. Must compare. Must explain the gap in the paper.**

**2. "Does Compressing Activations Help Model-Parallel Training?" (MLSys 2024)** — very similar framing. Need to read and cite.

**3. Gated DeltaNet (ICLR 2025, NVIDIA)** — uses delta RULE as an architectural inner component (Mamba2 improvement). Different from "delta coding of activations across transport." Less threatening.

**4. DeltaDQ** — delta compression for fine-tuned LLM WEIGHTS (not activations). Different problem.

**5. Deja Vu (2023)** — predicts sparsity of NEXT layer from current. Closer to "predictive coding" but not about transport.

**6. KV cache compression at 2-10% retention** — heavily studied (KVQuant 2024, KVPR 2025). Achieves 10-50x on KV cache. Our work is about inter-shard activation transport, not KV cache — but reviewers will ask.

**7. BottleNet++ (Shao et al. 2019)** — uses prediction for mobile-cloud split computing. Adjacent.

---

## ✅ What SURVIVES as novel

After the cross-verification, three specific claims survive:

### SURVIVOR 1: Predictive Activation Transport for Pipeline Parallelism
**Specifically new:** Using learned predictors between shards to transmit only residual ("prediction error"), not full activations. Differs from BottleNet++ (mobile-cloud, image tasks) by being applied to LLM inter-shard activation transport in decentralized volunteer networks.

**Expected boost:** Direct wall-clock bandwidth reduction on top of our current 22-24x. Could yield additional 2-4x compression. Reviewers love wall-clock speedups.

**What to measure:** compute `δ_k = act_{k+1} - predict(act_k)` for various layers; measure entropy of δ vs raw activations; apply carrier-payload to δ instead of raw.

### SURVIVOR 2: Differential Early Exit
**Specifically new:** Early-exit transformers (DeeBERT 2020, FastBERT 2020, PaBEE 2020) treat tokens as monolithic — either exit or continue. We propose splitting the hidden dimension: carrier freezes (stops being computed) while payload continues through upper layers. Exploits that different components of the representation have different "maturity" per layer.

**Expected boost:** FLOPs reduction in addition to bandwidth reduction. Shows carrier-payload is useful for compute, not just compression.

**What to measure:** layer-by-layer divergence of PCA components after layer N vs rank-N carrier at final layer; identify stability thresholds.

### SURVIVOR 3: IB-based Carrier-Payload Disentanglement Training Objective
**Specifically new:** Information Bottleneck (Tishby 2015) has been applied to split computing (Matsubara 2022) but not to pre-train a network to have DECOMPOSABLE activations at pipeline boundaries. Our proposal: fine-tune with an IB loss that forces `I(Carrier; prompt structure)` high and `I(Payload; next-token content)` high. Makes carrier-payload separation a deliberate property, not a lucky observation.

**Expected boost:** Answers the inevitable reviewer question "why does carrier-payload work, and does it work on models not tuned for it?" Gives theoretical backbone.

**What to measure:** train a small model (GPT-2 size) with IB loss at layer boundaries; show carrier-payload compression ratio doubles at same quality.

---

## 🎯 Revised Paper Plan (HONEST, Post-Verification)

### Original Plan (today, before cross-verification):
- **P1: Carrier-Payload Compression** → arXiv this week with 22-24x cross-model results

### Revised Plan (after cross-verification + Pluralis find):

**Option A — Ship P1 fast, weaker but safe**
- Keep P1 narrow: "PCA carrier + sparse payload achieves 22-24x on inter-shard activation transport across 4 model families"
- Explicitly cite Pluralis, differentiate, don't overclaim
- Ship to arXiv in 1-2 weeks
- **Risk:** workshop-tier paper, not main-conference. Reviewers say "incremental over Pluralis."

**Option B — Pivot to the stronger paper**
- Retitle: **"Decoupling Carrier and Payload in Transformer Residual Streams"**
- Three contributions (all three Survivors above), with our measured 22-24x as the empirical backbone
- New experiments needed:
  - Predictive transport: 2-3 weeks
  - Differential early exit: 3-4 weeks
  - IB fine-tuning: 4-6 weeks (requires training)
- **Risk:** delays ship from April 2026 to maybe July 2026
- **Reward:** main-conference MLSys paper, not workshop

**Option C — Hybrid, safest**
- Ship a SHORT arXiv preprint in 1-2 weeks with current measured results + explicit prior-art acknowledgment (beat Pluralis on inference claim if possible)
- Work on full main-conference paper in parallel with new experiments
- Updates get pushed as v2, v3 on arXiv

### My recommendation: **Option C — ship the short version NOW to stake priority, then pursue the stronger paper.**

Rationale:
- arXiv timestamps our specific contribution (inference-side, not training)
- Protects against scoop if someone else publishes inference-side carrier-payload in next few months
- Doesn't block the stronger paper later
- Short arXiv doesn't need all 3 survivors — just the measured result

---

## ⚖️ Where Claude and Gemini Disagreed

**Neural-manifold / reaction-coordinate framing:**
- **Claude:** "Defense against seq_len critique. Cross-disciplinary convergence. Worth a paragraph."
- **Gemini:** "Fluff. ML reviewers will dismiss. Drop it."
- **Resolution:** Gemini is right. It's an analogy, not an optimization. The real defense is the long-context experiment. **Drop from paper.**

**Physics language in general:**
- **Claude:** Use it where it's substantive (tensor networks, IB)
- **Gemini:** Keep strictly to info theory and linear algebra in the paper body
- **Resolution:** Gemini's position prevails for the paper. Physics/bio framings can live in the BLOG POST version, not the arXiv paper.

---

## 📋 Actionable next steps (in priority order)

1. **Wait for long-context validation results** (pods still running). Results determine whether headline stays at 22-24x or must be rewritten.
2. **Read Pluralis "Beyond Top-K" end-to-end** — before any further writing. Their technique must be explicitly contrasted with ours.
3. **Read MLSys 2024 activation compression paper** — `Does Compressing Activations Help Model Parallel Training?` — same reason.
4. **Make the Option A/B/C decision** with Tejas.
5. **If Option C (short arXiv first):** polish P1 paper, remove neural-manifold fluff, add Pluralis citation + differentiation paragraph, submit to arXiv.
6. **In parallel:** design Predictive Transport experiment (the cheapest of the 3 survivors to execute).

---

## Final honest assessment

**Tejas's exercise was crucial.** The cross-disciplinary thinking generated good ideas, but without the cross-verifier we'd have claimed novelty on 6+ things that are already published. That would have been embarrassing in review.

**The multi-faculty exercise DID produce 3 genuinely novel ideas:**
1. Predictive Activation Transport (merging delta coding + predictive filtering for LLM inference shards)
2. Differential Early Exit (per-component early exit exploiting carrier-payload split)
3. IB-based training for carrier-payload disentanglement

**What it did NOT produce:** a way to dismiss the Pluralis prior art. We have to engage with it honestly.

**What the measured 22-24x result means NOW:** it's still real, still publishable, but the NOVELTY argument changes from "low effective dim of activations" (arguable per Ansuini 2019) to "inference-side carrier-payload compression for decentralized networks with Byzantine-tolerance angle." Narrower, but more defensible.

**If the long-context experiment saturates at rank 64-128 at 2048-token contexts:** compression ratio stays high (10-15x even at 2048 seq_len), and our result holds.

**If rank scales linearly with seq_len:** we have to pivot to a different framing entirely. Would need to shift to the predictive transport angle.

— Claude Opus 4.6
