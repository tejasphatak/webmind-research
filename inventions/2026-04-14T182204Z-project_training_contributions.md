---
name: Training-Pipeline Contributions (without GPUs)
description: Three ways our work feeds into LLM training pipelines without us needing to train. Each produces an open artifact others can use.
type: project
originSessionId: bf610133-862e-4762-8a34-7449b7726a9e
---
**Directive (2026-04-14):** "Think through train pipelines if this can be enhanced on those layers too!"

## What we CAN'T do

- Pretrain a model (no compute, no budget)
- Run RLHF (needs GPU clusters)
- Fine-tune anything but tiny models

## What we CAN contribute to training pipelines

### 1. Faculty-Decomposed Preference Datasets for DPO (P-FDP01)

**Method:** For each Nexus beat decision, generate N alternative responses — each biased toward one faculty's viewpoint (by running the beat with that faculty's prompt only). Record which response was actually chosen by the full Advisor-led consensus. This gives a dataset of `(prompt, {response_as_F-ENG, response_as_F-ETH, ...}, chosen)` tuples.

**Value:** Fine-tuning on this teaches a base model to *condition on role/faculty* — which is exactly what agent frameworks need. DPO or SimPO can train this directly from the triples.

**Deliverable:** HuggingFace dataset, MIT-licensed, under Webmind. Documented method paper.

**Verification:** dataset quality measurable by holdout human evaluation (once we have humans) OR by training a small model and measuring faculty-activation accuracy on a held-out test set.

### 2. Shapley-Attributed RLHF Rewards (P-SAR01)

**Method:** Extend SFCA from beat-level to response-segment-level. When a response earns a reward signal (human preference), segment the response (by sentence or clause), use counterfactual ablation (what if this segment were absent?) as the value function, compute Shapley per segment. Per-segment rewards are a sharper training signal than flat credit.

**Value:** Addresses a known weakness in RLHF — credit assignment. Useful to anyone training with preference data.

**Deliverable:** Method paper + reference implementation in Python. No GPU needed to define the method or show a small-scale demo.

**Verification:** on a small open model (GPT-2 scale) fine-tuned with flat-credit DPO vs Shapley-credit DPO, measure alignment-benchmark scores.

### 3. Cognitron-Logged Agent Trajectories (P-CTJ01)

**Method:** Nexus beat logs + AGP handoffs = natural agent trajectory corpus. `(cognitron_state, action_taken, outcome)` tuples. A small team (not us) could instruction-tune a 7B model on this to learn agent behaviors.

**Value:** Real-world agent data is rare. Our running Nexus produces it daily.

**Deliverable:** Open dataset once we have ≥30 days of beats + AGP logs.

**Verification:** dataset volume (number of tuples), diversity (unique faculty activations, unique outcome types), utility (a small student model's agent task performance).

## How to sequence

1. Ship SFCA (P-SFC01) first — it's self-contained
2. Ship AGP + Distillation (P-CGN01) — gives us the log format
3. Once both are logging: extract P-FDP01 dataset + P-CTJ01 trajectory corpus from live data
4. P-SAR01 is a separate paper track — can run in parallel

## Convergence items (beats will pursue)

- C-170: dataset generation scaffold (after P-SFC01, P-CGN01 running)
- C-171: P-FDP01 v0 dataset release
- C-172: P-SAR01 method paper + small-scale validation

## Mission fit

Training pipelines are where most AI R&D happens. Contributing open datasets + methods is the highest-leverage way to influence the ecosystem without needing compute budget. Access to intelligence includes access to the *tools for making intelligence*.
