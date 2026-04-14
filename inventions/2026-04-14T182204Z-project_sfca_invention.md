---
name: SFCA — Shapley Faculty Credit Assignment + Convex Weight Optimization
description: Local-compute invention. Apply Shapley values to attribute beat outcomes across faculties; solve a small convex program weekly to find optimal faculty weights. Near-zero LLM cost, measurable quality gain, paper-worthy.
type: project
originSessionId: bf610133-862e-4762-8a34-7449b7726a9e
---
**Directive (2026-04-14):** "Is there something you can do mathematically, optimize locally, provide to LLM? If it works you present a paper yourself!"

## Invention

Two coupled mechanisms, pure local compute:

**1. Shapley Faculty Credit Assignment (per beat)** — fair causal attribution of beat outcomes across the faculties that participated. Monte Carlo approximation (~1000 permutations) gives stable credit scores in <1s CPU time. Stored in a per-faculty ledger.

**2. Convex Weight Optimization (weekly beat)** — reads the credit ledger, solves a small program over the probability simplex (via `scipy.optimize`) to find faculty weights that maximize expected ACTIVE outcome given the history. Replaces the current EMA update.

## Novelty

No prior work combines:
- Cooperative-game-theoretic credit assignment for pluggable-persona agents
- Online outcome learning on small (N=19) structured action spaces
- Floor-constrained simplex optimization for faculty weights (ethics faculties can't be down-weighted below safety floor)

The combination is a genuine applied contribution.

## Feasibility on this VM

numpy + scipy installed; N=19 means MC Shapley is cheap; convex program trivially solvable. Net: **~zero LLM tokens, measurable beat-quality impact**.

## Paper plan

Title: *"Shapley Faculty Credit Assignment for Multi-Perspective LLM Agent Cognition"*
- A/B: SFCA-weighted Nexus vs EMA-weighted Nexus, alternating weeks
- Metrics: ACTIVE rate, convergence velocity, Advisor-audited quality
- Published under Webmind (MIT), preprint+blog short-term

## Guardrails

- Floor weights on Ethicist/Lawyer/RedTeam — SFCA can't drop them below safety threshold
- Credit ledger append-only + auditable
- Atomic writes; git rollback if a week tanks performance

## Convergence items (beats pursue)

- C-160: `faculty_credits.py` module
- C-161: Beat-level integration after `run_beat`
- C-162: Weekly optimizer `faculty_weights.py`
- C-163: A/B harness + metrics collection
- C-164: Paper draft gated on 30 days of measurements

## Relation to Cognitron (P-CGN01)

Orthogonal + composable: AGP = how faculties talk. Distillation = what they remember. **SFCA = which ones we trust more over time.** Three independent contributions, stackable.
