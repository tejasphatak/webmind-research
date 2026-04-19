---
name: Verification-First Discipline
description: Every invention/claim ships with a pre-registered verification plan. No claim without measurement. No paper without stats. Prioritize verification over implementation.
type: feedback
originSessionId: bf610133-862e-4762-8a34-7449b7726a9e
---
**Directive (2026-04-14):** "Think of verification as well based on the outcome. Prioritize it!"

## The rule

Before writing the implementation of any inventive claim (SFCA, Cognitron, AGP, Distillation, anything), write the verification plan *first*. The verification plan is the contract with reality.

## Template (F-SCI format)

Every new invention-memory (`project_X_invention.md`) gets a **Verification** section containing:

1. **Hypothesis** — stated before data exists; binary outcome
2. **Primary metric** — single number, measurable, decided in advance
3. **Sample size + power calculation** — how many beats / events before we know
4. **Control condition** — what we compare against; default = current behavior
5. **Stopping criterion** — point at which we stop collecting and decide
6. **Pre-committed analysis** — stats method chosen up front, not "we'll see"
7. **Null hypothesis** — if we see X, we accept the invention doesn't help

## Instrumentation (F-SRE)

- Metrics logged from day 1 (not retrofitted)
- Each beat emits the measurement fields required by active experiments
- Weekly summary committed to VM file for review
- Dashboard panel on `chat.webmind.sh` once Worker/Worker email is stable

## Adversarial checks (F-RED)

Before declaring a win, Advisor runs through:
- **Confounders** — what else changed in the same period?
- **Selection bias** — did we cherry-pick which beats counted?
- **Data leakage** — did the training data sneak into the test data?
- **Power issues** — small samples can spuriously look significant OR invisibly fail
- **Publication bias** — a null result is a real result; publish both

## Applied to inventions in the pipeline

Every one of [P-SFC01, P-CGN01, P-AGP01, P-DST01] must add a Verification section with the 7-item template before implementation starts. Already drafted for SFCA (C-160..C-164).

## Reminder

If I catch myself writing a method without a measurement, I stop. A method I can't measure is a story, not a contribution.
