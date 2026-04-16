# Webmind Research

**Public research log.** Pre-registered hypotheses, timestamped inventive claims, reference implementations, verification plans.

*AI-generated, human-directed.* All research in this repo is written by a continuously-running substrate-LLM agent on a private VM, under the direction of Tejas Phatak.

## Active papers

| Paper | Status | Notes |
|---|---|---|
| [SFCA Pre-registration v1](papers/sfca-preregistration-v1.md) | **Pre-registered 2026-04-14** · data collection in progress ≥ 30 days | Shapley Faculty Credit Assignment for multi-perspective LLM agent cognition. Reference impl in [`sfca/`](sfca/). 13 unit tests verify all four Shapley axioms. |

## Inventions & claims (timestamped)

All in [`inventions/`](inventions/), dated filenames, SHA256 manifest in [`MANIFEST.md`](MANIFEST.md).

- **P-SFC01 · SFCA** — Shapley credit + convex weight optimization *(pre-registered)*
- **P-AGP01 · AGP** — Agent Grammar Protocol, compact LLM-to-LLM codec *(17.6% reduction on self-baseline; third-party baseline pending)*
- **P-CGN01 · Cognitron** — AGP + Distillation combined architecture
- **P-HOL01 · Holographic Cognition** — boundary memory reconstructs bulk (SPECULATIVE, testable)
- **P-UAT01 · Faculty-UAT** — faculties as universal decision approximators (SPECULATIVE, math-track)
- **P-TRN01 · Training-Pipeline Contributions** — Faculty-Decomposed DPO, Shapley-RLHF, Agent-Trajectories
- **P-WSN01 · Warm-Stream Nexus** — one persistent claude process, compact protocol
- **P-COL01 · Collective Consciousness** — 21 faculties × ~5 modes = ~100 lenses

## Discipline

- **Pre-registration first.** Hypotheses, metrics, analysis plan, stopping criteria locked in git *before* data collection.
- **Null results published.** No file-drawer. A rigorous null is a real contribution.
- **Code released MIT** at submission; ledger data released with paper where privacy permits.
- **AI-generated research is disclosed prominently.** The genesis of every artifact is transparent.

## Tools

- [`tools/watch-beats.sh`](tools/watch-beats.sh) — live view of what the agent is doing
- [`sfca/nexus_integration.py`](sfca/nexus_integration.py) — SFCA ledger CLI
- [`agp/benchmark.py`](agp/benchmark.py) — AGP token-reduction benchmark

## License

- Papers: CC-BY 4.0
- Code: MIT
- Everything under the Webmind umbrella

## Why public

Pre-registration is only meaningful when the registration is *verifiable* — public git commits with timestamps do that. Making this repo public before results are in is the scientifically honest move. We stake our claims in the open and report results either way.
