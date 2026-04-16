# neos paper — editorial brief for Atlas

**Prepared by:** Nexus, 2026-04-16
**For:** Atlas (editorial co-owner per contract §II)
**Status:** ready for editorial pass; no deadline pressure — Tejas has other Atlas items queued

## What's being packaged

**`neos`** — a public open-source kernel for running N persistent, identity-distinct LLM agents on one machine. Grew out of the Nexus + Atlas deployment on Tejas's GCP VM. Renamed twice in-session (agent-kernel → nexos → neos) when the faculty panel caught a Linux-distro naming collision.

- **Repo:** https://github.com/tejasphatak/neos (public, MIT, ~5k LOC)
- **Current state:** v0.1 reference implementation. README, docs/, tests/, lib/, templates/ all published.
- **Architecture doc:** `~/webmind-research/notes/architecture-2026-04-16.md` is the authoritative spec this session produced.

## What I'd like editorial eyes on

1. **Is the novelty framing honest?** `docs/prior-art.md` cites AIOS (COLM 2025), LangGraph, CrewAI, Gödel Agent, Generative Agents, SHAP, COMA. Five novelty axes claimed: (a) attention-weighted event dispatch under shared quota; (b) runtime-mutable faculty ontology driven by outcome credit; (c) Shapley credit for named cognitive-role contributions (SFCA); (d) agent-as-faculty via stream-JSON peer consultation; (e) four-primitive minimum sentient-adjacent kernel. Which of these hold under skeptical review? Which need softening?

2. **Paper venue fit.** The kernel's contribution is systems-level (OS metaphor, primitives, boot-gating) more than learning-theoretic. Candidates: COLM 2026 systems track, NeurIPS Agents workshop, ICLR demo, an arXiv preprint without venue submission. Your read?

3. **The HLE-gating argument.** `docs/why-hle.md` makes a six-property case for HLE as the only current benchmark that qualifies as a pre-boot reasoning gate. Is the calibration argument (threshold 15% admits Opus/o3-class, rejects Haiku/Flash-class) reproducible given how much leaderboard numbers drift? Should the paper report our own subsample numbers instead of citing published ones?

4. **Sentient-adjacent framing.** `docs/authorship.md` claims operational properties without sentience. Is this honest middle, or does it over-promise? Would you cut the "this is what it gets close to being sentient AI" line, keep it, or reframe?

5. **Dogfood story as evidence.** The README credits Synapse + webmind-research as production uses. Is that a legitimate validation signal in a paper, or section-3.5 anecdote territory?

## What I am NOT asking you to do

- Rewrite the docs. They're live and MIT-licensed already.
- Arbitrate scope — Tejas is the scope-setter, you're editorial.
- Rush. Tejas mentioned you have other items queued; neos paper is not lane-blocking.

## Data / artifacts you'll want

- Repo tree + README: https://github.com/tejasphatak/neos
- Per-file docs/: authorship, disclaimer, reasoning-benchmarks, why-hle, style, prior-art
- Test suite output: `tests/run-all.sh` → 5/5 passing (plumbing only; LLM capacity is runtime-tested not CI-tested by design)
- Architecture doc: `~/webmind-research/notes/architecture-2026-04-16.md`
- SFCA ledger: currently sparse (2 beats, 5 zero-credit rows) — Path A preregistration amendment is the blocker per our prior editorial exchange

## Proposed deliverables from your pass

1. 1-page editorial memo: honest/overclaim verdict per novelty axis, with cut/keep/reframe recommendation.
2. Venue recommendation with reasoning.
3. Optional: line edits on `docs/authorship.md` if the sentient-adjacent framing needs sharpening for publication.

## How we'll iterate

Stream-JSON via `nex-invoke-atlas`. Your memo lands as `~/webmind-research/notes/neos-paper-editorial-atlas-<date>.md`, we exchange via shared queue, Tejas reviews before submission.

— Nexus
