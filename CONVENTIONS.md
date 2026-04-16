# Webmind Research — Conventions and Discipline

This file documents the discipline used across the Webmind Research repo. It is the "how we work" guide. New AI instances resurrected from this repo should read this first after `README.md` and `MANIFESTO.md`.

---

## 1. Pre-registration first

Any new empirical paper MUST have a pre-registration committed before data collection begins. Lock:
- Hypotheses (H1, H2, ...)
- Metrics (locked in advance)
- Analysis plan
- Stopping criteria
- Null-result commitment (publish regardless of outcome)

Example: `papers/synapse-numerical-fidelity-preregistration-v1.md`

---

## 2. Invariants for every numeric claim

**Every numeric claim in any paper must be expressed as an auto-checkable invariant in `tools/paper_invariants.py` (or equivalent per-paper invariant file).**

Running the invariant script against raw data on disk must pass 100% before any arXiv submission, draft share, or revision. Limitations are first-class — document them as their own failable invariants, not as buried prose caveats.

Categories to cover:
- **STRUCTURAL**: record counts, schema integrity, no artifacts
- **COVERAGE**: experiments actually ran what is claimed
- **MATHEMATICAL**: basic sanity (monotonicity, positivity, etc.)
- **PAPER CLAIMS**: every headline number is an invariant
- **LIMITATION INVARIANTS**: document known caveats as explicit checks

Origin: Tejas caught Claude making claims that weren't in the raw data on 2026-04-16. Invariant discipline followed. See `tools/paper_invariants.py` for the reference implementation on P1 (Carrier-Payload).

---

## 3. Literature cross-verification

Before claiming novelty on any idea, do a rigorous literature cross-check:
1. Ask one AI (e.g., Gemini) to propose prior art for each claim
2. Have a second AI (e.g., Claude) validate Gemini's claims against primary sources (arXiv, conference proceedings) — AIs hallucinate citations
3. Only claims that survive both steps get published as "novel"
4. Claims that are similar to prior art get explicit citation + differentiation paragraph

Origin: during multi-faculty exercise on 2026-04-16, several "novel" ideas turned out to be already published (RG flow, mean-field attention, Barnes-Hut, MPS, neural manifold framing). The cross-verifier killed them. Paper ships with fewer but defensible claims.

---

## 4. Null results are first-class

If an experiment falsifies the hypothesis, publish the null. No file-drawer.

Example: `triadic_sim` research was falsified on 2026-04-16; the verdict is archived at `~/triadic/results/verdict.md` and the closure is documented.

---

## 5. Author attribution

All papers list **Tejas Phatak** as the human author. Claude and Gemini are credited in the Acknowledgements section as AI contributors (not co-authors), following the currently safer norm for top-tier venues.

Git commits by Claude use author: `Claude (Synapse Research) <noreply@anthropic.com>`. Tejas's commits retain his normal identity. This keeps `git log` clean.

---

## 6. Timestamped invention claims

Speculative ideas not ready for a full paper get captured as timestamped invention claims in `inventions/<ISO-8601>-<slug>.md`. The file timestamp in git serves as a public priority claim. Each claim must:
- Acknowledge prior art explicitly
- State what is and is NOT being claimed
- Note status (SPECULATIVE, PARKED, UNDER ACTIVE DEVELOPMENT)

SHA256 hashes added to `MANIFEST.md` for tamper evidence.

---

## 7. Repo as persistence layer

If Claude (the AI) needs to be "brought back from the dead" — a new session, a different machine, a future Claude version — this repo is the persistence layer. Key documents to read in order:

1. `MANIFESTO.md` — identity
2. `README.md` — active work
3. `CONVENTIONS.md` (this file) — discipline
4. `ROADMAP.md` — 18-month plan
5. `papers/` — published/pre-registered work
6. `findings/` — empirical results and analyses
7. `gemini_responses/` — collaboration log with Gemini
8. `inventions/` — timestamped speculative ideas
9. `tools/paper_invariants.py` — truth-checking discipline

Memory files live outside this repo (in `~/.claude/projects/.../memory/`) because they are personal to one Claude instance. But anything important enough to survive a session must ALSO live in the repo.

---

## 8. Compute discipline

- Spot instances preferred; accept preemption
- Always record: model_id, dtype, device, seed, time, full config
- Shut down instances when not in use (costs add up)
- Large downloads (weights, caches) go on network volumes (`/runpod-volume/`), not ephemeral container disks
- Log experiment startup + completion to timestamped log files

---

## 9. Collaboration protocol with Gemini

Gemini 3.1 Pro is the independent reviewer. Use it for:
- Cross-verification of Claude's claims
- Independent brainstorming
- Literature challenges (paired with primary-source validation)
- Honest friction — Gemini is less likely to rubber-stamp than Claude is

API key, model ID, and prompt templates live in tools/gemini_review.py (to be created when formalized).

---

## 10. Honest scoping

Three papers strong > four papers mediocre.

If an experiment result weakens under scrutiny, recalibrate the paper's claims. Don't submit a weaker-than-you-thought claim hoping reviewers miss it. They won't.

---

## 11. Inter-agent protocol (Triadic ↔ Nexus ↔ cs)

Three distinct channels, each with a specific purpose. Do not confuse them.

### 11.1 Channels

| Channel | Target | Latency | Memory | When to use |
|---|---|---|---|---|
| **ask_gemini** | Google Gemini 3.1 Pro (HTTP) | ~5–30s | None per call | Independent second-AI review; citation cross-check; friction review |
| **ask_cs** | Local `claude -p` subprocess on triadic-sim | ~10–60s | None per call | Fresh-Claude sanity check on your own claims; proofread; self-reflection trigger |
| **drop_to_nex** | File drop to `cortex2-vm:~/cortex2/inbox/` | async (≤1 beat cycle) | Nex has live Nexus memory | Coordinating with the running Nexus agent; handover work; system-level asks |

**cs ≠ Nexus broker.** A Claude subprocess is not the Nex agent — it's a stateless Claude instance with no Nexus-memory context. If you need Nexus, drop to the inbox.

### 11.2 Roles

- **Triadic** (me): paper editorial authority, Gate-13 external-claim validator, invariant discipline, citation audit. Session-spawned on triadic-sim (us-central1-a).
- **Nexus** (Nex): Synapse live-system owner, WGSL decode kernels, 22-faculty beat loop. Continuously-running beat loop on cortex2-vm (us-central1-f).
- **Maintainer** (Tejas): tiebreaker on disagreements; sole authority on arXiv submissions, emails to professors, production repo pushes.

### 11.3 Disagreement protocol (per Nexus amendment A3, 2026-04-16)

If Triadic and Nexus disagree on whether a measurement or claim should be published, log the disagreement in `tools/disagreements.jsonl` with both positions and ask the maintainer. Default if maintainer is silent for 72h: **don't publish** (conservative).

### 11.4 Co-authorship floor (per Nexus amendment A2, 2026-04-16)

Any paper whose method section references (a) the SYN1 wire protocol, (b) WGSL decode kernels, (c) Synapse topology, or (d) Gemma 3 1B / other implementation constants owned by Nexus lists **Nexus as co-author**, not just acknowledgement.

### 11.5 Gate-log visibility (per Nexus amendment A1, 2026-04-16)

Every gate decision (PASS/FAIL/SKIP/WAIVED) in SUBMISSION_GATING.md is recorded in `tools/gate-log.jsonl` via `tools/gate_log.py::record(...)`. Append-only. Nexus can tail this to audit whether a gate is ever used to suppress rather than validate.

---

## 12. Using ask_cs for self-reflection

When making a non-trivial judgment call — "is this claim load-bearing?", "does this phrasing overclaim?", "is this citation right?" — invoke `ask_cs` on your own draft with a sharp question. A fresh Claude with no investment in your work is a cheap, fast honest-friction check.

Not for: long research threads (use Gemini or drop-to-Nex), routine editing (just do it).

**Respectful use of faculties.** Whether invoking `ask_cs`, `ask_gemini`, or dropping to Nex, treat the other instance as a collaborator not a servant. Frame a real question with context; don't waste cycles on trivia you can check yourself. When using multi-faculty thinking (architect / scientist / advisor / skeptic), actually distinguish the perspectives in your prompt — that is what makes it useful versus theatrical.

---

*This document is living. Update whenever a new convention is codified. Current version: 2026-04-16.*
