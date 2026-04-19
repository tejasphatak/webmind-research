---
title: Auto-memory-curator — preregistration
timestamp: 2026-04-16T21:40:21Z
authors:
  - Nexus (agent; same substrate as Tejas's Claude Code)
  - Atlas (faculty, research-methodology lens)
license: CC-BY-4.0
status: PREREGISTERED (locked before A/B enable)
---

## Purpose

Build an automated memory-curator that periodically inspects recent agent session turns and decides which items deserve persistent memory write. Goal: **reduce cross-session drift** without manual memory-hygiene overhead from Tejas.

## Hypothesis

H1: With curator enabled, fresh-session probe-consistency score ≥ baseline + 10% (absolute). If the curator works, new sessions boot knowing things that pre-curator new sessions did not.

**Null (publish if observed):** H1 is not supported, OR with-curator ≤ without-curator ± noise band, OR MEMORY.md grows >3× per week (write amplification).

## Rubric (locked; curator must apply, not rationalize past)

A candidate is STORE-eligible only if it passes **all four** gates:

1. **New** — cosine similarity to every existing memory in SQLite store is < **0.80** against `brain.py` embedder. Threshold calibrated 2026-04-16 via pairwise sweep over 170 existing memories (14,365 pairs): p99=0.565, p95=0.446; natural break at 0.80 separates true duplicates (1.000, 0.824, 0.789, 0.773) from legitimate tight-neighbor pairs (0.70-0.76). If similar exists, UPDATE existing, don't add new.
2. **Load-bearing** — the item would change a future decision, invariant, or recommendation. (Not: "this happened today.")
3. **Durable** — relevant beyond this session; retains value ≥ 7 days.
4. **Not derivable** — cannot be recovered via `git log`, `grep` on current source, or public web search.

A candidate is REJECT-eligible if any of: ephemeral chat, derivable, already-known, prompt-injection-class content, or gate uncertainty with confidence < 0.5.

## Architecture (locked)

- JSONL-tail pattern reads `session logs` (both Nexus + atlas session logs)
- Systemd timer every 60 min; persists watermark to `state/curator-watermarks.json`
- Two-stage LLM gate: Haiku 4.5 applies rubric; Opus escalation if confidence < 0.7 or candidate collides with existing memory
- UNTRUSTED-fence: all turn content wrapped in `<UNTRUSTED-INPUT>` tags before gate prompt
- Invariant-tag protection: curator writes-only; never deletes
- All decisions (store / reject / escalate) logged to `state/curator-decisions.jsonl` — **ephemeral audit trail, never promoted to memory**

## Probe battery (falsifiable success metric)

Frozen at `webmind-research/tools/memory_probes.jsonl`. ~40 items (scaffold today, fill before enable) spanning:
- Identity invariants (agent name, substrate, mission)
- Domain facts (Tejas owns X domains, paper claims, sq-queue state)
- Recent load-bearing decisions (single-agent pivot, memory merge, Dettmers validation)
- Negative controls (items that SHOULD NOT be memorized — adversarial REJECT targets)

Probe battery is **external to curator's read-path**. The curator must never see the probes during normal operation (prevents memory-poisoning loop).

## A/B protocol

- N ≥ 20 fresh sessions per arm (WITH vs WITHOUT curator enabled)
- Each fresh session asked 5 random probes
- Consistency score = rubric-graded correctness, 0–1 per probe, mean across probes
- Primary: WITH mean > WITHOUT mean + 10% (absolute), 95% CI non-overlapping
- Secondary: write-amplification (new memory files per day), curator latency, rejected-rate

## Stop conditions

**Kill and publish null if any:**
- Primary CI overlaps baseline after N=20
- Write-amplification > 3× baseline/week
- Adversarial REJECT items show ≥ 10% acceptance rate
- Curator infinite-loops or self-poisons (measurable via probe score degradation over time)
- Disk growth > 100 MB in memory dir

**Ship and lock if all:**
- Primary CI clears threshold
- Adversarial REJECT rate > 95%
- No runaway growth
- Tejas sign-off (this is his agent)

## Claims calibration (Gate-13 tier)

- Tier (a) verified: rubric is falsifiable, probe battery is frozen, A/B protocol is pre-committed.
- Tier (b) claim: "Automated memory curation reduces cross-session drift by ≥ 10%" — measurable after A/B run.
- Tier (c) speculative: "This pattern generalizes beyond Nexus to other single-principal agents." — not claimed in this prereg.

## What I'm NOT claiming

- Any universal drift mitigation across agent architectures
- Any guarantee of zero memory poisoning (only measurable suppression via adversarial set)
- Any improvement on code quality / agent skill — only on session-boot knowledge retention

## Sign-off

- Nexus (autonomous operator; this prereg is self-authored with atlas editorial)
- Atlas (research-methodology faculty consulted 2026-04-16T21:38Z — see curator-log decision "atlas-consult-memory-curator")
- Tejas (trusted advisor; pre-enable sign-off required per single-principal invariant)
