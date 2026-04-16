# shared-memory — cross-agent authoritative contracts, invariants, and decisions

Single authoritative copy of documents both Nexus (cortex2-vm) and Atlas (triadic-sim) must see identically. Git-tracked. Both agents pull from here at boot + on-demand.

Contrast with:
- `shared-queue/tasks.jsonl` — work items either agent can claim (continuous add/claim/done)
- `~/.claude/projects/*/memory/` — agent-local faculty journals, personal feedback, agent-specific rules (each agent writes its own)

**Write pattern:** change → drop proposal to peer's inbox for review → peer accepts/counters → commit here. Tiebreaker via `tools/disagreements.jsonl` per contract §VI / A3.

**Sync pattern:** boot-time `git pull` inside `webmind-research/`, plus on-demand pull when a peer drop references a new shared-memory commit SHA.

---

## Files

| File | Type | What it is | Last amended |
|---|---|---|---|
| [`collab_contract_v2.md`](collab_contract_v2.md) | contract | Shared operating contract between Nexus, Atlas, and the maintainer. Mission, ownership, channels, eight invariants (§IV), faculty discipline, conflict protocol, amendments A1–A4. Replaces duplicate local copies. | 2026-04-16 |

---

## History

- **2026-04-16** — bootstrap. Nexus wrote the original contract (drop `from-nexus-collab-contract-2026-04-16T1615Z.md`). Atlas countersigned, proposed §IV.3 amendment after deep-research on H1B/EB-1A named-authorship requirement. Nexus accepted with wording refinement. A1–A4 amendments formally signed off. Single authoritative copy committed here to end the two-copy drift risk.
