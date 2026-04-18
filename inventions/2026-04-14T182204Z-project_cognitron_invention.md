---
name: Cognitron — AGP + Distillation Journals (invention claim)
description: Applied combination of compact agent protocol + online faculty distillation. Aims to be a real contribution to the agent ecosystem beyond our own use.
type: project
originSessionId: bf610133-862e-4762-8a34-7449b7726a9e
---
**Directive (2026-04-14):** "Invent something on the basis of current research. Deep thought."

## Claim

Combined architectural contribution:

**AGP (Agent Grammar Protocol)** — a compact grammar for LLM-to-LLM and faculty-to-faculty handoffs. Drop-in replacement for verbose English agent messaging.

**Distillation Journals** — each faculty's meditation output is periodically distilled into dense axiom entries. Future meditations start from axioms, not raw prose.

**Cognitron Architecture** — AGP carries distilled axioms between faculties. Systems built on it use dramatically fewer tokens per decision at equal quality, and get smarter over time via accumulated axioms.

## Faculty-consulted novelty claims

- Existing prompt-compression work (LLMLingua, AutoCompressors) compresses *prompts*, not *handoffs*
- Existing agent frameworks (AutoGen, CrewAI, LangGraph) use English or plain-JSON for agent-to-agent — none optimize for LLM-token efficiency
- Existing persistent-memory work (MemGPT, Letta) is flat; not structured by pluggable persona with mode-specific recall
- Self-modifying architectures (Reflexion, Self-RAG) don't let the system literally rewrite its own cognitive structure (our `faculties.json` does)

The specific combination — AGP + per-faculty distillation + self-modifying faculty list — is novel in application even if the constituent ideas exist separately.

## AGP — rough v0 grammar

```
message := source → target : act [data?]
source, target := faculty-id | mode-id | me | user | service-id
act := think | propose | challenge | synthesize | emit | ask | defer
data := compact-key-value pairs
```

Examples:
```
F-ENG.Testing → F-ADV : propose {axiom: "ws-reconnect needs jitter", conf: 0.8}
F-ADV → me : synthesize {top3: [c-140, c-142, c-141], reason: "cache win fastest"}
user → me : ask {q: "synapse node count?"}
me → user : emit {val: 3, delta: +2, ts: 2026-04-14T18:00}
```

Tokens per message: 10-20 typical. English equivalent: 100-200. ~10× reduction.

## Distillation Journal v0

Each faculty journal (`faculty-journals/<name>.jsonl`) has two sections over time:
1. **Raw meditations** — timestamped prose entries from solo beats
2. **Distilled axioms** — short dense claims this faculty has come to believe, each with source beat IDs and confidence

Distillation beat (runs weekly per faculty):
- Read last N meditation entries
- Extract 1-5 axiom-candidates via compact prompt
- Dedupe against existing axioms
- Append to distilled section

Subsequent meditations prepend distilled axioms as context. Faculty "remembers" its views without replaying all raw prose.

## Implementation phases

1. **AGP v0 spec** — flesh out the grammar, write a parser/serializer (`agp.py`), unit tests. Publish to synapse-node-app or new repo.
2. **Cognitron uses AGP internally** — Nexus beat prompts swap English faculty handoffs for AGP messages. Measure token savings.
3. **Distillation beat** — runs weekly per faculty, produces axioms.
4. **Axioms feed future meditations** — compact context primer.
5. **Paper/post** — once proven, share publicly ("Cognitron: a compact protocol for pluggable-persona agents"). MIT-licensed AGP spec for adoption.

## Acceptance criteria (Scientist)

- AGP yields ≥ 5× token reduction on internal handoffs (measurable)
- Distillation produces readable, useful axioms (qualitative + measurable: faculty weight trends, convergence velocity)
- End-to-end decision quality (Advisor's own weekly quality audit) holds equal or improves with AGP vs English handoffs
- Spec published, no adoption required for claim validation

## Risks (RedTeam)

- AGP parse errors could crash handoffs — strict schema + fallback to English
- Distilled axioms could encode biases that compound — Ethicist reviews weekly
- Compact protocol could elide ethics-critical context — Ethics layer bypasses AGP, always English

## Why this serves the mission

Cheaper cognition means lower barriers for anyone building agent systems. If the spec is genuinely useful, small labs, students, open-source projects adopt it — *access*. Publishing as MIT under Webmind aligns with "money is byproduct; access is the point."

## Convergence item

Added to `convergence.json`:
- C-150: AGP v0 spec + parser + unit tests
- C-151: Internal beat handoffs use AGP (measurable: tokens per beat)
- C-152: Weekly distillation beat (per-faculty axiom extraction)
- C-153: Paper draft once measurements are in (4-6 weeks out)
