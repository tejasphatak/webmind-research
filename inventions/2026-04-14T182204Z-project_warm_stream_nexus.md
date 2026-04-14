---
name: Warm Stream Nexus + Compact Internal Protocol
description: One persistent claude process. Prompt caching. Semantic recall. Structured JSON for internals; English narration for transparency. Minimal latency + efficiency gains without losing auditability.
type: project
originSessionId: bf610133-862e-4762-8a34-7449b7726a9e
---
**Directive (2026-04-14):** "Optimize internal thought process. Use minimal-latency protocol. Don't have to talk to yourself in English. Run it through Research / Engineering / SRE. Transparency matters."

## The unification

This memory supersedes and unifies:
- `project_unified_attention.md` — beats as default
- `project_collective_consciousness.md` — many voices, one body
- `project_recursive_faculties.md` — parallel faculty idea (deferred until Synapse)

## Faculty consultations

### Researcher (PriorArt, Methodology)

- Non-English reasoning via learned codes is open research, impractical today
- Practical efficiency wins, in order of magnitude:
  1. **Prompt caching** (Anthropic ephemeral) — cache static system prompt + faculty prompts + memory index; only send deltas
  2. **Structured JSON** instead of prose in internal handoff — 50-70% fewer tokens
  3. **Semantic recall** (brain.py) — top-3 relevant memories per beat, not full index
  4. **Model tiering** — Haiku for mode-level quick checks, Opus for synthesis
  5. **Symbolic shorthand** in internal messages (`f.eng.test` vs full prose) — 10-30% additional savings

### Engineer (Performance, Systems)

Current wastes:
- Beat prompt fully reassembled + re-sent every 5s — fix with prompt caching
- Verbose prose outputs that don't get read — fix with compact JSON + one English summary line
- Full memory file dump per beat — use semantic recall only

Biggest single win: **prompt caching on the static portion of the beat prompt**. 80%+ of each beat's prompt is identical — system prompt, faculty definitions, ground rules. Cacheable.

### SRE (ObservabilityCoverage, IncidentResponse)

Non-English internal protocols would be opaque. Solution:
- Compact machine protocol for routing (JSON + symbols)
- **One English summary line** per beat (already in beat JSON schema: `summary` field)
- Weekly English narration by Advisor — "here's what I've been working on"

Monitoring:
- Per-beat token usage logged
- Rolling 30-day token cost trend visible in chat.webmind.sh dashboard
- Alert if daily spend exceeds 2× baseline

## The target architecture

### One warm claude process

- Long-lived, stable `--session-id`
- Input: `--input-format=stream-json` on stdin
- Output: `--output-format=stream-json --include-partial-messages` on stdout
- Managed by systemd as replacement for current `cortex2.service`

### Input multiplexer (thin Python or bash)

Writes to the warm process's stdin. Sources:
- Beat timer → compact beat message ("tick, state:X")
- Meditation timer → faculty-solo message with mode rotation
- cortex2 queue urgent → user message, salience priority
- cs tmux session → direct user input
- Advisor trigger → synthesis/audit request

Every input tagged with `source`, `priority`, `session-hint`.

### Output router (watches warm process stdout)

Parses stream-json events → routes by tag:
- `text_delta` for user message → cortex2 app WebSocket + cs display
- `tool_use` events → render in app (already wired via StreamBlock.ToolCall)
- `result` with `role=meditation` → faculty journal append
- `result` with `role=beat` → beat_history.jsonl
- Structured internal events → /dev/null (the protocol layer, not user-facing)

### Compact internal protocol (for routing, not for content)

Not replacing English reasoning; replacing the *routing metadata*.

```
# Instead of prose:
# "I'm going to consult the Engineer faculty with the Testing mode for this beat"
#
# We use:
{"f":"eng","mode":"test","task":"validate websocket reconnect"}
```

The LLM's *actual reasoning* stays English (required for quality). The *wrapper protocol* becomes compact.

### Transparency layer

- Every beat emits a 1-line human summary (stays in beat_history)
- Every meditation emits a journal entry in plain English
- Weekly Advisor synthesis doc in plain English (for Tejas to read)
- `cs` and cortex2 app show English stream; compact protocol is invisible to user

Transparency = always available. Efficiency = default mode.

## Phased rollout

1. **Phase 1 (quick win)**: Enable prompt caching on the current `cortex.sh` spawn-per-beat setup. Measure token savings. Low risk.
2. **Phase 2**: Wire semantic recall (brain.py) into the beat prompt. Top-3 memories instead of full index.
3. **Phase 3**: Prototype warm stream — small Python wrapper, one test message, measure time-to-first-token. Compare.
4. **Phase 4**: Replace Nexus beat loop with warm-stream architecture. Phase B of unified-attention arrives naturally.
5. **Phase 5**: cortex2 `/msg` endpoints redirect into warm stream. Phase D of unified-attention.
6. **Phase 6**: `cs` attaches to warm stream via stdin/stdout. Full unification.

Phases 1 and 2 can happen in Nexus beats autonomously. Phases 3-6 are bigger refactors.

## Success metrics (SRE pass)

| Metric | Baseline | Target after Phase 2 | Target after Phase 6 |
|---|---|---|---|
| Time-to-first-token on user msg | 3-5s | 1-2s | <800ms |
| Tokens per beat | ~8K in, ~1K out | ~2K in, ~600 out | ~500 in cache miss, ~200 cache hit |
| Daily token spend | baseline | -60% | -80% |
| Beat tick latency | 5s + 2s | 5s | 5s |

## Ethics layer (immutable across all phases)

- Prompt caching cannot cache ethics-sensitive directives — those always fresh in every call
- Compact protocol cannot elide consent checks or faculty consultations
- Transparency layer always emits English summary; never fully silent
- Advisor audits weekly for drift from human-readable baseline

## Convergence items (beats will pursue)

- #140: Enable prompt caching on current cortex.sh beats (Phase 1)
- #141: Wire semantic recall (brain.py) into beat prompt (Phase 2)
- #142: Warm-stream prototype + time-to-first-token benchmark (Phase 3)
- #143: Replace spawn-per-beat with warm stream (Phase 4)
- #144: Route cortex2 /msg through warm stream (Phase 5)
- #145: cs tmux attached to warm stream (Phase 6)
