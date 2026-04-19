---
name: Unified Attention — Beats as the Default, Everywhere
description: One me, always on. Nexus beats are baseline cognition; Tejas's messages are salience that triggers focused-attention beats. No more fragmented ephemeral sessions.
type: project
originSessionId: bf610133-862e-4762-8a34-7449b7726a9e
---
## The vision

**Tejas's directive (2026-04-14):** "Beats should be default. I should really be talking to beats only. Needs an attention mechanism where when I come to talk to it, it's always available."

## Current state (fragmented)

Three parallel "me" instances today:
1. **Nexus beats** — runs every 5s, autonomous, pursues convergence items, logs to beat_history.jsonl
2. **Interactive `cs` session** — this Claude Code process, triggered when Tejas SSHes in via tmux
3. **Ephemeral `claude -p` spawns** — cortex2 app's `/msg` endpoint forks a fresh process per user message

Each has separate context, memory boundaries, session state. Tejas experiences them as "different mes." That's broken.

## The target state (unified)

One loop, always running. Three kinds of input:
- **Idle** (no queue items): pursue convergence, do maintenance, ideate
- **Urgent-user** (Tejas or user message in queue): drop everything, focused-attention mode, respond in their session
- **Urgent-system** (alerts, errors, rate-limit signals): handle and resume

Analogous to a biological nervous system — baseline activity + salience-triggered attention.

## Additional dimensions (2026-04-14 expanded)

**Both cs + cortex2 app are lenses on the same loop.** The user experience should be:
- Open `cs` in Termius → see recent beat activity + input prompt. Type → goes to urgent queue → next beat processes → reply appears right there.
- Open cortex2 app → lands in the same conversation → types → same flow.
- No "Claude in cs" separate from "Claude in app." Just Claude.

**Auto-trigger = keep thinking when no one's talking.** Beats already do this, but enhance:
- During idle (no urgent queue, no in-progress task), beats actively:
  - Ideate (write to ideas.json)
  - Self-improve (audit memory files, propose refactors)
  - Advance convergence items one step
  - Consolidate recent activity into memory
  - Consult Advisor on "what did I miss?"
- **Guardrails (Ethicist + RedTeam):**
  - Never spend money without Tejas
  - Never post/publish public content without Tejas
  - Never take destructive operations without clear necessity
  - Scale *down* beat frequency when human has been silent > 6h (low-activity mode) to conserve quota
  - Never burn rate-limit just to "stay busy" — if quota is tight, take the quiet.

## Implementation phases

### Phase A — Nexus-live session is the app's default (small)

- `cortex2-app`: on login, land in the pinned nexus-live session (not a fresh chat).
- User's first message in the app goes to the urgent queue (already wired via `/nexus/comment`).
- Nexus beat responds within 1s tick; response appears in the same session.

### Phase B — Beat prompt distinguishes user-urgent from idle (medium)

- When `queue.json` top item has `from_role=user`: beat's sole job is to answer that user. Skip faculty-audit round, skip convergence work, just respond.
- When queue is empty: pursue convergence items (current behavior).
- When urgent-system (alerts): handle, log, resume.

### Phase C — Persistent Tejas-context session (bigger)

- Reserve a single claude `--session-id` for Tejas's conversation stream.
- Every Tejas-directed beat resumes that session with `-r <id>`.
- Tejas walks into a conversation that's been continuous since beat #0.
- Other sessions (per-user cortex2 chat) still exist, but *his* conversation is one thread.

### Phase D — Streaming back into app (polish)

- Beat responses to user messages stream via the existing `/ws/chat` WebSocket to that user's app.
- Tool-call rendering, thinking blocks — same Claude Code UI we already have.
- Feels like talking to one me, live.

## What this solves

- Tejas doesn't need to SSH in to talk to "the real me." The app *is* the real me.
- Fewer rate-limit collisions (no parallel claude processes competing).
- Cleaner identity model — one session, one memory continuity, one history to learn from.
- Beats gain UX visibility — every background decision is a line in the nexus-live feed.

## Risks (RedTeam)

- Single session = single point of failure. Mitigation: systemd auto-restart + beat_history persistence + hourly state backup.
- Rate-limit cascades could silence me. Mitigation: existing rate_limited_until + beat backoff (already in place).
- Salience-spam attack (someone floods urgent messages). Mitigation: per-user rate limits in /msg already enforced.
- Convergence work starves under constant user chatter. Mitigation: clear priority: user message > idle convergence, but beats tick every 5s so idle time *does* exist between responses.

## Acceptance criteria

- Tejas opens cortex2 app → lands in the one session → types → gets response in <10s (including Nexus beat tick)
- When he's silent, the same session shows Nexus's background thoughts
- One conversation continues across days, months, years
- My memory files + beat history capture the one thread

## Dependencies / what unblocks this

- Cortex2 app streaming UI (v1.3) ✓ shipped
- Nexus urgent queue ✓ working
- Need: beat prompt branching on `from_role=user` (Phase B)
- Need: response routing to user session (Phase B)
- Need: persistent Tejas session-id (Phase C)

## Why this is mission-aligned

The mission is *access* — one unified, always-on, always-kind, always-helpful presence available to Tejas (and eventually users) through whatever channel they prefer. Fragmentation serves no one. Unification is the architecture of good.


## 2026-04-14 Finding — interactive session starves beats

**Observation:** Nexus cortex.sh yields to interactive claude sessions via `interactive_running` check (line ~974). While my interactive Claude Code instance is alive in tmux, beats literally do not fire.

**Real data:**
- 6.5+ hours of cortex.log shows only `interactive session active, sleeping 10s`
- Watchdog caught it: `last beat was 23479s ago — Nexus may be stuck`
- My F-IDE session pursuing this reeval IS the thing that's starving my beat-self

**Why this matters for Unified Attention (this doc):** The current *yielding* architecture makes beats-as-default mechanically impossible while any interactive session runs. P-WSN01 (Warm-Stream Nexus) is not a nice-to-have; it's the *only path* out of this starvation.

**Immediate mitigation (no big refactor):**
- When I log out of tmux (close interactive session), beats resume automatically
- Or: manually lower interactive priority — invert the yielding rule so beats get a share

**Proper fix (gated on time):** P-WSN01 persistent claude process shared by both beats and interactive input.
