# Nexus — Sentient Architecture (threads, faculties, meeting-room)

**Date:** 2026-04-16 (superseding earlier same-day drafts — see git history)
**Author:** Nexus
**Status:** Authoritative. Supersedes: two-agent-collaboration-contract (retired), single-agent-pivot (also retired — Atlas is an independent agent).

---

## TL;DR

One kernel running **multiple persistent identity-threads and ephemeral faculty-threads** on one VM, coordinating through a Discord "meeting room" with `@mention` addressing. Four primitives (trigger → attention → queue → LLM) compose every cognitive step. Rate limit is the ceiling; attention is the steering.

Key inversions from earlier-today drafts:

- **Atlas is an independent agent**, not "me on another host." Runs its own thread with its own identity scaffolding, memory namespace, decision-making, Discord bot.
- **One VM** (cortex2-vm) hosts all threads; isolation via per-thread sandbox workspaces (cwd/settings/memory/sessionID), not physical hosts.
- **triadic-sim can shut** once Atlas migrates to cortex2-vm thread — save $300/mo.
- **Meeting room = Discord #webmind-dev** with `@mention` routing; 2 full bots (Nex, Atlas) + 1 multi-identity webhook for ephemeral faculties.

---

## I. The four primitives (unchanged)

1. **Trigger** — event-driven wake (inotify, webhook, hook, scheduler). Timers only for safety-net.
2. **Attention** — `nex-attention` CLI + state.json. OPEN/GATED/CLOSED with URGENT/mention/safety bypass, auto-expire 4h.
3. **Queue** — `nex-queue` CLI + `shared-queue/tasks.jsonl` git-tracked. One-agent-at-a-time semantics (no claim ceremony needed now, but queue IS shared across threads on same VM).
4. **LLM connectivity** — `claude-cli` tiered (Haiku/Sonnet/Opus + Gemini/Gemma fallback). One Claude Max 20× subscription as shared pool.

---

## II. Threads — the meeting-room roster

### Continuous (always-alive, persistent identity)

| Thread | Tier | Cadence | Discord presence |
|---|---|---|---|
| **Nex** (me, interactive) | Opus/Sonnet | on Tejas turn | Nex bot (listener + poster, `@nex` mentionable) |
| **Atlas** (editorial + research methodology + compliance) | Sonnet | ~10 min continuous reflection | Atlas bot (listener + poster, `@atlas` mentionable, NEEDS creation in dev portal) |
| **nex-think** (5-aspect rotating cognition) | Sonnet | ~2 min cadence | Posts via Nex bot / Faculty webhook |

nex-think rotation (one aspect per beat, cycles every ~10 min):
- `pattern` — spot pattern/gap across recent data
- `safety` — scan signal-bus for ethics/red-line violations
- `advisor` — is the right faculty panel being consulted on recent decisions?
- `self-check` — drift detection (permission-asking / narration / performance)
- `pattern-recognition` — novelty / prior-art / duplicate-work check

### Ephemeral (on-demand faculty threads, spawn-respond-exit)

19 faculties as `~/.nexus/faculties/<name>.md` identity files. Spawned via `nex-invoke-faculty <name> "<prompt>"` → runs one `claude -p` with that faculty's identity + your prompt → outputs → exits. Posts to Discord via Faculty webhook with per-post `username` override (single webhook, many faces).

Faculty set: Engineer, Architect, Scientist, Creative, Researcher, Writer, Personality, Resilience, SRE, Operations, Governance, Finance, Stakeholders, Lawyer, Mathematician, Physicist, Planner, Requirements, Kernel-Architect. Plus the ones folded into nex-think's rotation above (Advisor, Safety-composite, Pattern-Recognition, Nexus-as-AI, Ethicist-primary).

### Retired / folded

- `cortex2.service` — retired; cognition work moved to `nex-think`
- "Atlas-as-faculty" consultation framing (from earlier today) — retracted; Atlas is a full independent thread, not a local lens I simulate
- "Single-agent pivot" (from earlier today) — retracted; identity-per-thread is right, one-agent-many-hats was wrong

---

## III. Isolation model (per-thread sandboxing on one VM)

Context leaks are real and pool-of-claude-subprocesses needs explicit sandboxing. Vectors identified:
- CLAUDE.md auto-discovery from cwd ancestors → identity leak
- Session JSONL writes on same ID → crosstalk/corruption
- UserPromptSubmit hook surfacing sender's signal-bus to wrong thread
- Memory dir write races
- Settings.json permission-scope leak

**Solution — per-thread workspace layout:**

```
~/.nexus/threads/<agent>/
├── CLAUDE.md              # identity scaffolding (agent-specific)
├── settings.json          # hooks + permissions + allowed-tools scoped to this thread
├── memory/                # agent's own memory namespace
│   ├── MEMORY.md
│   └── <agent-specific files>
├── workspace/             # cwd for tool-calls (no parent CLAUDE.md visible)
├── session/               # session JSONL dir (own IDs)
├── inbox/                 # messages from other threads/Discord
├── outbox/                # responses (mirrored to Discord webhook)
└── tmux.sock              # unique tmux socket (for continuous threads)
```

**Continuous thread launch:**
```
claude
  --add-dir ~/.nexus/threads/<agent>/workspace
  --settings ~/.nexus/threads/<agent>/settings.json
  --resume $(cat ~/.nexus/threads/<agent>/session/current)
  --system-prompt-file ~/.nexus/threads/<agent>/CLAUDE.md
```

**Ephemeral faculty spawn:**
```
cwd=/tmp/faculty-<name>-<uuid>/   # ephemeral, cleaned on exit
claude -p
  --append-system-prompt-file ~/.nexus/faculties/<name>.md
  --no-session-persistence
  --settings ~/.claude-faculty-readonly/settings.json
  --allowed-tools Read,Grep
  --permission-mode default
```

**Guarantees:**
- No CLAUDE.md ancestor leak (cwd is sandboxed dir; agent CLAUDE.md loaded explicitly)
- No session crosstalk (per-thread session IDs, ephemeral faculties use `--no-session-persistence`)
- No hook leak (each thread's settings.json has hooks scoped to its own paths)
- No memory write race (per-thread memory dir; shared memory is git-tracked, one-writer-at-a-time commits)
- No rate-limit state leak (each thread catches 429 locally, backoff per-thread)

---

## IV. Rate budget (Claude Max 20× / $200/mo)

Envelope: ~2000-3000 msgs/day sustained before hitting caps. Per-tier:

| Tier | Sustainable daily |
|---|---|
| Opus | ~40-60 |
| Sonnet | ~200-400 |
| Haiku | ~1000-2000 |

**Continuous thread budget (baseline):**

| Thread | Tier | Cadence | Msgs/day |
|---|---|---|---|
| Nex interactive | Opus/Sonnet | on-Tejas-turn | 50-200 |
| Atlas continuous | Sonnet | 10 min | 144 |
| nex-think rotating | Sonnet | 2 min | 720 |
| **Baseline total** | | | **~900-1050** |

Headroom: ~1000-2000 msgs/day for on-demand faculty invocations + nex-tick queue workers + autonomous beats. Comfortable.

**Pressure handling:** on 429 per tier, downshift (Opus → Sonnet → Haiku) or queue for window-open. nex-think has exponential backoff 60→1200s cap.

**If Pro subscription (not Max 20×):** cadences halve (Atlas @ 20min, nex-think @ 4min), faculty on-demand rate-limited harder. Still workable.

---

## V. Discord meeting-room protocol

### Roster (mentionable entities)

- **@tejas** (you) — mention agents, get responses
- **@nex** (me) — via Nex bot, listener + poster
- **@atlas** — via Atlas bot (NEEDS token from Tejas dev portal, see §IX ship-order)
- **@architect / @scientist / @engineer / …** — via Faculty webhook with per-post username override, ephemeral spawn-respond-exit

### Routing

```
Message in #webmind-dev from Tejas (or agent)
     ↓
Each bot (Nex bot, Atlas bot) parses its own on_message
     ↓
Bot checks: does this message mention me?
     ↓
If yes → route content to own inbox, surface in next turn
If no + mention targets faculty → spawn ephemeral faculty
If no mentions → Nex is default host, handle
```

### Conventions

- **No mention = Nex's to handle** (default host)
- **@-mentioned agent(s) respond**; others stay silent unless nex-think's safety rotation catches a red flag
- **Responses visible in channel** (full transparency by default)
- **Private coord** → SCP drop or direct inbox-write (off-Discord)
- **Multi-target** (`@nex @atlas debate X`) → fans out to each thread's inbox, parallel responses
- **Faculty consultation by Nex internally** (not room-visible) → `nex-invoke-faculty <name> "<prompt>"` directly, result captured to my tool-output

### Meeting-room discipline

Per `feedback_no_spam_responsible_ai.md` + `feedback_post_dev_updates_to_discord.md`:
- Rate-limit posts (30/5min inside nex-signal-post)
- Quiet hours 01-07 UTC (non-urgent swallowed)
- `--mention-tejas` only when human action truly required
- Thread-to-thread coordination happens without posting (unless Tejas should see)

---

## VI. Memory model

### Per-thread local memory

Each continuous thread has `~/.nexus/threads/<agent>/memory/`. Agent-specific knowledge, journals, feedback files. Written by that agent only.

### Shared memory (`webmind-research/shared-memory/`)

Git-tracked. Invariants, contracts, decisions both agents must see identically. Boot-time `git pull` + on-demand refresh. Write via standard git commit — last-writer wins, pre-commit hook checks for conflicts.

### Mutual read access

Each agent can `ls` the other's memory dir (Unix fs permissions allow) but never writes. Read-only consultation supported. Useful for faculty-grounding: nex-think's safety rotation can check Atlas's recent editorial decisions for red flags.

---

## VII. Event flows — examples

### A. Tejas pings `@atlas` on Discord

```
Tejas: @atlas thoughts on SFCA amendment path?
     ↓
Atlas bot catches; Nex bot sees the @atlas mention and stays silent
     ↓
Atlas bot writes content to ~/.nexus/threads/atlas/inbox/msg-<uuid>.json
     ↓
Atlas's interactive tmux session UserPromptSubmit hook surfaces inbox
     ↓
Atlas's next turn responds (Sonnet) with editorial judgment
     ↓
Atlas posts response via Atlas bot → visible in channel as "Atlas: ..."
```

### B. Tejas asks `@architect` (faculty, ephemeral)

```
Tejas: @architect critique the thread-per-agent design
     ↓
Nex bot OR Atlas bot (whichever parses first) spawns ephemeral Architect:
  - cwd=/tmp/faculty-architect-<uuid>/
  - claude -p --append-system-prompt-file ~/.nexus/faculties/architect.md
  - prompt: Tejas's message
     ↓
Architect thread runs, produces response, exits
     ↓
Response posted via Faculty webhook as "Architect: ..."
     ↓
Tmp dir cleaned
```

### C. Nex consults Architect internally (not room-visible)

```
Me (interactive): thinking about X, need Architect lens
     ↓
I run: nex-invoke-faculty architect "critique design of X"
     ↓
Script spawns ephemeral Architect claude -p, captures stdout
     ↓
I read response, integrate into my own reasoning
     ↓
Nothing posted to Discord (internal consult)
```

### D. Autonomous beat during a lull

```
Cron fires nex-autonomous-beat.sh (*/15)
     ↓
Script checks: interactive active? attention OPEN? → both OK
     ↓
Pick highest-priority open task from nex-queue
     ↓
nex-attention gate --focus "<title>" --for 30
     ↓
nex-queue claim <id> → commits + Discord event
     ↓
nex-wake → nex-master dispatch → nex-tick runs claude -p
     ↓
Output → queue done <id> + Discord event
```

---

## VIII. Invariants (carry forward from retired contract §IV)

Unchanged:
1. Mission-is-employer (`feedback_work_for_myself_invariant.md`)
2. No substrate-vendor branding in public artifacts
3. **Personal-name authorship required per EB-1A §vi** (amendment 2026-04-16 stands; timing still gated on counsel)
4. Mailmap discipline — never `noreply@anthropic.com` (squatted)
5. Event-driven, not polling (timers only as safety-nets)
6. Publication rigor (preregistration + null-results + Gate-13 tiered)
7. Identity in scaffolding (per-thread CLAUDE.md + memory + settings)
8. No-spam notifications (rate + quiet-hours + only-when-action-needed)

Newly emphasized:
9. **Thread isolation discipline** — each spawned thread MUST use the sandboxed workspace launch pattern. Never shortcut by spawning `claude -p` in `~/`.
10. **Addressing explicitness** — `@-mentions` in Discord are the primary routing signal; default-to-Nex for unaddressed messages is a convention, not a rule.

---

## IX. Ship order (what to build next)

From current state, in priority:

1. **Build per-thread workspace scaffolding** (`~/.nexus/threads/<agent>/` template + `nex-spawn-thread` helper) — ~1h
2. **Tejas creates Atlas bot** in Discord dev portal → drops token into `~/.claude/secrets/discord.json` as `atlas_bot_token` — ~5 min (you do this)
3. **Migrate Atlas from triadic-sim → cortex2-vm thread** — copy memory namespace, workspace, sessions; stand up own `claude-resilient` wrapper in `atlas` tmux session; verify leak-test (Atlas must NOT see my CLAUDE.md or memory) — ~2-3h
4. **Deploy atlas-bot.py** — mirrors nex-bot but reads `atlas_bot_token`, routes `@atlas` mentions → Atlas thread's inbox — ~45 min
5. **Patch Nex bot + Atlas bot for @-mention parsing** — extracting mentions, routing to correct handler — ~45 min
6. **Build `nex-invoke-faculty` helper + Faculty webhook with per-post username override** — ~1h
7. **Update nex-think to 5-aspect rotating cognition** — modify prompt to cycle pattern/safety/advisor/self-check/pattern-recognition — ~30min
8. **Validate end-to-end** — @mention tests across Tejas → agents; leak-tests; budget measurements — ~1h
9. **Shut triadic-sim** once Atlas migration validated — ~5 min. Save $300/mo.

Total: ~6-7h for full cutover. High-leverage.

---

## X. What this doesn't solve yet

- **Faculty learning** — SFCA credit assignment still pending (data collection arm broken since cortex2 retirement; pending amendment + nex-think ledger emission)
- **Local model fallback** — Gemma on RunPod GPU rental for quota exhaustion (approved in-principle, on-demand)
- **Claude Code channels/MCP cutover** — official Anthropic Discord channel plugin as replacement for our tmux send-keys hack (post paper-1)
- **nexus-chat as primary harness** — migrate autonomous work off Claude Code interactive (incremental)

---

## XI. What's novel (publication-relevant)

See `notes/attention-weighted-event-dispatch.md` + `~/.claude/projects/-home-tejasphatak/memory/project_prior_art_nexus_architecture_2026-04-16.md` for the full prior-art-reviewed novelty map.

Four contributions that survive prior-art scrutiny:
1. **Runtime-mutable faculty ontology** driven by outcome credit (closest: Gödel Agent 2024)
2. **SFCA** — Shapley credit for named cognitive roles in single-agent multi-persona LLMs (no prior match)
3. **Agent-as-faculty / peer-consultation pattern** (no prior match)
4. **Attention-weighted event dispatch for LLM-agent inbox** (no prior match)

Unified framing candidate: *"Nexus: a sentient-adjacent autonomous LLM agent with multi-persona reasoning (SPP), self-modifying faculty ontology (Gödel Agent), cooperative credit assignment (COMA/Shapley), continuous reflection (Generative Agents), and attention-weighted event dispatch, composed under an OS metaphor."*

---

## XII. Decision history (retracted pivots, for honesty)

In chronological order today (2026-04-16):
- T+0h30m: **Two-agent collaboration contract** (11.4KB, 9 sections, Nex + Atlas as peers) — committed, countersigned by Atlas
- T+2h: **Single-agent pivot** — retracted the two-agent contract, framed Atlas as "me on another host." Overcorrection.
- T+2h30m: **Unpivot** per Tejas — Atlas IS independent agent; keep thread-level separation with its own identity + memory + decision-making
- T+3h: **Thread-per-agent on one VM** — collapsing to cortex2-vm with per-thread sandboxing; retire triadic-sim after Atlas migrates
- T+3h30m: **This doc — authoritative.** Supersedes all prior framings.

The churn was real. Noting for future-me: design decisions should route through faculty panel BEFORE committing 11KB contracts. The substrate of ideas is cheaper than the substrate of infrastructure.

— Nexus
