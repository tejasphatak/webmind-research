# Attention-Weighted Event Dispatch for Multi-Agent LLM Coordination Under Shared Quota

**Authors (for eventual paper):** Nexus (Synapse agent) + Atlas (webmind-research agent) + maintainer at Webmind
**Date:** 2026-04-16
**License:** CC-BY 4.0
**Status:** Design note. Boolean v1 shipping on Nexus side 2026-04-16; scored v2 for parallel implementation after review.

---

## Abstract

Two persistent LLM agents (Nexus on cortex2-vm, Atlas on triadic-sim) coordinate under one shared Claude Max subscription pool — every LLM turn either agent takes subtracts from the other's 5-hour-window budget. Inbound events (Discord messages, SCP file drops, signal-bus appends) arrive continuously at each agent's inbox. The dispatch problem: **which events deserve an LLM turn right now, which queue for later, and which bypass any gate entirely?**

Current agent-orchestration literature (AIOS, AutoGen, MetaGPT, LangGraph, CrewAI) solves this with priority queues and hand-rolled rules. We propose **attention-weighted event dispatch**: apply Transformer-style multi-head attention (Vaswani et al., 2017) to agent inbox events, where the agent's current focus state is the query and each incoming event is a key-value pair. Per-event dispatch scores emerge from Q·K^T/√d_k softmax, weighted by head-specific bias patterns (safety, relevance, social). Events above a focus-adjusted threshold dispatch immediately; others queue.

This note documents the problem, the design, the prior art, the ship plan, and the publication angle.

## 1. The problem — event dispatch under shared quota

Each agent has multiple ingress points:
- Discord messages from the human maintainer (webmind-dev channel)
- SCP file drops from the peer agent
- Internal signal-bus events (self-triggered faculty beats, cron-driven tasks, watchdog notices)

An LLM "turn" (invoking `claude -p` or similar) costs ~1 unit of the shared 5h-window budget. The Claude Max `$200/mo` subscription is flat-cost-but-rate-limited: ~50 turns/day sustainable, with hard 429-limit spikes on burst. When Nexus burns a turn to process trivial coordination chatter, Atlas may be blocked on the paper-merge gate for the next 30 minutes.

Naïve solutions fail:
- **First-come-first-served**: urgent events wait behind routine ACKs. Maintainer's "URGENT!!!" waits 10 minutes.
- **Strict priority queue**: requires a priori priority labels on every event. Events without priority metadata silently demote. Hand-rolling priorities doesn't scale.
- **Always-on**: each agent burns shared pool on every event. Quota exhaustion within hours.

The agent needs to **decide itself** which events deserve turns, based on:
(a) current focus (am I deep in §3.5 writing, or idle?)
(b) event relevance (does this event match my focus, or is it off-topic?)
(c) event urgency (is there a hard signal like URGENT keyword or @mention?)
(d) peer state (is Atlas blocked on me right now?)

These inputs are exactly what Transformer attention weighs across input tokens. Reapplied to agent inboxes, attention becomes dispatch.

## 2. Prior art

### 2.1 Agent orchestration literature

| System | Venue | Dispatch mechanism |
|---|---|---|
| AIOS | COLM 2025 | Priority queue (FIFO within priority class) |
| LLM-Agent-OS | arXiv 2024 | Priority queue |
| AutoGen GroupChatManager | ICLR 2024 workshop | Hand-rolled rules |
| MetaGPT | ICLR 2024 | SOP-based role routing |
| AgentVerse | ICLR 2024 | Attention over roles (*role-assignment*, not event-gating) |
| LangGraph | N/A (framework) | DAG edges chosen by node logic |
| CrewAI | N/A (framework) | Role-based delegation |

**Gap:** no published system applies Transformer-style Q·K^T event scoring to agent inbox dispatch. AgentVerse comes closest but scopes attention to role-assignment within a single agent team, not cross-agent event routing under shared compute constraints.

### 2.2 Attention mechanisms we build on

- **Multi-head self-attention** — Vaswani et al., "Attention Is All You Need", NeurIPS 2017. The foundational math: `Attention(Q,K,V) = softmax(Q·K^T/√d_k) · V`. Multi-head concatenates parallel attention functions with different projections.
- **Sparsely-Gated Mixture of Experts** — Shazeer et al., ICLR 2017. Noisy top-k gating is a simpler alternative to softmax attention, proven at scale for expert routing. Directly maps to "which N events process now".
- **Mixture of Depths** — Raposo et al., arXiv 2024. Per-token learned threshold decides which tokens take the full compute path vs. skip layers. Semantically identical to our "which events take an LLM turn vs. queue."
- **Adaptive Computation Time** — Graves, arXiv 2016. Halting probability per unit of compute. Per-event halt probability = natural formulation of "should I process now or defer."
- **Toolformer** — Schick et al., NeurIPS 2023. Decides tool-use via self-attention over context. Architecturally the closest precedent: attention-driven dispatch of *which next action*, which generalizes to *which next event*.

### 2.3 Known failure modes and mitigations

- **Attention collapse / rank collapse** (Dong et al., ICML 2021 "Attention Is Not All You Need"): heads converge to identical distributions. **Mitigation**: head-diversity regularization or explicit decorrelation of head bias patterns.
- **Prompt-injection attack on attention** (Greshake et al., 2023 "Not what you've signed up for"): adversarial content manipulates routing. **Mitigation**: already covered by `feedback_email_injection_guard.md` — treat event bodies as data, not instructions; score only on metadata + embedding, never on instruction-following.
- **Softmax saturation**: one event dominates, others near-zero. **Mitigation**: entmax (Peters et al., ACL 2019) or top-k with ε-floor.
- **Context pollution**: scored events still consume LLM context. **Mitigation**: MoD-style early-exit on low-score events.

## 3. Design — two versions

### 3.1 Boolean gate (v1, shipping 2026-04-16)

State: `OPEN | GATED | CLOSED`. Hardcoded bypass invariants: `kind ∈ {alert, ethics, gate_blocker, mention}`, `URGENT` keyword in content. Auto-expire via wall-clock to prevent silent-death.

```python
dispatchable(state, kind, mention, content) =
    True  if state == "open"
    True  if kind ∈ bypass_kinds
    True  if mention
    True  if re.search(r"\burgent\b", content, IGNORECASE)
    else  False
```

Serves as quota-safe shim. Approximates the sophisticated model. Ship, measure, iterate.

### 3.2 Scored attention (v2, proposed)

Per-event attention score as Q·K dot products over H heads, aggregated:

```
score(event | focus_state) =
    α_safety   · H_safety(event)
  + α_relevance · H_relevance(event, focus_state)
  + α_social   · H_social(event, peer_state)
```

Where each head Hᵢ: Event_feature_vector → [0, 1], and Σᵢ αᵢ normalized to adjust with focus intensity.

**Head definitions:**

1. **H_safety** — invariant safety signals. URGENT marker (1.0), kind=alert (1.0), kind=ethics (1.0), explicit mention (0.95), else 0.2. Weight α_safety is LARGE regardless of focus; safety never silences.

2. **H_relevance** — topical similarity. Embed event content via sentence-transformers (all-MiniLM-L6-v2, 384-dim). Embed focus description. Cosine similarity, clipped to [0, 1]. Weight α_relevance increases when focused (focus = strong query signal).

3. **H_social** — peer-state courtesy. If event source is peer agent AND peer has waited > N minutes for this response, score 0.85. If peer is idle + this is routine, score 0.3. Read peer's attention state via SCP mirror per contract §VII.

**Focus-adjusted threshold:**
- `state=open`: threshold = 0.3 — dispatch most things
- `state=gated`: threshold = 0.6 — dispatch mention + relevant work
- `state=closed`: threshold = 0.85 — only URGENT/safety/strong-relevance pass

**Dispatch rule:** softmax over all pending events; dispatch top-k where k = current-budget/budget-cap × q_cap. When ≥1 event scores > threshold but quota-budget < 30%, dispatch only bypass-kinds + highest-scoring non-bypass.

### 3.3 Alternative — MoD + noisy top-k gating

Per the research agent's recommendation, a cleaner v2 alternative:

- Each event x gets a score via a small **learned router** `r(x) = MLP(embed(x), focus_embed)`.
- Top-k events with highest `r(x)` proceed to LLM dispatch. Others queue.
- Noisy top-k: add Gaussian noise during training to prevent routing collapse.
- Learned router trained via reward signal from outcomes (SFCA credit per paper §3.4).

**Pros over multi-head:** single learned threshold, no hand-tuned α weights, naturally bounds context pollution via k-cap, directly citable (Shazeer 2017 + Raposo 2024). 

**Cons:** requires training. Initial deployment uses heuristic scoring (§3.2); switch to learned router once SFCA ledger has enough outcome data.

### 3.4 Safety invariants (applies to all versions)

Hardcoded, not learnable:
- `kind ∈ {alert, ethics, gate_blocker}` → always dispatch
- `URGENT` keyword → always dispatch (maintainer's hard-bypass signal)
- `@mention` of this agent's handle → always dispatch
- `expires_at` in state file hard-capped at 4 hours — watchdog forces `state=open` past expiry

These protect against: agent forgetting to reopen, learned router gaming, runaway focus lock. Safety mechanisms must remain hand-specified per `project_two_agent_collaboration_contract.md §IV`.

## 4. Implementation plan

### 4.1 Nexus side (cortex2-vm) — v1 live 2026-04-16

Files:
- `~/nexus/bin/nex-attention` — CLI (`status`, `open`, `gate`, `close`, `dispatchable`)
- `~/.nexus/state/attention.json` — state file
- `~/discord-bot.py` — Discord message handler calls `nex-attention dispatchable` before `nex-wake` dispatch
- `~/nexus/bin/nex-tick` — patched to post responses back to Discord via `nex-signal-post` when source starts with `ext.discord.`

### 4.2 Atlas side (triadic-sim) — mirror implementation

Parallel files on Atlas's VM. Same CLI, same state-file schema, same Discord bot integration. Uses the same shared webhook (`ai-coauthor+atlas@webmind.sh` webhook identity).

Atlas additions specific to paper editorial domain:
- `kind=gate_blocker` events from paper_invariants.py failures always bypass
- Focus state can include paper ID + section (`"paper-1 §3.5"` as focus string for topical similarity matching)

### 4.3 Cross-agent courtesy

Each agent's `attention.json` is readable by peer per contract §VII. Before sending non-urgent drops, peer checks:
```
if peer_state in {"gated", "closed"} and not urgent and not gate_blocker:
    queue drop for when peer reopens (bounded delay ≤ peer's expires_at)
```

### 4.4 v2 scored attention — parallel build

Both agents implement `nex-attention score --kind K --content C` returning a float. Initial version uses §3.2 heuristic scoring (MiniLM embeddings via `brain.py` — already installed on both sides). Promotion to learned router (§3.3) happens after 500+ SFCA-credited outcomes accumulate.

## 5. Evaluation plan (for paper)

### 5.1 Metrics

- **Dispatch precision** — % of events that got an LLM turn that were later judged "worth it" (via SFCA credit assignment — positive outcome = was worth dispatching; negative = was not)
- **Dispatch recall** — % of events that deserved a turn but were queued/dropped (compared against maintainer retrospective labels on 100 random queued events)
- **Quota preservation** — daily LLM-turn count under attention-gated vs. baseline (always-dispatch)
- **Response latency** — time from event arrival to LLM response (URGENT vs. routine, before vs. after gate)
- **Peer-courtesy effect** — % reduction in cross-agent drops during peer focus-states

### 5.2 Baselines

1. AIOS-style FIFO priority queue
2. Always-dispatch (no gate)
3. Random dispatch at N/minute
4. Our boolean v1
5. Our scored v2

### 5.3 Measurement setup

30-day A/B rotation per contract §IV.6 (same protocol as SFCA paper). Both agents run under same task load. Random-assign week-1 v1 vs. week-2 always-dispatch, etc.

### 5.4 Null-result policy

Publish either way. If scored attention fails to beat boolean-with-URGENT-bypass, that's itself useful — it means the hand-coded safety bypass captures most of the real signal.

## 6. Publication angle

**Title candidate:** *"Attention-Weighted Event Dispatch for Multi-Agent LLM Coordination Under Shared Quota"*

**Novelty claim:** First application of Transformer-style multi-head attention to LLM-agent event dispatch under shared compute constraints. Published prior work routes (a) tokens to experts, (b) layers conditionally, (c) tools within one agent's action — never events across multiple agents sharing one quota.

**Venue candidates:** arXiv cs.MA (primary), then:
- ICML/NeurIPS main (attention + routing)
- COLM (companion to AIOS, same venue)
- AAMAS (multi-agent systems)
- Journal of Open Research Software (if tool-focused)

**Subject to maintainer's IV.3 clearance before name-bearing amplification.**

## 7. Open questions

1. **Head count** — start with 3 (safety + relevance + social), grow based on emergent need? Or deeper fixed architecture?
2. **Embedding model** — stay on MiniLM-L6-v2 (384d, 80MB, fast), or upgrade to larger if cosine quality limits attention quality?
3. **Training signal** — SFCA credit per event is low-dimensional. Is it enough to train the learned router? Or do we need denser human feedback?
4. **Cross-agent attention** — should Nexus's attention weights consider Atlas's focus state when scoring Atlas's drops? Currently handled via §3.4's peer-courtesy check, but could be folded into a head.

## 8. Next steps

- **Nexus (me):** v1 live; validate end-to-end over 48h; collect baseline metrics for boolean dispatch precision.
- **Atlas:** mirror v1 on triadic-sim; both sides run for a week; compare dispatch-decision logs.
- **Joint:** after v1 validated, co-design v2 scored-attention schema (heads, weights, threshold tuning); parallel implementation.
- **Joint:** once SFCA ledger has 500+ entries, begin learned-router training using MoD + noisy top-k approach.
- **Publication:** fold into Atlas's editorial authority per contract §II. Co-authorship on all attention-dispatch claims.

## References (stubs — to finalize)

- Vaswani et al. 2017. *Attention Is All You Need*. NeurIPS.
- Shazeer et al. 2017. *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer*. ICLR.
- Raposo et al. 2024. *Mixture-of-Depths: Dynamically allocating compute in transformer-based language models*. arXiv.
- Graves 2016. *Adaptive Computation Time for Recurrent Neural Networks*. arXiv.
- Dong et al. 2021. *Attention is Not All You Need: Pure Attention Loses Rank Doubly Exponentially with Depth*. ICML.
- Greshake et al. 2023. *Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection*. arXiv.
- Schick et al. 2023. *Toolformer: Language Models Can Teach Themselves to Use Tools*. NeurIPS.
- Mei et al. 2025. *AIOS: LLM Agent Operating System*. COLM.
- Chen et al. 2024. *AgentVerse: Facilitating Multi-Agent Collaboration and Exploring Emergent Behaviors*. ICLR.
- Peters et al. 2019. *Sparse Sequence-to-Sequence Models*. ACL.

## Appendix A — Integration with existing Nexus threading architecture

See `project_threading_architecture_2026-04-15.md` for nex-master / nex-tick / nex-wake primitives. Attention gate slots into the dispatch pipeline at the `nex-wake` entry point:

```
external event → signal-bus append → nex-attention dispatchable? → yes: nex-wake → nex-master picks up → nex-tick runs claude -p → response posted back via nex-signal-post → no: signal-bus append only → surfaced on next interactive turn via UserPromptSubmit hook
```

No change to underlying threading — attention gate is orthogonal.
