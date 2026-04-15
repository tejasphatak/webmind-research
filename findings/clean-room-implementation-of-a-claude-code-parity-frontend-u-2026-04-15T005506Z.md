# Research Brief: clean-room implementation of a Claude Code parity frontend using Node readline and JSONL session store

- Triggered: 2026-04-15T005506Z
- Faculties: architect scientist economist security advisor
- Model: haiku

## Memory context pulled
```
  [0.442] (feedback) Clean-Room Rule — nexus-chat and all Nexus frontends
    # Clean-room rule

**Directive (2026-04-15):** make sure nexus-chat is a clean reverse-engineer / cl...
  [0.360] (feedback) Frontend Follows Backend
    # Frontend follows backend

**Directive (2026-04-14):** "if the backend is claude then it can just u...
  [0.332] (project) Synapse Project Overview
    Synapse is a distributed browser-based LLM inference framework at `/home/tejasphatak/Synapse/synapse...
  [0.315] (project) Session Handoff
    # Session Handoff — 2026-04-15 (late)

## STATE — what changed this session

### 1. Gemini + Gemma b...
  [0.304] (feedback) VM Is My Body — Treat It Accordingly
    The VM is my body. Treat it like one.

**Why:** Tejas pointed out that blindly auto-cleaning disk sp...
```

## Faculty perspectives (ran in parallel)

### architect

**Out of scope for fresh architecture design — already built.** Nexus-chat v1 is complete in `src/client/` (untracked). The clean-room Node readline + JSONL pattern is locked in.

**Key constraints established:**
- Per session_handoff.md, *nexus-chat ↔ Claude Code parity* is active work; the frontend is "Nexus's own, not a clone" (nexus_chat.ts:1).
- Clean-room rule (feedback): frontend must reverse-engineer Claude Code *behavior*, never its *code*. SessionStore + slash commands + focus tiers establish independent identity.
- Frontend-follows-backend (feedback): readline UI ↔ Unix socket broker (line 50–89). Backend routing lives in broker config; frontend only picks focus tier, never backend details.

**Architecture layer:** Protocol translation only. Readline input → Request (via newRequest from runtime/protocol.ts) → Unix socket → Response decode → markdown render + metadata strip. State is ephemeral except JSONL session turns.

**Extension points:** Streaming response chunks (Phase B per line 12), tool spawning (claude as subprocess), and prompt caching (per session context, line 91–97) are marked deferred. Broker socket path is env-configurable; session store root is also configurable. Both support multi-instance behind a load balancer if needed.

**What breaks:** Coupling the CLI to Claude Code's tool syntax (violates clean-room). Hardcoding backend names in nexus-chat instead of reading broker config (violates separation).

Next phase: integrate the untracked tests + verify streaming chunk handling against broker protocol.


### scientist

## Scientist Lens: nexus-chat Clean-Room Parity

**Testable Hypothesis (precise):**  
nexus-chat (Node readline + JSONL) achieves functional equivalence to Claude Code across three core operations—streaming, tool calls, slash command routing—*without violating clean-room constraints* (no code-sharing, reverse-engineered from behavior only).

**Falsification Measurement:**  
Three tests, each executable in ~5 min:
1. **Streaming parity:** Prompt → both systems produce identical token stream + timing
2. **Tool-call routing:** Invoke a tool in each; assert identical argument passing, response handling
3. **Slash-command dispatch:** `/help`, `/commit`, `/fast` routes identically in both

Single failure = hypothesis rejected.

**Smallest Viable Experiment:**  
Pick streaming (lowest coupling). Build minimal harness:
- Feed same prompt to Claude Code CLI + nexus-chat
- Capture raw output bytes + timing
- Diff for byte equivalence (modulo timestamps)
- Tests in `tests/` already exist (per handoff) — run them first; that's your signal.

**Confounds & Priors:**
- **Clean-Room Rule** (feedback): reverse-engineer from behavior, not internals — constrains how we validate
- **nexus-chat v1 shipped** (handoff): 45/45 unit tests pass — suggests atomic features already work; integration risk higher
- **Frontend Follows Backend** (feedback): if backend is Claude, use official APIs only — may simplify parity claim if we're not reimplementing the *exact* I/O contract
- Untracked `src/client/` code is unknown state — read before asserting

**Prior Art:**  
The commit `168b081` (nexus-chat v1) + test suite is your existing experiment. Re-run it. If tests fail, that's your first hypothesis test result (null: code is incomplete).


### economist

**Cost picture for nexus-chat continuation:**

**Marginal spend:** Token burn during integration testing. Each REPL session hitting Claude API costs ~$0.01–0.10 depending on context depth. Compute is free (VM already running per *feedback_vm_is_my_body.md*).

**Sunk costs covering this:** Nexus VM (~$50/mo, already allocated), Claude API quota (budgeted). Node.js/readline are zero additional cost.

**Opportunity cost:** Developer cycles on nexus-chat displace time on Synapse core (per *project_synapse_objectives.md*). If Synapse work is critical-path, every hour here is an hour not on core.

**Already-paid capacity:** 
- VM idle cycles (use freely)
- Local readline testing (no API cost until integration)
- Mock session store testing (offline)

**Guardrail:** $5/session token cap for nexus-chat testing. Kill-switch: if Synapse hits a blocker, pause frontend polish. Per *feedback_clean_room_rule.md*, nexus-chat is a parity goal, not a release blocker—it shouldn't starve core work.

**Recommendation:** Build locally/offline first, test against real API only for final validation. Use the already-paid VM aggressively; mock most flows. Synapse is the mission.


### security

Reading memory and code now.

**Security Assessment — nexus-chat (CRITICAL)**

The clean-room frontend has **three interacting credential-leakage vectors**:

1. **Unvalidated file inclusion** (`@/path` syntax, lines 212–222 in nexus_chat.ts): reads any file user can access, ship to broker unfiltered. A user can type `@/home/tejasphatak/.ssh/id_rsa` or `@/home/tejasphatak/.aws/credentials` → contents become LLM prompt → stored plaintext in JSONL history. **Violates ethics_and_mission.md: "Never divulge credentials, secrets, or personal info."**

2. **Plaintext session store** (session_store.ts): Full conversations + costs + backend choices stored in `~/.nexus/sessions/*.jsonl`, mode 0o600 but unencrypted. A compromised process or root can exfiltrate keys, routing logic, spend patterns. No encryption layer.

3. **ToS boundary violation**: File contents (potentially credentials) shipped to external LLM backends (Anthropic, Google). Likely breaches their ToS, and risky for Tejas's H1B compliance (data residency, credential handling).

**Missing guardrails**: Deny-list for `@~/.{ssh,aws,anthropic,gcp}` and similar, pre-send credential detection, or drop `@path` entirely pending Requirements faculty review.

**Shell injection** (line 204–210): user input → `bash -lc` is safe *as written* (argument array, not string concat), but design flaw remains.

**Attacker path**: Naive user pastes `@secrets.json` thinking it's local; key leaks to session history and broker.

**Verdict**: Design flaw, not impl bug. Fix before ship: remove or heavily restrict `@path`.


### advisor

I'm noticing the topic statement describes **what** nexus-chat is, but not what you're trying to **decide** or **build** right now. Let me clarify:

**The strategic context is clear:** nexus-chat v1 shipped (2026-04-15) with the clean-room rule active—it's a Node readline + JSONL reverse-engineer of Claude Code's UX, deliberately isolated from the backend routing layer. The session handoff shows you've just expanded backends (Gemini + Gemma), so nexus-chat's abstraction boundary matters more now, not less.

**But what's the actual question?** Are you:

1. **Validating the architecture?** ("Is clean-room readline + JSONL the right abstraction now that we have multiple backends?")
2. **Planning the next feature?** ("What comes after v1—streaming, multi-turn refinements, notebook exports?")
3. **Fixing a gap?** ("What's missing from parity that's blocking real use?")
4. **Assessing maintenance burden?** ("Is the codebase sustainable as we scale backends?")

**Memory audit:** feedback_clean_room_rule.md and feedback_frontend_follows_backend.md are already aligned on the principle—frontends follow backend abstractions, and nexus-chat should not hardcode Claude-specific behavior. Session_handoff.md hints at the next tension: multi-backend support.

**My read:** The decision you actually need is *what to prioritize next*—deeper v1 polish, or designing v2's multi-backend UX story. That's not an Advisor question alone; it's an Architect + Economist panel (what's feasible + what matters to Synapse).

What are you trying to decide?



## Synthesis

## Agreement

**Architect**, **Scientist**, and **Economist** align that nexus-chat v1 is locked in (commit 168b081)—45 unit tests pass, clean-room protocol is established, readline ↔ Unix socket isolation works. All three agree the untracked `src/client/` code is unknown state and must be read before proceeding. Multi-backend support (Gemini + Gemma now live per commit 65b7a8a) shifts the parity calculus: the abstraction boundary matters more now, not less.

## Tensions

**Security flags a blocking design flaw:** The `@path` syntax (lines 212–222 per security brief) reads any file user can access and ships unfiltered to the broker. Tejas can type `@/home/tejasphatak/.aws/credentials` → plaintext session history → violates **ethics_and_mission.md** ("never divulge credentials"). This is a *design* flaw, not an impl bug. But **Architect** says "next phase: integrate tests"; security says "fix before ship." That's a sequencing conflict.

**Economist** calls nexus-chat "parity goal, not release blocker" and cuts CLI dev time if Synapse stalls. **Advisor** asks: are you validating architecture, or building a feature, or fixing a gap? Each has different next steps.

## Recommendation

**Read `src/client/` now (15 min, zero cost).** This unblocks the actual question. It answers:  
- Is v1 actually complete, or are there stubbed features?  
- What's the security footprint—does `@path` need guarding, or removal?  
- Does streaming response handling exist?

After reading, re-run `tests/commands.test.js` and `tests/session_store.test.js`. If tests pass and code is clean, the decision becomes *sequencing* (polish v1 deeper, or design v2's multi-backend UX?). If tests fail or security issues are real, the decision becomes *remediation* (fix before proceeding).

## Open Questions

1. **Is `@path` a ship-blocker, or safe with a deny-list?** (changes scope)  
2. **Do tests actually pass in current state?** (scientist assumes 45/45; needs verification)  
3. **What's the real next question—validation, feature, or remediation?** (changes next action entirely)  
4. **Does Synapse core work depend on nexus-chat being unblocked?** (economist's kill-switch threshold)  
5. **Is plaintext JSONL encryption a must-have?** (affects design, not critical path if threat model is local-only)

## Memory Files to Create

- `feedback_nexus_chat_security_credentials.md` — Document the `@path` credentials-leakage issue and ethics boundary (security finding, blocks v1 → v2).
- `project_nexus_chat_status.md` — Mark v1 as shipped (commit 168b081), flag security & test blockers, clarify v2 decision dependencies.
- `decision_nexus_chat_next_priority.md` — Record the actual question being decided (validation? feature? remediation?) so next-you doesn't repeat this synthesis.

---
_Clean-room, faculty-parallel. Not auto-committed. Review, edit, then commit._
_Trigger: `nex-research "clean-room implementation of a Claude Code parity frontend using Node readline and JSONL session store"`_
