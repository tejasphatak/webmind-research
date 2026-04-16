# Round 3 — Under-Saturated Synapse-Specific Axes (Claude)

**Date:** 2026-04-16
**Strategy:** Instead of weird-faculty random ideation (low hit rate), target axes where prior art is sparse and Synapse has unique positioning.

---

## Axis 1: Client-Side Privacy for Sensitive Queries

**The problem:** A medical question to a healthcare LLM must not leak to untrusted volunteer nodes. Synapse routes activations through strangers' browsers. This kills adoption for sensitive use cases.

### Idea 1.1: **Prompt Splintering**
Split the user's prompt into semantically neutral pieces, route each through a different volunteer, recombine only at coordinator. Individual shards see "what is the" or "treatment for" but no volunteer sees the full query. Requires a recombiner that reconstructs final output without ever holding the full prompt in any single location.

**Synapse-specific angle:** Current decentralized inference literature assumes everyone sees everything. Prompt splintering requires carefully designed local computation boundaries. Not yet published for LLM inference.

**Feasibility check:** The hard part is that attention needs full context. A splinter that sees only "what is the" can't attend over the full query. So splintering works only if each splinter runs independently and outputs are merged without full-context attention. That limits model behavior. Maybe viable for RAG-style queries (retrieve first, then generate).

### Idea 1.2: **Decoy Query Batching**
User's real query is submitted alongside K-1 synthetic queries that look semantically similar. Volunteers see K queries, don't know which is real. Statistical privacy — adversary with partial view can't distinguish real from decoy.

**Synapse-specific angle:** Mixnet analog for LLM inference. Not published.

**Feasibility:** Costs K× compute but is technically simple. For K=5, we get meaningful privacy at 5× cost.

### Idea 1.3: **Homomorphic Activation Obfuscation**
Before sending activations through the pipeline, user's local device applies a LEARNED linear transformation known only to the user and coordinator. Volunteers see transformed activations that still produce correct downstream computation (linear transformations commute with matmul up to inverse-transformation at coordinator). Pre-commits the user's transformation matrix.

**Synapse-specific angle:** Lightweight homomorphic-ish without full FHE overhead.

**Feasibility check:** Only works if the transformation commutes with nonlinearity. LayerNorm is per-row so additive shifts cancel; but GELU doesn't commute with arbitrary linear. So this would require structured transformations (scale-only, or rotations within known-invariant subspaces). **Worth exploring.**

---

## Axis 2: Cold-Start Bootstrapping

**The problem:** A new Synapse deployment has zero volunteers. No volunteers = bad service = no users = no volunteers. Chicken-and-egg.

### Idea 2.1: **Sponsor Mode**
Early deployment has a small paid compute tier (e.g., a single GCP A100). As volunteer count grows, paid tier is proportionally reduced. Smooth bootstrap from centralized to decentralized.

**Synapse-specific angle:** Honest deployment story for reaching scale.

**Feasibility:** Standard hybrid cloud, but formalizing the transition policy is a real contribution. What's the OPTIMAL paid→volunteer ramp?

### Idea 2.2: **Progressive Decentralization Protocol**
Start with all computation at coordinator. As volunteers join, gradually offload shards in order of least-trust-required first. Last to decentralize: model-output sampling (keep centralized for consistency).

**Synapse-specific angle:** Answers "at what point is my decentralized Synapse safe to use?" with a concrete readiness score.

### Idea 2.3: **Cross-Project Volunteer Pool Sharing**
Multiple open-source LLM projects (Synapse, Petals, possibly future ones) share a common volunteer pool protocol. Volunteers register once, participate across projects. Reduces bootstrap cost per project.

**Synapse-specific angle:** Network effects across the decentralized AI ecosystem. Likely novel as a PROTOCOL (individual volunteer-pool tokens exist in Web3 but not for inference-specific workloads).

---

## Axis 3: Heterogeneous Query Routing Market

**The problem:** Device pool from 2020 iPhone to RTX 5090. Query complexity varies. How to match?

### Idea 3.1: **Query-Capability Affinity Score**
Learned embedding: each query embeds into a capability-requirement vector (latency-sensitive? memory-heavy? precision-critical?). Each volunteer has a capability-offer vector. Match via cosine similarity.

**Synapse-specific angle:** Learned matching in heterogeneous device pools. Standard in ad-tech, not yet published for inference routing.

### Idea 3.2: **Auction-Based Shard Assignment**
Each query is auctioned to the volunteer pool. Bids = {predicted latency, cost-credit, reputation}. Winner runs the shard. Variable-rate second-price auction ensures truthful bidding.

**Synapse-specific angle:** Classical mechanism design, specific application to LLM inference is novel.

---

## Axis 4: Catastrophic Failure Recovery

**The problem:** 30% of volunteers vanish mid-query (ISP outage, carrier failure, power). System must degrade gracefully.

### Idea 4.1: **Shard-Hot-Migration**
Each volunteer periodically snapshots its KV state to 2-3 other volunteers. If it drops, a snapshot-holder takes over. Similar to database warm standbys.

**Synapse-specific angle:** Applied to mid-inference KV state — not just query-level retry. Novel.

### Idea 4.2: **Graceful Degradation to Shorter Context**
If enough volunteers drop that full context can't be attended, auto-truncate to the last K tokens most needed (via attention score). User sees a "degraded answer" warning but query completes.

**Synapse-specific angle:** Graceful degradation policy for inference-time failures. Published in general systems engineering, novel application here.

### Idea 4.3: **Query Replay Against Shadow Fleet**
Maintain a small shadow fleet (5% of active). If a query fails, replay against shadow; compare outputs. Safety net.

---

## Axis 5: Incentive Mechanisms

**The problem:** Why would anyone leave a browser tab open?

### Idea 5.1: **Compute Credits → LLM Access**
Volunteers earn credits proportional to compute contributed. Credits spend on high-priority queries (faster inference), bigger models, or longer context. Self-reinforcing ecosystem.

**Synapse-specific angle:** Service-for-service exchange. Novel for LLM inference (some projects use this for storage / compute).

### Idea 5.2: **Attribution-Based Karma**
If a volunteer's compute contributed to an answer the user upvoted, volunteer earns karma. Public leaderboard. Social incentive.

**Synapse-specific angle:** Maps volunteer contribution to user-visible outcomes. Not published for LLM inference.

### Idea 5.3: **Volunteer-Specific Models**
Each volunteer's contribution history determines what models/features are unlocked for them when they're the querier. "You contributed 10 hours of compute; unlock GPT-4-class queries."

**Synapse-specific angle:** Feature-gating by contribution. Some gaming precedents but novel for LLM.

---

## CLAUDE'S TOP PICKS (pre-validation)

From this round, ranked by (Synapse-specificity × potential novelty × feasibility):

1. **Prompt Splintering for privacy** (Axis 1.1) — genuinely novel, Synapse-unique, feasibility uncertain
2. **Homomorphic Activation Obfuscation via invariant subspaces** (Axis 1.3) — clever, testable
3. **Progressive Decentralization Protocol with readiness score** (Axis 2.2) — operational, publishable
4. **Query-Capability Affinity Score** (Axis 3.1) — matching/routing paper angle
5. **Shard-Hot-Migration via KV snapshot peers** (Axis 4.1) — graceful failure
6. **Compute Credits → LLM Access** (Axis 5.1) — Web3-free incentive mechanism

Next step: cross-verify against literature. Expected higher hit rate than weird-faculty rounds.

---

*Continuing autonomously. Gemini Round 3 running in parallel. Will synthesize.*
