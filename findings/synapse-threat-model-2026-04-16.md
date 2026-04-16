# Synapse Threat Model & Security Roadmap

**Date:** 2026-04-16
**Author:** Claude Opus 4.6 (Synapse Research), reviewed by Gemini 3.1 Pro
**Scope:** Synapse (webmind.sh) — decentralized LLM inference via volunteer WebGPU devices
**Extended scope:** Nexus trajectory (continuously self-learning distributed model)

---

## System Architecture (for threat analysis)

```
User → Coordinator → [Shard 0: Device A] → [Shard 1: Device B] → ... → [Shard N: Device K] → Response
                          ↑                       ↑                          ↑
                   Volunteer browser        Volunteer browser         Volunteer browser
                   (WebGPU, untrusted)      (WebGPU, untrusted)      (WebGPU, untrusted)
```

- Model: Gemma 3 1B IT, sharded across volunteer devices
- Transport: WebGPU compute, activations serialized between shards
- Trust boundary: Coordinator is trusted; all volunteer devices are untrusted

---

## Threat Enumeration

### T1 — Byzantine Activation Poisoning (CRITICAL)
**Attacker:** A malicious volunteer device
**Action:** Returns corrupted activations at shard boundary (garbage, biased, or subtly shifted)
**Impact:** Incorrect output served to user. In Nexus trajectory: poisoned training signal.
**Detectability today:** NONE. No verification mechanism exists.
**Blast radius:** Single query (Synapse); potentially entire model (Nexus, if training on poisoned inference)

**Subtypes:**
- T1a: Random noise injection — easy to detect via statistical checks
- T1b: Targeted logit manipulation — attacker shifts specific token probabilities (e.g., makes model recommend a product, inject URLs, flip factual claims)
- T1c: Stealth drift — micro-perturbations below fp16 noise floor. Undetectable by Freivalds. Cumulative effect over many queries could bias model behavior.

### T2 — Sybil Attack on Volunteer Pool
**Attacker:** Entity controlling many volunteer devices
**Action:** Floods network with colluding devices; gains control of entire inference pipeline for targeted users
**Impact:** Full control of model output for targeted queries
**Detectability today:** None. No reputation system exists.
**Nexus impact:** Massive — attacker controls training data distribution

### T3 — Model Extraction via Activation Interception
**Attacker:** Any volunteer device
**Action:** Records all activations passing through their shard. Over many queries, reconstructs model weights or functional equivalent.
**Impact:** IP theft (if model is proprietary); privacy breach (if activations encode user data)
**Note:** For Gemma 3 1B (open-weights), IP theft is moot. But for Nexus (self-learned model), this becomes critical.

### T4 — Prompt/Data Extraction from Activations
**Attacker:** Any volunteer device
**Action:** Inspects activations to infer user's prompt or private data
**Impact:** Privacy violation
**Severity:** HIGH for sensitive queries (medical, legal, financial)
**Mitigation difficulty:** Fundamental tension — you must share activations for inference to work

### T5 — Denial of Service / Straggler Attack
**Attacker:** Volunteer device
**Action:** Accepts shard assignment, then responds slowly or not at all
**Impact:** Query latency spikes; coordinator must timeout and reassign
**Detectability:** Easy (timeout)
**Mitigation:** Redundant shard assignment + fastest-response-wins

### T6 — Coordinator Compromise
**Attacker:** External attacker targeting the coordinator
**Action:** Takes over coordinator; can route queries to attacker-controlled shards, log all traffic, inject responses
**Impact:** Total system compromise
**Note:** Out of scope for volunteer trust model; standard server-hardening applies

### T7 — Numerical Divergence as Attack Surface
**Attacker:** Passive (heterogeneous hardware)
**Action:** Different WebGPU backends (Mali, Adreno, Apple, NVIDIA) produce different fp16 rounding
**Impact:** Non-deterministic outputs across queries; makes verification harder (attacker hides behind legitimate rounding noise)
**Status:** Under active measurement (Artifact 1 — numerical fidelity study)

### T8 — Training-Time Data Poisoning (NEXUS ONLY)
**Attacker:** Malicious volunteer
**Action:** In Nexus, inference traffic feeds back into training. Attacker crafts inputs + activations to poison learned weights.
**Impact:** Model degradation, backdoor insertion, bias injection
**Severity:** EXISTENTIAL for Nexus — this is the #1 threat to the self-learning trajectory
**Mitigation:** Requires gradient-level verification, not just activation-level. Much harder than T1.

---

## Current Defenses

| Threat | Current defense | Gap |
|---|---|---|
| T1 Byzantine activations | None | Critical gap |
| T2 Sybil | None | Critical gap |
| T3 Model extraction | N/A (open weights) | Becomes critical at Nexus |
| T4 Prompt extraction | None | Needs research |
| T5 DoS/straggler | Likely timeout | Needs redundancy |
| T6 Coordinator compromise | Standard infra | Adequate for now |
| T7 Numerical divergence | Under measurement | Pending Artifact 1 |
| T8 Training poisoning | N/A (inference only) | Nexus blocker |

---

## Prioritized Roadmap

### Phase 1 — Synapse (current, inference only)

**P1.1: Probabilistic activation verification (Freivalds-based)**
- Coordinator sends random probe vectors to verify shard outputs
- Catches T1a (random noise) and T1b (targeted manipulation) with tunable false-positive rate
- Does NOT catch T1c (stealth drift below fp16 noise floor)
- Estimated dev effort: 2-4 weeks
- Paper: "Probabilistic Verification of Distributed LLM Inference via Freivalds Probes"

**P1.2: Reputation system**
- Track per-device verification pass rate over time
- Devices that fail probes get deprioritized; persistent failures get banned
- Catches T2 (Sybil) over time — new devices start with low trust, must earn it
- Estimated dev effort: 1-2 weeks

**P1.3: Redundant shard assignment**
- For high-value queries, assign same shard to 2+ devices, compare outputs
- Catches T1 at the cost of 2x compute
- Also solves T5 (straggler) — fastest response wins
- Estimated dev effort: 1 week

**P1.4: Activation compression at shard boundaries**
- Carrier + Payload (this research) reduces bandwidth requirements
- Side benefit: compressed representation is harder to poison precisely (attacker must corrupt in PCA space, not raw activation space)
- Estimated dev effort: 2-3 weeks after Artifact 1 validates feasibility

### Phase 2 — Pre-Nexus (before enabling self-learning)

**P2.1: Differential privacy on activation transport**
- Add calibrated noise to activations before transmission
- Trades inference quality for privacy (T4 mitigation)
- Quantify the quality-privacy Pareto frontier

**P2.2: Secure aggregation for gradient updates**
- When Nexus begins learning from inference, use secure aggregation (federated learning literature) to prevent any single volunteer from seeing or controlling the full gradient
- Blocks T8 (training poisoning) from single actors; colluding majority is still a risk

**P2.3: Anomaly detection on incoming gradients**
- Byzantine-resilient aggregation (Krum, trimmed mean, coordinate-wise median)
- Established literature from federated learning; adapt to Nexus topology

### Phase 3 — Nexus (continuously self-learning)

**P3.1: Verifiable computation via optimistic proofs (opML-style)**
- Full algebraic verification of computation trace
- Expensive but provides cryptographic guarantees
- Only for high-value inference or training checkpoints

**P3.2: TEE enclaves for critical shards**
- Run the most sensitive layers (embedding, final logit projection) inside Trusted Execution Environments
- Hybrid: TEE for trust-critical layers, volunteer WebGPU for bulk compute
- Hardware dependency (Intel SGX, ARM TrustZone) limits volunteer pool

---

## Comparison: Synapse vs Prior Art

| System | Trust model | Verification | Bandwidth | Privacy |
|---|---|---|---|---|
| **Synapse (current)** | None (trust all) | None | Full fp16 | None |
| **Synapse (proposed)** | Reputation + Freivalds | Probabilistic | Carrier+Payload compressed | Future (DP) |
| **Petals** | Reputation + spot-check | Heuristic | Full fp16 | None |
| **opML** | Fraud proofs (bisection) | Deterministic | Full | None |
| **ZKML (EZKL, etc.)** | Zero-knowledge proofs | Cryptographic | Full | Partial |
| **Federated Learning** | Secure aggregation | Statistical | Gradients only | DP/SecAgg |

Synapse's niche: **lightweight probabilistic verification + compression for bandwidth-constrained volunteer devices**. Not trying to be cryptographically complete (too expensive for WebGPU); trying to be economically secure (cheating costs more than it gains).

---

## Positioning vs ZKML and TEEs

**Why not ZKML:** Zero-knowledge proofs for neural network inference (EZKL, Modulus Labs) provide cryptographic guarantees but with 100-10000x overhead. On volunteer WebGPU devices with limited compute, this is a non-starter. Synapse's contribution is showing that probabilistic verification achieves practical security at 1-5% overhead.

**Why not TEEs:** Trusted Execution Environments (SGX, TrustZone) require specific hardware. Synapse's value proposition is "any WebGPU browser can contribute." Mandating TEE support shrinks the volunteer pool to near-zero. Hybrid model (TEE for critical shards only) is the pragmatic compromise for Phase 3.

**The Pareto argument:** Synapse operates on a different point of the security-cost-accessibility frontier. ZKML and TEEs sacrifice accessibility for security; Synapse sacrifices security guarantees for accessibility, compensating with economic/reputation mechanisms. For the target users (schools, clinics, low-resource), accessibility wins.

---

## Key Unknowns

1. **How much activation information leaks prompt content?** No published study quantifies this for pipeline-parallel inference specifically. Needed before making privacy claims.
2. **Does carrier+payload compression affect attack surface?** Compressing activations changes what an attacker can manipulate. PCA basis is public (shared prior); only the residual crosses the wire. Does this help or hurt?
3. **What's the minimum reputation history for Sybil resistance?** Depends on volunteer churn rate, which isn't measured yet.
4. **Can Freivalds probes be predicted by a sophisticated attacker?** If the attacker knows when probes happen, they can pass probes while poisoning non-probe queries. Probe scheduling must be unpredictable (random, not periodic).

---

## Recommended Reading

- Garg et al. (2017) "SafetyNets: Verifiable Execution of Deep Neural Networks on an Untrusted Cloud"
- opML: https://docs.opml.io/
- Blanchard et al. (2017) "Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent"
- Petals: https://petals.dev/ — closest architectural comparison
- SmoothQuant / LLM.int8! — outlier handling that affects verification thresholds
