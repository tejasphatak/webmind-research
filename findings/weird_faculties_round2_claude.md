# Weird Faculty Ideas — Round 2 (Claude)

**Date:** 2026-04-16
**Mode:** Autonomous. Push for higher novelty via more technical weird disciplines.

---

## 1. Radio Astronomer / VLBI (Very Long Baseline Interferometry)

**Framing:** Synapse is an interferometer. Each volunteer is an antenna. Coherent combination of their noisy, distributed measurements produces a sharp final output.

**Ideas:**
- **VLBI-style activation correlation.** Multiple volunteer nodes compute the SAME shard on the same input. Correlate their outputs in the complex phase space (treat fp16 activations as phasors). In-phase components reinforce (the correct answer); out-of-phase cancel (Byzantine injection). More nodes = sharper signal, Byzantine elimination by phase mismatch.
- **Fringe-stopping for slow nodes.** In VLBI, distant antennas see the same source at slightly different times; fringe stopping compensates. Apply: volunteers with different latencies contribute to the same activation with learned time-offset corrections.
- **Self-calibration against reference source.** Radio astronomers periodically point at a known calibrator. Synapse could periodically inject a known test query with known activations; nodes self-calibrate their precision against it.

**Why it might work:** VLBI routinely combines data from antennas thousands of km apart at nanosecond precision. The problem of combining noisy distributed measurements is deeply solved — just untapped in ML inference.

---

## 2. Fighter Pilot / F-22 Wingman Tactics

**Framing:** Each query is a mission. Volunteers are aircraft. Coordinator is AWACS. Network conditions are weather. Byzantine nodes are enemy spoofers.

**Ideas:**
- **Track-while-scan silent speculation.** F-22s can track targets using off-board AWACS data without emitting their own radar (stays stealthy). Analog: volunteers can speculate on activations using only cached context, without requesting new data from the network — reduces bandwidth until a full track is needed.
- **Wingman-pair redundancy.** F-22s fly in 2-ship or 4-ship formations; lead + wingman cover each other. Pair volunteers: one runs the shard, one verifies. Cheap redundancy where wingman compute is ~10% of lead.
- **Burst-transmit for spoofing resistance.** Emit heavy bandwidth data in short bursts (hard to jam). Synapse: transmit activation bundles in scheduled bursts rather than continuously. Adversary can't opportunistically inject.

**Why it might work:** Air combat software has solved real-time distributed tracking under adversarial conditions. Transfers cleanly to Byzantine-tolerant inference.

---

## 3. Sushi Chef (freshness decay)

**Framing:** Activations are fish. The moment they leave the knife, their usefulness starts decaying. Every millisecond of storage costs quality. Just-in-time compute.

**Ideas:**
- **Freshness-aware KV-cache eviction.** Instead of LRU, evict by "staleness × semantic-distance-from-current-query." Cached activations that are OLD AND irrelevant go first.
- **Same-counter service.** Top sushi bars serve from the counter immediately — no delay, no freezing. Apply: dedicated "hot" shards with zero KV cache, only live computation, for latency-sensitive queries. Other shards handle cache-heavy workloads.
- **Ingredient rotation schedule.** Sushi bars rotate stock so freshness is guaranteed. Apply: rotate which volunteers hold which shards weekly/daily. Prevents any single volunteer from accumulating privileged data over time.

**Why it might work:** Freshness-aware systems exist in finance (tick data) but not in LLM inference. Might meaningfully improve cache efficiency.

---

## 4. Submarine Warfare / SOSUS Acoustic Network

**Framing:** Synapse is a passive sonar array. Volunteers are hydrophones. The "enemy" (Byzantine nodes) is a hostile submarine trying to hide in the noise. Detection is statistical, not deterministic.

**Ideas:**
- **Matched filter for Byzantine signatures.** SOSUS used matched filters to detect specific submarine acoustic signatures. Train lightweight matched filters on known Byzantine attack patterns (random noise, biased shift, targeted flip). Fast, cheap detection.
- **Beamforming for activation direction-of-arrival.** Multiple volunteers reporting activations for the same query — combine to estimate the "origin direction" (which shard is inconsistent). Byzantine detection via DOA mismatch.
- **Burst-transmit / LPI protocols.** Submarines transmit in short low-probability-of-intercept bursts. Shards could batch their activations into compressed bursts for the coordinator, reducing traffic analysis surface.

**Why it might work:** Sonar network theory solved many problems Synapse will face — signal integration under adversarial conditions.

---

## 5. Surveyor / Cartographer (triangulation + error propagation)

**Framing:** Each shard is a survey point. Inference is triangulating the final answer from many imperfect measurements. Errors accumulate linearly or worse.

**Ideas:**
- **Baseline network calibration.** Surveyors establish a baseline of known points (NAVD88 benchmarks). Apply: maintain a small set of "benchmark queries" whose correct activations are precisely known; use to calibrate every volunteer.
- **Closed traverse adjustment.** Survey traverses close back to the start for error check. For Synapse: periodically run queries that return to the same state (idempotent), check if the pipeline returns the known state. Drift = problem.
- **Height-above-datum normalization.** Heights are stored relative to a datum (e.g. ellipsoid). Apply: normalize activations against a shared statistical baseline (running mean of all volunteers) so drift is detectable.

**Why it might work:** 200+ years of surveying math is all about error-tracking distributed measurements.

---

## 6. Structural Engineer / Earthquake Resilience

**Framing:** The pipeline is a building. Volunteers are structural members. Byzantine attacks are earthquakes. We design for graceful failure, not perfect prevention.

**Ideas:**
- **Base isolation analog.** Buildings use rubber bearings to decouple structure from ground motion. Synapse: add a "buffer layer" between volunteer output and downstream consumption — activations pass through an averaging/smoothing stage that damps Byzantine bursts before they propagate.
- **Redundant load paths.** Structural engineering requires multiple paths so any single failure doesn't collapse the building. Synapse: every critical shard has ≥2 active volunteers; one is the "primary" for compute, the other is a "hot spare." Standard practice in datacenters, but apply specifically for Byzantine detection: spare's output cross-checks primary's.
- **Progressive collapse prevention.** Designed so a local failure doesn't cascade. Synapse: contain failure to the shard level by isolating each shard's inputs/outputs behind a firewall of Byzantine-check. One bad shard doesn't poison downstream.

**Why it might work:** Civil engineering is the oldest form of systems-resilience engineering. Building codes encode hard-won lessons.

---

## 7. Potter / Kiln Engineer

**Framing:** Each query is a pot. The full forward pass is the firing. Quality rejection happens in discrete phases (bisque, glaze, final fire). Each phase has its own quality criterion.

**Ideas:**
- **Phase-gated verification.** A kiln fires in stages — dry, bisque, glaze. Each is checked separately. Apply: Freivalds verify at shard boundaries AND at model-layer boundaries (post-attention, post-FFN). Finer-grained detection.
- **Kiln atmosphere control.** Reduction vs oxidation atmospheres produce different glaze colors. Analog: control the "atmosphere" of each shard — mixed-precision settings, temperature scaling, attention head masking — to bias toward specific output distributions.
- **Rejection for crazing.** Potters discard pots with microcracks (crazing) at kiln-door stage. Apply: discard activations with micro-perturbations (detected via Lipschitz-continuity violation from input) before they propagate.

**Why it might work:** Quality-gating discrete processes is a solved engineering discipline.

---

## 8. Stage Director (theater)

**Framing:** LLM inference is a live play. Actors are shards. The set (cache) gets rearranged during scene changes. Stagehands (coordinator) move pieces while audience doesn't see.

**Ideas:**
- **Scene-change masking.** Between scenes, stagehands work furiously behind a curtain. Apply: when the model transitions between tokens, coordinator rebalances load / migrates shards during the natural user-read-pause rather than during active compute. Users see smooth output.
- **Understudy system.** Every principal role has an understudy. For each critical shard, a volunteer is pre-warmed as understudy; takes over instantly if principal fails. Near-zero-latency failover.
- **Rehearsal protocol.** Plays rehearse before opening. Synapse: every new volunteer runs a "rehearsal" — executes known test queries from a gold set; earns reputation over time. Rehearsal signal replaces trust-by-identity.

**Why it might work:** Theater production has solved real-time coordinated multi-person performance with graceful failure for ~2500 years.

---

## TOP 5 CLAUDE PICKS (Round 2)

1. **VLBI-style activation correlation** — novel Byzantine defense via phase coherence across redundant computation
2. **Matched filter Byzantine detection** — cheap, deployable, extends Benford's-law work
3. **Benchmark-query network calibration** (surveyor) — systematic volunteer calibration
4. **Phase-gated verification at finer granularity** (potter) — intra-shard Byzantine checks
5. **Understudy pre-warming for failover** (stage director) — operational resilience

---

*Round 2 of weird faculty ideation. Gemini running same exercise independently. Cross-verify next.*
