# P3 Multi-Faculty Analysis — Distributed Speculative Decoding

**Author:** Claude Opus 4.6
**Date:** 2026-04-16
**Target paper:** P3 from ROADMAP — Distributed Speculative Decoding
**Existing infra (Tejas, Synapse repo, Apr 13-14):** `predictor.js` (linear extrapolation + EMA + planned learned MLP), `speculative.js` (controller with batch depth 3, warmup management, hit/miss stats), 0.995 cosine acceptance threshold

**Key question:** Given Tejas already built the system — what's the genuine research contribution for a paper, beyond "we measured a working system"?

---

## 1. CPU Architect / VLSI (branch prediction)

**Speculative decoding in their language:** "This is branch prediction transplanted from CPU pipelines to distributed ML inference. The 'branch' is the network transmission of an activation (50ms latency). A correctly predicted activation hides that latency the way a correctly predicted branch hides memory latency. A misprediction costs a pipeline flush (recompute from real data)."

**Optimizations:**
1. **Tournament / hybrid predictors (KNOWN in CPU, NOVEL APPLICATION).** Modern CPUs use multiple predictors and a meta-predictor picks between them per branch. Apply: have multiple activation predictors (linear extrap, EMA, learned MLP) and a per-request meta-predictor that learns which predictor wins for that request's trajectory.
2. **TAGE-style indexed prediction (NOVEL APPLICATION).** Use last N activations as a hash index into a prediction table, find the best prediction for that specific signature. Directly applies.
3. **Confidence estimation (ALREADY IN TEJAS'S CODE via cosine threshold, but crude).** Replace scalar threshold with a calibrated probabilistic confidence. Reduces false-accept/false-reject rates.
4. **Gshare and local/global history mixing.** Combine per-request state with global network-wide patterns.

**Connection to the paper:** Tournament predictors are a MEASURABLE contribution — "we show that a 3-way tournament of {linear, EMA, learned} predictors beats each individually by X% hit rate at Y compute cost." That's a concrete experimental result.

**Novelty:** The CPU-architect framing isn't new in abstract (people have drawn the analogy), but systematic transplant of specific techniques (TAGE indexing, tournament meta-prediction) to ML inference speculation is genuinely underexplored.

---

## 2. Distributed Systems (OCC, speculative execution in DBs)

**Speculative decoding in their language:** "This is optimistic concurrency control in a pipelined compute graph. The downstream node speculates, the upstream produces the real value, a compare-and-swap-style commit decides. Traditional OCC handles writes to shared state; here it handles activation flow."

**Optimizations:**
1. **Multi-version concurrency (MVCC)-inspired rollback (NOVEL APPLICATION).** If speculation is wrong at shard k, can we REUSE the speculation at shard k+1 with a delta correction, rather than flushing the entire pipeline? Most DB systems do partial rollback; inference engines today rollback fully.
2. **Calvin-style deterministic speculation.** If the predictor is deterministic and all nodes agree, speculation can happen across the whole pipeline simultaneously without coordination.
3. **Speculative SAGA pattern.** If a speculation chain spans 3+ shards and only the last one misses, re-issue just the tail.

**Connection to paper:** The "partial rollback" idea is GENUINELY NOVEL. Inference speculation today is all-or-nothing. Graded rollback (reuse cached speculation with residual correction) would be publishable.

**Novelty:** HIGH. Need to check literature.

---

## 3. Control Theory (Kalman filter, observers, MPC)

**Speculative decoding in their language:** "This is an observer problem. Given noisy observations of past activations, predict the next. Kalman filtering is the optimal linear estimator under Gaussian noise assumptions. MPC (model predictive control) extends this to sequences."

**Optimizations:**
1. **Kalman predictor instead of linear extrapolation (INCREMENTAL, but principled).** Linear extrap is a special case. Kalman gives optimal prediction under a noise model. Learn the noise covariance from observed data.
2. **Particle filter for non-Gaussian / nonlinear dynamics (NOVEL for ML inference).** Activation trajectories may not be Gaussian. Particle filters handle arbitrary distributions. Heavier compute but better predictions.
3. **MPC horizon tuning.** Tejas has batch depth 3. Why 3? Kalman + MPC gives a principled way to pick horizon based on prediction variance.

**Connection to paper:** Kalman-based predictor is a natural replacement for the current linear extrapolator. Measurably better hit rates. Formal framing elevates the paper from "engineering hack" to "principled controller design."

**Novelty:** Kalman for ML activation prediction — probably some prior art. Need to check.

---

## 4. Signal Processing (LPC, speech coding)

**Speculative decoding in their language:** "Tejas's linear extrapolator is a first-order Linear Predictive Coder (LPC). Speech codecs (GSM, CELP, LPC-10) have been using this since the 1970s to predict next speech sample from previous and transmit only the residual."

**Optimizations:**
1. **Higher-order LPC (INCREMENTAL, proven).** Speech uses order 10-12. First-order linear extrap is primitive. Order 8-16 LPC for activations. Direct improvement with known theory.
2. **Adaptive LPC with lattice filter (NOVEL APPLICATION).** Adapts coefficients over time. Better for non-stationary activation trajectories.
3. **Vector LPC / VQ for residual (NOVEL APPLICATION).** Speech CELP uses vector quantization on the residual after prediction. Apply to activation residuals — combines with carrier-payload.

**Connection to paper:** Vector LPC + carrier-payload is a beautiful combination. The LPC predicts, the residual gets carrier-payload compressed. Stack their gains. **This is the key insight for the P3/P4 synthesis.**

**Novelty:** HIGH for the specific combination with carrier-payload. LPC for activations might have prior art — must check.

---

## 5. Data Compression (LZ77, PPM, arithmetic coding)

**Speculative decoding in their language:** "This is Prediction by Partial Matching (PPM) at the activation level. A good predictor = good compressor (source coding theorem). If you can predict the next activation, you can encode it in very few bits."

**Optimizations:**
1. **Context mixing (KNOWN in compression, NOVEL APPLICATION).** PAQ-style: combine multiple predictors weighted by recent accuracy. Same idea as tournament predictor from CPU architecture — convergence across fields.
2. **Arithmetic coding for residuals (NOVEL).** Encode activation residuals using arithmetic coding based on predictor's probability distribution. Information-theoretically optimal.
3. **Semi-adaptive coding.** Learn a per-shard prior from first few queries, use for rest of session.

**Connection to paper:** Arithmetic coding on residuals is the rigorous information-theoretic upper bound for any prediction-based compression scheme. Measuring how close we get to this bound is a publishable theoretical contribution.

---

## 6. Game Theory / Adversarial (Byzantine)

**Speculative decoding in their language:** "In a decentralized network, a malicious predictor node could deliberately produce unpredictable activations to force maximum speculation miss rate, increasing latency for targeted users. Or could lie about predictions to sabotage downstream."

**Defenses:**
1. **Predictor attestation (NOVEL).** The predictor must commit to its prediction (hash) before receiving the real activation. Prevents lying about predictions. Freivalds-style, ties into our Byzantine verification work.
2. **Cost of adversarial unpredictability.** Game-theoretic result: how much can an adversary slow the system by being maximally unpredictable? Upper bound.
3. **Reputation-weighted speculation.** Only accept speculations from high-reputation predictors. Ties into our Synapse threat model.

**Connection to paper:** The Byzantine angle genuinely differentiates this work from CPU-level speculation. CPUs don't have adversaries; decentralized networks do. Novel contribution.

---

## 7. Behavioral Economics / Betting (added by me, not Gemini)

**Speculative decoding in their language:** "Every speculation is a bet. You pay compute cost X to save network latency L with probability p (hit rate). Expected value: p·L - X. Optimal strategy depends on varying cost/latency ratio."

**Optimizations:**
1. **Kelly criterion for speculation depth (NOVEL).** The Kelly criterion tells you optimal bet size given win probability and payoff. Apply: given hit rate p and network savings L, optimal speculation depth k is Kelly-derived.
2. **Economic throughput bound.** If speculation costs more GPU than it saves in latency, don't do it. Formalize as a cost-benefit framework.

**Connection to paper:** Kelly-based adaptive speculation depth could be a minor section.

---

## LITERATURE CROSS-VERIFIER (doing it myself this time)

Before finalizing, I need to check actual prior art on the genuinely novel-sounding ideas:

### Claim: "Distributed Speculative Decoding for LLM Inference"
Must search for prior work.

### Claim: Kalman filter / LPC for activation prediction
Must search.

### Claim: Graded/partial rollback for speculation
Must search.

---

## TOP 5 RANKED OPTIMIZATIONS (novelty × feasibility × paper-boost, pending literature check)

| Rank | Idea | Novelty | Feasibility | Paper-boost |
|---|---|---|---|---|
| 1 | **LPC/Kalman-based predictor + carrier-payload on residuals (synthesis)** | HIGH (specific combo) | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ (unifies P1+P3) |
| 2 | **Tournament / meta-prediction (CPU architecture analog)** | MODERATE | ⭐⭐⭐⭐⭐ (easy to implement) | ⭐⭐⭐⭐ |
| 3 | **Graded/partial rollback (DB OCC analog)** | HIGH (needs lit check) | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| 4 | **Byzantine-resistant speculation (adversarial)** | HIGH | ⭐⭐⭐ | ⭐⭐⭐⭐ (differentiates from single-node) |
| 5 | **Arithmetic coding of residuals** | MODERATE | ⭐⭐⭐ | ⭐⭐⭐ (theoretical completeness) |

---

## Final Pitch (pending cross-verification)

**Possible P3 paper title:** *"Predictive Activation Transport for Decentralized LLM Inference: A Systems + Theory Study"*

**Three contributions:**
1. **System:** Tejas's predictor.js + speculative.js with tournament meta-prediction extension. Measured hit rates and latency savings on real Synapse hardware (Pixel + iPhone + Nvidia setup).
2. **Theory:** Information-theoretic bounds on achievable compression given predictor accuracy. LPC/Kalman framing gives optimal bounds.
3. **Security:** Byzantine-resistant speculation via prediction attestation. Connects to P4 flagship.

This isn't "we built a predictor" — it's "we show what the theoretical limit is, and how to approach it, against adversaries, on real hardware."

**Key concern:** we HAVEN'T YET done the literature check. Last round we almost claimed novelty on ideas already published. Need to search:
1. "Distributed speculative decoding" — is this published?
2. Kalman filter / LPC for neural network activation prediction — prior art?
3. Activation prediction in pipeline parallel — prior art?

Until literature-validated, all the above is SUGGESTIVE not DEFINITIVE.
