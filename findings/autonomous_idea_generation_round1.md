# Autonomous Idea Generation — Round 1

**Date:** 2026-04-16
**Mode:** Fully autonomous; Tejas instructed "keep asking what you can optimize about Synapse and LLMs, validate, iterate."

Goal: surface fresh optimization ideas beyond the 4-paper roadmap. Each idea immediately gets a novelty check. Only survivors get elaborated.

---

## Idea generator output (brainstorm, pre-validation)

Let me think about Synapse from angles we haven't explored. The current system:
- WebGPU-sharded LLM across volunteer browsers
- Pipeline-parallel, activation-transported
- Coordinator routes queries, samples tokens
- Predictor.js + speculative.js hide network latency
- Carrier-payload compresses shard-boundary activations (proposed P1)
- Target users: schools, clinics, low-resource settings

### Fresh ideas (unfiltered)

**I-1. Activation diffing across the request graph ("what-if" reuse)**
> Most LLM queries share prefixes (system prompts, conversation history). Activations for the shared prefix are computed once but often recomputed across requests. Cache activations by prefix hash, reuse across requests with different suffixes. This is *prefix caching* — heavily studied for KV cache (Paged Attention, vLLM SGLang, RadixAttention).
>
> **Novelty check:** vLLM, SGLang all do this. NOT NOVEL.

**I-2. Cross-request activation interpolation**
> If two users ask similar questions, their activations are correlated. Could a shard compute *one* forward pass and interpolate activations for both users? Akin to batching, but at the activation-similarity level rather than batch-alignment level.
>
> **Novelty check:** Semantic caching (Vector DB-backed) exists at the token level (OpenAI-compatible gateways do it). At the activation level — unclear. Might be a real gap. Flag for follow-up.

**I-3. Volunteer reputation-based precision allocation**
> Each volunteer has a reputation score. High-reputation nodes get full-precision work; low-reputation nodes get only "verifiable" work (small activation tiles that can be independently checked). Combines P4's Byzantine work with Synapse's volunteer model.
>
> **Novelty check:** Adjacent to opML and Petals reputation. Specific "precision-allocation-by-reputation" formulation might be fresh. Flag for follow-up.

**I-4. Coordinator as anchor: probabilistic full re-execution sampling**
> The coordinator holds the full model weights (but not typically does compute). Occasionally (1% of queries), it re-executes a random shard's work and compares. This is analogous to Ethereum fraud proofs but at the inference level. Catches malicious shards at minimal cost.
>
> **Novelty check:** opML does this at the operation level. Does Petals do spot-checks? Probably yes, in an ad-hoc way. Formalized protocol might be novel.

**I-5. WebGPU compute graph pre-compilation + signed shaders**
> WebGPU shader compilation is expensive; compile once on the coordinator, sign, distribute. Volunteers just run pre-compiled shaders. Reduces per-query setup cost AND prevents malicious volunteers from injecting their own shaders.
>
> **Novelty check:** Chrome has WebGPU persistent caching. Does Synapse do it already? Check. Shader signing might be new.

**I-6. Activation swarming (M-of-N redundant execution)**
> Send the same shard work to 3 volunteer nodes in parallel. Take the median of their output activations. Byzantine fault-tolerant at the activation level. Tradeoff: 3x compute cost for trust.
>
> **Novelty check:** Classic BFT. Applied to inference — federated-learning-adjacent. Has this been done for inference? "Committee-based inference" exists (ensemble methods) but not for Byzantine-tolerant single-model sharded execution. Flag.

**I-7. Warm-pool of partial shard precomputations**
> For any LLM with hot system prompts (ChatGPT-style), shards can precompute the first few layers on the system prompt alone, cache, and only do the user-suffix compute live. This is aggressive prefix caching but on the shard level.
>
> **Novelty check:** Adjacent to I-1. KV cache sharing exists. Shard-level prefix precomputation — flag for novelty check.

**I-8. Multi-model mixture-of-shards**
> Instead of one model sharded across volunteers, have MULTIPLE models (different sizes, different fine-tunes), and route user queries to the best-fit model's shard chain. Network-wide model selection.
>
> **Novelty check:** Model routing / RouteLLM exists. Multi-model across volunteer network might be fresh. Flag.

**I-9. Gossip-protocol-based model weight distribution**
> Volunteer joining the network fetches shard weights from N randomly-chosen existing volunteers via gossip. Bandwidth-efficient, robust to any single CDN failure. Similar to IPFS / BitTorrent.
>
> **Novelty check:** BitTorrent + ML has been done (BitsTorrent, BlockTorrent). Applied to Synapse-style active inference networks — might be fresh.

**I-10. Attention-head-parallel sharding**
> Current pipeline parallelism splits by LAYER. Alternative: split by attention head. Each head can run on a different volunteer since heads are independent within a layer. Less network hops for attention computation, but more for residual stream.
>
> **Novelty check:** Tensor parallelism typically does this. But across volunteer WebGPU devices? Unusual — most TP is within a single machine. Check.

**I-11. Probabilistic layer skipping based on input uncertainty**
> Easy queries get fewer layers (early exit); hard queries get full depth. Router per layer decides "continue or exit." Adaptive compute.
>
> **Novelty check:** CALM, SkipNet, early-exit transformers — heavily published. NOT NOVEL.

**I-12. Synapse as training substrate for LoRA adapters**
> Users could contribute compute for fine-tuning LoRAs on their own devices, shared back to the network. Federated LoRA training on consumer devices.
>
> **Novelty check:** Federated LoRA exists (FedLoRA, FedIT). Decentralized volunteer variant — flag.

**I-13. Inference-time model distillation via volunteer observations**
> Volunteers observe activations flowing through their shards. Use these as training data to distill a smaller/faster model over time. The network grows smarter by running.
>
> **Novelty check:** This is a Nexus idea (self-learning from inference). Overlaps with continual learning + knowledge distillation. Might be genuinely novel at the *volunteer-decentralized* scale. FLAG STRONGLY — this connects to Tejas's Nexus trajectory.

**I-14. Zero-knowledge proof of correct inference — but only on TOKENS, not activations**
> ZKML on full activations is too expensive. But proving "given this input and KV state, the output token is correct" might be tractable via circuits over just the final softmax + a hash chain of intermediate states. Much narrower proof, still tamper-evident.
>
> **Novelty check:** EZKL does layer-level ZK. Token-level ZK — check. Might be real gap.

**I-15. Cross-layer activation delta cache**
> Within a single forward pass, activations at layer k and layer k+1 are often similar (residual streams). Could we transmit only `(layer_{k+1} - layer_k)` across shards?
>
> **Novelty check:** Residual stream compression — MOMENT (Rethinking Self-Attention via ...) touches this. Zero-DCE and residual-only frame prediction in video. Applied to LLM pipeline — likely adjacent work. Flag.

---

## Next: literature cross-verification on the flagged ideas

From the 15 above, these are worth literature-checking:
- I-2 (cross-request activation interpolation)
- I-3 (reputation-based precision allocation)
- I-4 (coordinator spot-checking / probabilistic fraud proofs)
- I-6 (M-of-N redundant inference for Byzantine tolerance)
- I-7 (shard-level prefix precomputation warm pool)
- I-8 (multi-model mixture-of-shards on volunteer network)
- I-9 (gossip-protocol weight distribution)
- I-10 (attention-head-parallel across WebGPU volunteers)
- I-12 (federated LoRA on volunteer network)
- I-13 (inference-time distillation for Nexus) — HIGH PRIORITY
- I-14 (token-level zero-knowledge proofs)
- I-15 (cross-layer delta cache)

Killed:
- I-1 (prefix caching — vLLM, SGLang have it)
- I-5 (WebGPU shader caching — Chrome does it)
- I-11 (early exit — widely published)

Next: dispatch literature validator on the surviving 12.
