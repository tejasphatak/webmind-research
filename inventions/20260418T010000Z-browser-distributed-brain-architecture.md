# Browser Distributed Brain — Convergence Architecture

**Inventor:** Tejas Phatak
**Date:** 2026-04-18
**Status:** Architecture convergence — combining all session learnings

## Invariants (non-negotiable)

1. Runs on the INTERNET — devices join/leave freely
2. Runs in BROWSERS (WebGPU/WASM) — no app install
3. Each device: ~500MB storage, browser tab
4. Must survive devices dropping mid-inference
5. Users provide prompts → system learns from usage
6. Must produce USEFUL text generation

## What we proved today

| Finding | Implication |
|---------|------------|
| MoeMoe tensor parallelism: exact decomposition | Splitting models works mathematically |
| Pipeline parallelism: cross-device info redundant at 1.5B | Small models don't need cross-device communication |
| Brain v2: 100% accuracy, LAN dropout = better resilience | Biological learning works, adversity helps |
| Brain 200-neuron: 15% accuracy with Hebbian | Scale helps, but pure bio learning is slow |
| Gemini Pro: flat ReLU neurons ≠ transformer | Need transformer-like structure, not just neurons |
| Gemini Pro: 0.5 tok/s max on internet | Latency is the real bottleneck |

## Key decisions

### Decision 1: Each "neuron" is a SMALL TRANSFORMER, not a matmul
- Each browser runs a small complete model (e.g., 500M params)
- This IS the proven approach — collaborative distillation
- Each device is independently useful
- Collaboration improves quality

### Decision 2: Inference is LOCAL-FIRST, network-second
- User types prompt → their browser generates a response using LOCAL model
- LOCAL model is small but fast (50+ tok/s on WebGPU)
- In the background: query is sent to N peer browsers
- Peers run their models on the same prompt
- Results come back asynchronously → refine/improve the response
- If no peers respond: user still gets an answer (degraded but functional)

### Decision 3: Training happens through FEDERATED GOSSIP
- Each browser trains its local model on the prompts it sees
- Periodically: share weight UPDATES (not full weights) with peers
- Weight averaging via gossip (no centralized server)
- More active users = more training = better model
- New device joining: downloads a snapshot from a peer (like BitTorrent seeding)

### Decision 4: Latency solved by SPECULATION
- User's browser generates response immediately (local model)
- Meanwhile, peers generate too
- If peer response is better (lower perplexity) → swap it in
- User sees instant response that gets refined over 1-2 seconds
- Like: draft appears fast, then "sharpens" as peer confirmations arrive

### Decision 5: Communication is JUST LOGITS
- Between devices: send top-k token probabilities (not activations)
- Top-100 tokens × (token_id + probability) = ~600 bytes per position
- At 50 tok/s = 30KB/s — trivial bandwidth
- This is our proven collaborative distillation approach

## The architecture

```
User types prompt in browser
    ↓
LOCAL MODEL (500M, WebGPU) generates response immediately
    ↓                    ↓ (async, background)
User sees response      Query sent to N peers via WebRTC
    ↓                    ↓
    ↓              Peers run their local models
    ↓                    ↓
    ↓              Peer logits arrive (top-100, ~600 bytes each)
    ↓                    ↓
    ↓              Average with local logits → better distribution
    ↓                    ↓
Response REFINES in real-time as peers respond
    ↓
User rates response (implicit: continued conversation = good)
    ↓
Local model trains on this interaction (online learning)
    ↓
Weight updates gossiped to peers over time
```

## Why this works

1. **Local-first**: instant response, no latency dependency
2. **Peer enhancement**: quality improves with more peers (like brain v2)
3. **Resilient**: peers dropping = slightly worse quality, never failure
4. **Browser-native**: WebGPU for inference, WebRTC for P2P, no install
5. **Self-improving**: every query trains every participating model
6. **Scales naturally**: more users = more models = more training data = better quality

## What's needed

1. A small base model (~500M) that runs in WebGPU (Synapse already does this!)
2. Online learning in the browser (gradient updates on WebGPU)
3. WebRTC P2P for logit exchange (Synapse already has this!)
4. Gossip protocol for weight averaging
5. A way to bootstrap (first user gets a pre-trained seed model)

## This IS Synapse v2

Synapse already runs sharded inference in browsers. This architecture replaces sharding with:
- Complete models per device (proven better than shards)
- Collaborative inference via logit averaging
- Online learning from user traffic
- Gossip-based weight updates

The infrastructure exists. The research is proven. The architecture converges.
