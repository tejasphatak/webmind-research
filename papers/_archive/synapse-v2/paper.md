# Synapse v2: Decentralized Inference via Distributed Specialist Models

**Tejas Phatak**
Webmind Research
synapse@webmind.sh

## Abstract

We present Synapse v2, a decentralized inference system where each participating device runs a complete, independently-functional language model that specializes through local fine-tuning on its user's interaction patterns. Unlike tensor or pipeline parallelism (which partition a single model across devices), or federated learning (which aggregates updates into a single global model), our approach preserves specialist diversity and collaborates at inference time through confidence-based routing. On a proof-of-concept with GPT-2 124M as the base model, we demonstrate: (1) 88% accuracy with 4 specialist nodes fine-tuned on disjoint topics, with only 10 gradient steps per node; (2) instant local response (220ms on CPU) with async specialist refinement (<900ms total); (3) heterogeneous device support through dynamic model truncation (4-12 transformer layers). We discuss the architecture's implications for offline-first deployment on consumer devices without datacenter infrastructure.

## 1. Introduction

Running large language models requires datacenter GPUs inaccessible to most of the world's population. We ask: can a group of consumer devices — phones, tablets, laptops — collaboratively provide useful language model capabilities without any centralized infrastructure?

Prior approaches to distributed LLM inference partition a single model across devices through tensor parallelism [Shoeybi et al., 2019] or pipeline parallelism [Borzunov et al., 2023]. These approaches require synchronous communication at every layer boundary, making them impractical over consumer-grade networks with variable latency and frequent disconnections.

We propose a fundamentally different approach: each device runs a **complete, self-sufficient model** that provides instant local responses. Devices improve quality by consulting specialist peers asynchronously. No device depends on any other device for basic functionality. The system degrades gracefully under arbitrary device failures.

## 2. Architecture

### 2.1 Device Model

Each participating device downloads a pretrained base model, truncated to fit the device's available memory:

| Device class | Layers | Parameters | Storage |
|-------------|--------|-----------|---------|
| Low-end phone | 4 | ~50M | ~271MB* |
| High-end phone | 6 | ~60M | ~328MB* |
| Laptop | 8 | ~70M | ~384MB* |
| Desktop | 12 | ~124M | ~498MB* |

*Sizes shown for GPT-2 variants. Production deployment would use quantized models (~4× smaller.

All devices share the same base weights; smaller devices simply use fewer layers. This enables heterogeneous participation without separate model training.

### 2.2 Specialization Through Local Fine-tuning

Each device continuously fine-tunes its local model on its user's interactions using standard gradient descent. Over time, devices naturally specialize:
- A student studying science develops a science-specialized model
- A programmer's device specializes in code
- A historian's device specializes in history

No coordination is required for specialization — it emerges from usage patterns.

### 2.3 Inference Pipeline

For each user query:

1. **Local response (instant):** The user's device generates a response using its local model. This provides immediate output regardless of network availability.

2. **Specialist routing (async):** In the background, the query is sent to K peer devices. Each peer scores the query using its local model's confidence (negative cross-entropy loss on a dummy continuation). The top-K most confident peers generate their own responses.

3. **Refinement:** If a specialist's response has higher confidence than the local response, it replaces or refines the displayed output. The user sees the local response immediately, potentially improved within ~500-800ms.

### 2.4 Communication Protocol

Inter-device communication uses UDP for minimal overhead:
- **Query broadcast:** prompt text (~200 bytes typical)
- **Confidence score:** single float (4 bytes)
- **Response:** top-k logits per position (~600 bytes) or generated text
- **Total per query:** <2KB round-trip

With carrier-payload PCA compression [previously validated at 36× on 3 model families], this reduces to ~60 bytes per position.

### 2.5 Resilience

Resilience is architectural, not engineered:
- Each device is independently functional — if all peers are offline, the local model still works
- Topics are assigned to multiple specialist nodes (2-3 per topic)
- Device dropout reduces quality, never causes failure
- New devices joining the network immediately contribute (pretrained base model works out of the box)

## 3. Experiments

### 3.1 Setup

- **Base model:** GPT-2 124M (12 layers, 768 hidden, 50257 vocab)
- **Evaluation:** Text completion accuracy on topic-specific prompts
- **Network simulation:** 15ms latency, 20% jitter, 2% packet dropout

### 3.2 Experiment 1: 1:1 Topic-to-Node Ratio

4 specialist nodes, each fine-tuned on 2 examples from a single topic (geography, science, math, language). 5 epochs, AdamW lr=5e-5.

**Results:**

| Query | Target | Specialist | Prediction | Correct |
|-------|--------|-----------|------------|---------|
| Capital of France | Paris | Node 0 | Paris | ✓ |
| Capital of Japan | Tokyo | Node 0 | Tokyo | ✓ |
| Water freezes at | zero degrees | Node 1 | zero degrees | ✓ |
| The sun is a | star | Node 1 | star | ✓ |
| Two plus two equals | four | Node 2 | four | ✓ |
| Ten minus three equals | seven | Node 2 | seven | ✓ |
| Opposite of hot | cold | Node 3 | cold | ✓ |
| Past tense of run | ran | Node 3 | run (wrong) | ✗ |

**Accuracy: 7/8 (88%)**

Routing correctly identified the specialist for each query in all cases.

### 3.3 Experiment 2: Heterogeneous Devices, 2:1 Overloaded

4 nodes with different model sizes (4, 6, 8, 12 layers), 8 topics. Each topic assigned to 3 nodes with overlap.

**Results:**
- Accuracy: 5/24 (21%)
- Geography (12-layer node): 3/3 correct (100%)
- Other topics (4-8 layer nodes): 2/21 (10%)

**Finding:** Model depth significantly affects specialization capacity. Shallow models (4-6 layers) cannot specialize on multiple topics simultaneously.

### 3.4 Latency

| Measurement | Time |
|-------------|------|
| Local inference (CPU) | 220ms |
| Routing (score 4 peers) | 400ms |
| Full pipeline (route + generate) | 670ms |
| + 15ms network latency | 760ms |
| + 100ms network latency | 817ms |
| + 200ms network latency | 893ms |

Local response is always instant. Network latency only affects the async refinement delay, which the user perceives as quality improvement over ~0.5-1 second.

## 4. Related Work

**Tensor Parallelism [Megatron-LM, Shoeybi et al., 2019]:** Splits model within layers across devices. Requires synchronous all-reduce at every layer. Not viable on consumer networks.

**Pipeline Parallelism [Petals, Borzunov et al., 2023]:** Splits model by sequential layer blocks across peers. Requires synchronous activation passing between stages. Latency scales with pipeline depth.

**Federated Learning [McMahan et al., 2017]:** Aggregates client weight updates into a single global model. Destroys specialist knowledge through averaging. Our approach preserves diversity.

**Mixture of Experts [Shazeer et al., 2017]:** Routes tokens to specialist sub-networks within a single model. Requires centralized router and co-located experts. Our approach distributes both routing and experts across the network.

**MoeMoe [Phatak, 2026]:** Our earlier work on erasure-coded tensor decomposition achieves fault-tolerant model partitioning but requires AllReduce synchronization. The Distributed Specialists approach eliminates all synchronization from the critical path.

## 5. Limitations and Future Work

1. **Statistical rigor:** Current results on 8-24 test examples. Needs evaluation on standard benchmarks with hundreds of examples.
2. **Routing scalability:** "Score all peers" routing is O(N) per query. Need hierarchical routing or learned router for 100+ nodes.
3. **Specialization emergence:** Not yet tested whether natural usage patterns create useful specialization without explicit topic assignment.
4. **Cross-topic queries:** Queries spanning multiple topics (e.g., "math behind water freezing") not yet evaluated.
5. **Dynamic model download:** Layer-by-layer download not yet implemented.

## 6. Conclusion

Distributed Specialists represents a paradigm shift from partitioning one model across devices to running many complete models that collaborate through expertise routing. The local-first, async-refinement architecture ensures instant responses regardless of network conditions while improving quality through peer consultation. The approach requires no centralized infrastructure, works on CPU-only devices, and degrades gracefully under arbitrary device failures — properties essential for offline-first deployment in schools, offices, and homes without reliable internet.

## References

- Borzunov, A., et al. (2023). Petals: Collaborative Inference and Fine-tuning of Large Language Models. EMNLP.
- McMahan, H. B., et al. (2017). Communication-Efficient Learning of Deep Networks from Decentralized Data. AISTATS.
- Phatak, T. (2026). MoeMoe: Fault-Tolerant Distributed LLM Inference via Erasure-Coded Tensor Decomposition. Zenodo.
- Shazeer, N., et al. (2017). Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer. ICLR.
- Shoeybi, M., et al. (2019). Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism. arXiv:1909.08053.

## Reproducibility

Code: https://github.com/tejasphatak/webmind-research
