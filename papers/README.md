# Webmind Research — Papers

Research papers from the Webmind project exploring self-evolving AI architectures.

**Live demo:** [guru.webmind.sh](https://guru.webmind.sh) | **Model:** [HuggingFace](https://huggingface.co/tejadabheja/guru) | **Code:** [GitHub](https://github.com/tejasphatak/webmind-research)

## Table of Contents

| # | Paper | Date | Status | Key Result |
|---|-------|------|--------|------------|
| 1 | **[Guru: Self-Evolving Graph Reasoning Engine](guru-self-evolving-graph-reasoning-2026-04-21.md)** | 2026-04-21 | Published | 87% EM after RLHF, 35.8% blended, 254ms, CPU-only |
| | [Engineering Guide](guru-engineering-guide.md) — mermaid diagrams, data flow, API reference | | | |
| 2 | **[Attention Without Weights](new-gen-ai/paper.md)** | 2026-04-18 | Published | Convergence loop = attention on transparent substrate, 250+ tests |
| 3 | **[Self-Evolving Retrieval](self-evolving-retrieval-2026-04-18.md)** | 2026-04-18 | Benchmarked | 0.7% → 25.3% EM on held-out data, browser-native 214MB |
| 4 | [SAQT Distributed Cognition](saqt-distributed-cognition-2026-04-18.md) | 2026-04-18 | Draft | P2P multi-hop traversal, 84% keyword coverage |
| 5 | [Ethics Invariants](saqt-ethics-invariants-2026-04-18.md) | 2026-04-18 | Design doc | 7-layer safety architecture, ethics as knowledge |
| 6 | [Synapse v2: Distributed Specialists](synapse-v2-distributed-specialists-2026-04-18.md) | 2026-04-18 | Draft | Decentralized inference, 88% on 8-example test |
| 7 | [MoeMoe: Fault-Tolerant Inference](moemoe-preliminary-2026-04-17.md) | 2026-04-17 | Preliminary | 21-143x resilience, tensor-parallel experts |
| 8 | [SFCA: Shapley Credit Assignment](sfca-preregistration-v1.md) | 2026-04-14 | Pre-registered | 30-day A/B experiment (ongoing) |

## Evolution

The research evolved through three stages:

1. **Self-Evolving Retrieval** (Paper 3) — proved that a growing knowledge base with convergence can answer questions without neural network training. 25.3% EM on held-out data.
2. **Attention Without Weights** (Paper 2) — formalized the convergence loop as transformer attention reimplemented on a transparent graph substrate. Proved the math is the same.
3. **Guru** (Paper 1) — the current system. LMDB persistence, session WAL, teach/correct/protect APIs, 39K Q→A pairs, live at guru.webmind.sh.

Papers 4-8 explore related directions: distributed inference, ethics, fault tolerance, and agent credit assignment.

## Current Architecture

```
304K neurons | 7M edges | 39K Q→A pairs | 54MB CSR + 1.8GB LMDB
CPU only | No GPU | No gradient descent | Learns from every conversation
```

## Discipline

- Every claim must be verified by execution
- Null results published — we don't file-drawer
- All code MIT, papers CC-BY 4.0
- Honest about limitations: 1.8% cold EM, convergence returns garbage on untaught topics

## Reproducing

```bash
pip install -r ../requirements.txt
```

> **Note:** Some experiments depend on large model files. If you can't reproduce a result, [open an issue](https://github.com/tejasphatak/webmind-research/issues).
