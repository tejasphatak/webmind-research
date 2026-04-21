# Webmind Research — Papers

Research papers exploring self-evolving AI architectures.

**Live demo:** [guru.webmind.sh](https://guru.webmind.sh) | **Model:** [HuggingFace](https://huggingface.co/tejadabheja/guru) | **Code:** [GitHub](https://github.com/tejasphatak/webmind-research)

## Papers

| # | Paper | Status | Key Result |
|---|-------|--------|------------|
| 1 | **[Guru: Self-Evolving Graph Reasoning Engine](guru-self-evolving-graph-reasoning-2026-04-21.md)** | Published | 87% EM after RLHF, 35.8% blended, 254ms, CPU-only |
| | [Engineering Guide](guru-engineering-guide.md) — mermaid diagrams, data flow, API reference | | |
| 2 | [Attention Without Weights](new-gen-ai/paper.md) | Published | Convergence loop = attention on transparent substrate |
| 3 | [Self-Evolving Retrieval](self-evolving-retrieval-2026-04-18.md) | Benchmarked | 0.7% → 25.3% EM on held-out, browser-native 214MB |
| 4 | [SAQT Distributed Cognition](saqt-distributed-cognition-2026-04-18.md) | Draft | P2P multi-hop traversal, 84% keyword coverage |
| 5 | [Ethics Invariants](saqt-ethics-invariants-2026-04-18.md) | Design doc | 7-layer safety architecture |
| 6 | [Synapse v2: Distributed Specialists](synapse-v2-distributed-specialists-2026-04-18.md) | Draft | Decentralized inference, 88% on 8-example test |
| 7 | [MoeMoe: Fault-Tolerant Inference](moemoe-preliminary-2026-04-17.md) | Preliminary | 21-143x resilience, tensor-parallel experts |
| 8 | [SFCA: Shapley Credit Assignment](sfca-preregistration-v1.md) | Pre-registered | 30-day A/B experiment |

## Directory Structure

```
papers/
├── README.md                                          ← you are here
├── guru-self-evolving-graph-reasoning-2026-04-21.md   ← main paper
├── guru-engineering-guide.md                          ← engineering diagrams + API
├── new-gen-ai/                                        ← Guru codebase + prototype paper
│   ├── paper.md                                       ← "Attention Without Weights"
│   ├── server.py                                      ← FastAPI server
│   ├── src/                                           ← Engine code
│   ├── static/                                        ← Chat UI
│   └── teach_conversations.py                         ← Reproducible teaching script
├── self-evolving-retrieval-2026-04-18.md              ← earlier retrieval paper
├── benchmarks-self-evolving/                          ← benchmark data + logs
├── saqt-distributed-cognition-2026-04-18.md
├── saqt-ethics-invariants-2026-04-18.md
├── synapse-v2-distributed-specialists-2026-04-18.md
├── moemoe-preliminary-2026-04-17.md
├── sfca-preregistration-v1.md
└── _archive/                                          ← old drafts, tex files, scripts
```

## Current Architecture

```
304K neurons | 7M edges | 39K Q→A pairs | 54MB CSR + 1.8GB LMDB
CPU only | No GPU | No gradient descent | Learns from every conversation
```

## Discipline

- Every claim verified by execution
- Null results published
- Honest about limitations: 1.8% cold EM
- All code MIT, papers CC-BY 4.0
