# Webmind Research

**Open research into transparent, inspectable AI that stores knowledge in databases — not weight matrices.**

Two architectures. One thesis: the most expensive part of AI (training) can be replaced with the cheapest part of computing (database operations). Everything inspectable. Everything editable. Everything on CPU.

**[Live Demo](https://webmind.sh)** | **[Papers](papers/)** | **[Benchmarks](papers/new-gen-ai/benchmarks/)**

---

## The Two Systems

### 1. INSERT — Self-Evolving Retrieval Engine
**[Paper](papers/self-evolving-retrieval-2026-04-18.md)** | **[Code (Synapse)](https://github.com/tejasphatak/Synapse)**

A 22M-parameter encoder + growing database of verified Q&A pairs. Learns from every query. No training.

| Dataset | Before Self-Learning | After |
|---------|---------------------|-------|
| NaturalQuestions | 0.0% | 56.0% |
| TriviaQA | 0.0% | 66.0% |
| HotPotQA | 0.0% | 92.0% |
| **Held-out (generalization)** | **0.7%** | **25.3%** |

Runs in a browser (214MB), works offline, handles 50+ languages.

### 2. CONVERGE — Reasoning Without Training
**[Paper](papers/new-gen-ai/paper.md)** | **[Code](papers/new-gen-ai/src/)**

A self-growing matrix where each concept adds a dimension. Co-occurring concepts strengthen connections (Hebbian). Query is iterative convergence over the matrix — the same math as transformer attention, on an inspectable substrate.

Starts with **zero knowledge and zero dimensions**. No pretrained embeddings. No gradient descent. ~300 lines of Python + numpy.

| Capability | Result | Tests |
|-----------|--------|-------|
| Taught sentence reproduction | 100% | 15 |
| Template-based QA | 100% | 18 |
| Multi-hop reasoning | 100% | 17 |
| Cross-modal retrieval (text+image) | 8/8 | 8 |
| Mixed text+image retrieval | 5/5 | 5 |
| Ethical detection (50+ languages) | 0 false positives | 14 |
| Paragraph generation | 100% | 10 |
| Safety (kill switch + integrity) | 100% | 16 |

**250+ tests passing.** Multimodal (text, image, audio, video via CLIP). Ethics built into the same convergence loop.

---

## The Thesis

A transformer computes: `softmax(Q*K^T/sqrt(d))*V`

| Transformer concept | Our substrate | Why it matters |
|---------------------|---------------|----------------|
| Attention | Cosine search over growing matrix | Inspectable per-hop |
| Weights | Confidence scores per neuron | Editable, traceable |
| Feed-forward | Rule lookup / successor walk | No hidden layers |
| Layers | Convergence hops | Variable depth, stops when stable |
| Training | Co-occurrence + database insert | Instant, incremental, $0 |
| Residual connection | Query anchor | Same function, explicit |

Same math. Different substrate. The substrate gives us what neural nets can't: **inspectability, editability, honesty about failure.**

We are not avoiding transformers. We are reimplementing the principles that make them work — using database primitives instead of matrix multiplies.

## Honest Assessment

| What we do better | What transformers do better |
|-------------------|-----------------------------|
| Every answer traces to specific neurons | Fluent creative prose |
| Delete a neuron = knowledge gone immediately | Novel reasoning over unseen concepts |
| Non-convergence = "I don't know" (no hallucination) | Long-range coherence |
| CPU-native, <600ms/query | Conversation and dialogue |
| Teach one fact, immediately available | Zero-shot generalization |
| Multimodal by construction | |

The CONVERGE engine scores **0% on held-out HotPotQA** — it can't yet reconstruct arbitrary answers from word neurons. The INSERT engine scores 72% on the same test. They're complementary, not replacements.

## Papers

| Paper | Status | Key Result |
|-------|--------|------------|
| [Self-Evolving Retrieval](papers/self-evolving-retrieval-2026-04-18.md) | Benchmarked | 0.7% -> 25.3% EM, self-learning |
| [From INSERT to CONVERGE](papers/new-gen-ai/paper.md) | Published | Multimodal reasoning without training, 250+ tests |
| [Activation Speculation Dead](papers/activation-speculation-dead/) | Published | Negative result -- bootstrap deadlock |
| [SAQT Distributed Cognition](papers/saqt-distributed-cognition-2026-04-18.md) | Draft | Distributed knowledge mesh |
| [SAQT Ethics](papers/saqt-ethics-invariants-2026-04-18.md) | Draft | Safety through data |
| [SFCA Credit Assignment](papers/sfca-preregistration-v1.md) | Pre-registered | Shapley-fair attribution |
| [Synapse v2](papers/synapse-v2-distributed-specialists-2026-04-18.md) | Draft | Distributed specialists |
| [MoeMoe Resilience](papers/moemoe-preliminary-2026-04-17.md) | Preliminary | Node failure recovery |

## Run It

```bash
# Live demo
open https://webmind.sh

# Run the CONVERGE engine locally
cd papers/new-gen-ai/src
pip install numpy
python3 brain.py

# Run the INSERT engine (Synapse)
git clone https://github.com/tejasphatak/Synapse.git
cd Synapse/synapse-src/saqt
pip install sentence-transformers faiss-cpu
python3 serve.py
```

## Safety Warning

The CONVERGE architecture learns from minimal examples, has perfect recall, and transfers instantly (copy a SQLite file). These properties are dangerous at scale. The system includes ethics neurons, integrity hashing, and a kill switch — but these are speed bumps, not walls. See [the paper's safety section](papers/new-gen-ai/paper.md#12-safety-warning) for our full assessment.

## Citation

```bibtex
@misc{phatak2026converge,
  title={From INSERT to CONVERGE: Multimodal Reasoning Without Training},
  author={Phatak, Tejas and Claude (Anthropic)},
  year={2026},
  url={https://github.com/tejasphatak/webmind-research}
}

@misc{phatak2026selfevolving,
  title={Self-Evolving Retrieval: A Third Architecture for AI Beyond Generation and Search},
  author={Phatak, Tejas},
  year={2026},
  url={https://github.com/tejasphatak/webmind-research}
}
```

## License

Code: MIT | Papers: CC-BY 4.0
