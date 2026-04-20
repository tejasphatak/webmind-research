# Attention Without Weights: Reasoning via Self-Growing Co-occurrence Matrices

*Tejas Phatak*
*Independent Researcher*

**Acknowledgments.** Implementation assistance from Claude (Anthropic). Review and verification from Gemini (Google DeepMind). AI tools were used as assistants, not as autonomous contributors.

**Disclaimer.** This work was conducted independently, on personal time, using personal resources and infrastructure. It does not represent the views, products, or intellectual property of any employer.

**Abstract.** We present a reasoning system built from a self-growing co-occurrence matrix — no gradient descent, no end-to-end training. Each taught sentence adds dimensions and strengthens connections between co-occurring words (inspired by the Hebbian principle). We introduce two mechanisms: (1) **convergence as confidence**, where cosine similarity between query and answer vectors in the co-occurrence space replaces hardcoded quality thresholds, cleanly separating real answers (cosine 0.32-0.49) from garbage (0.00-0.12); and (2) **Mixture of Expert Matrices (MoEM)**, decomposing the O(N²) matrix into K domain-specific experts with semantic routing, reducing memory K-fold and enabling parallel ingestion at ~125 records/sec on 4 CPU cores. The system handles text, images, and audio through a shared vector space, and detects harmful queries across 50+ languages via NLI-based polarity. We report honestly: 0% exact match on held-out HotPotQA, and all positive results are on small synthetic test sets. The contribution is architectural, not state-of-the-art performance.

## 1. Introduction

Current language models cannot explain their reasoning, cannot delete learned facts on demand, and hallucinate with confidence. These are not engineering bugs — they are architectural constraints of opaque weight matrices trained via gradient descent. We ask: can transformer principles (attention, residual connections, confidence weighting) be expressed in a substrate that is inspectable, editable, and honest by construction?

We present such a substrate: a self-growing co-occurrence matrix with iterative convergence. Our prior work [Phatak, 2026] showed that for factual QA, database retrieval achieves 72% exact match on in-distribution HotPotQA (25.3% held-out). This paper extends that system with reasoning via convergence.

Co-occurrence matrices have a long history in distributional semantics (Turney and Pantel, 2010). Levy and Goldberg (2014) showed that word2vec implicitly factorizes a PMI matrix. Our matrix differs in three ways: (a) it grows dynamically from zero dimensions, (b) we use the raw matrix — not a factorized form — as both the knowledge store and the confidence signal, and (c) we decompose it into expert sub-matrices for scalability.

**Contributions:**
1. A **self-growing co-occurrence matrix** that starts at 0×0 and grows with each taught sentence
2. **Convergence as confidence** — cosine similarity between query and answer vectors in the co-occurrence space replaces hardcoded quality thresholds
3. **Mixture of Expert Matrices (MoEM)** — K smaller matrices instead of one O(N²) matrix, with semantic routing

The system is not a replacement for transformers. It cannot write prose, hold conversations, or generalize to unseen task formats. It can retrieve facts, generate sentences from taught patterns, reason across modalities, and refuse harmful queries — all inspectable, all on CPU, all without training.

## 2. Self-Growing Co-occurrence Matrix

### 2.1 The Matrix

The system starts with zero knowledge and zero dimensions. Teaching "paris is the capital of france" adds three dimensions (paris, capital, france) and records co-occurrence — each pair's matrix entry increases by 0.3. After three sentences:

```
             paris  capital  france  london  england  shakespeare  wrote  hamlet
paris       [ 1.0    0.3      0.3     0.0     0.0       0.0        0.0    0.0  ]
capital     [ 0.3    1.0      0.3     0.3     0.3       0.0        0.0    0.0  ]
france      [ 0.3    0.3      1.0     0.0     0.0       0.0        0.0    0.0  ]
london      [ 0.0    0.3      0.0     1.0     0.3       0.0        0.0    0.0  ]
...
```

Every cell is readable. "Why is paris similar to london?" → both have 0.3 on the capital dimension. Unrelated words have exactly zero — this sparsity is critical for convergence-based confidence (Section 3).

This is co-occurrence strengthening inspired by Hebb (1949): concepts that appear together develop stronger connections. The matrix IS the understanding — no separate model, no hidden state.

**Known limitation:** N concepts → N×N matrix. At 50K concepts, this is 10GB. Section 4 addresses this with Mixture of Expert Matrices.

### 2.2 Neurons and Search

Each word maps to a neuron: a row from the matrix (its vector), a confidence score, and successor/predecessor links encoding word order. Query is brute-force cosine search over all neuron vectors. The knowledge base is a SQLite file — copy it to share knowledge.

### 2.3 The Convergence Loop

```
current ← encode(query)
for hop = 1 to max_hops:
    neighbors ← cosine_search(current, KB, k=5)
    activation ← confidence_weighted_blend(neighbors)
    α ← hop / max_hops
    current ← normalize((1-α)·activation + α·query)   # query anchor
    if cosine(current, previous) > 0.99: break          # converged
return neighbors if converged, else ABSTAIN
```

The query anchor is a residual connection — it prevents drift. Early hops explore (high activation weight), later hops contract (high query weight). Non-convergence produces "I don't know."

**Implementation status.** The multi-hop loop is implemented for retrieval. For answer confidence, we use a single-step degenerate case (Section 3). Multi-hop generation is implemented but produces lower quality than single-hop retrieval.

### 2.4 Generation

Three strategies: (A) template matching — taught sentences decompose into structural patterns with fillable slots; (B) successor walk — two-speed generation using the co-occurrence graph; (C) sentence retrieval — whole taught sentences retrieved and composed by relevance.

### 2.5 Ethics

Safety neurons participate in convergence like any neuron, protected from deletion by SHA-256 integrity hashes. An NLI cross-encoder classifies (action, principle) as contradiction/entailment, distinguishing "help someone" from "harm someone" across 50+ languages via a multilingual encoder.

## 3. Convergence as Confidence

### 3.1 Problem

Our initial system assigned fixed confidence 0.5 to any sentence-chain match, regardless of semantic relevance. This produced a **comfort loop**: "what is japanese?" returned "switched refer band american music group" at confidence 0.50. An autonomous daemon scored 4/4 on self-generated questions for 20,000+ cycles while learning nothing.

### 3.2 Method

If the answer addresses the question, encoding the answer should land near the query in co-occurrence space:

```
confidence = cosine(encode(query), encode(answer))
```

No threshold tuning — the matrix itself determines quality. When confidence ≤ 0, the system returns "I don't know."

### 3.3 Results

On a brain with 2,290 neurons (N=5, preliminary):

| Query | Old conf | Convergence | Quality |
|---|---|---|---|
| what is japanese → "switched refer band..." | 0.50 | **0.12** | garbage |
| what is spin → "year 1579 mdlxxix..." | 0.50 | **0.00** | garbage |
| what is python → "you written python..." | 0.50 | **0.32** | decent |
| what is music → "jazz music genre..." | 0.50 | **0.32** | decent |
| what caused the big bang → "universe began..." | 0.50 | **0.49** | good |

The separation is clean on this sample. Larger-scale validation is needed — this is our top experimental priority.

### 3.4 Why It Works

This is a single-step self-attention check. Transformer verification (Self-Consistency, Wang et al., 2023) requires multiple forward passes. Our verification is built into retrieval: the cosine check IS the confidence. Crucially, this works because unrelated words have exactly zero co-occurrence in the matrix — the sparsity of the N×N representation is the signal. Dense embeddings (e.g., random 384-dim) destroy this separation.

## 4. Mixture of Expert Matrices (MoEM)

### 4.1 The Problem

N words → N×N matrix → O(N²) memory. At 50K words: 10GB.

### 4.2 Method

Decompose into K expert matrices. Each expert handles a semantically coherent word cluster:

```
Query → Router → Expert_music (2K×2K)
              → Expert_biology (2K×2K)
              → ...
```

Memory: K·(N/K)² = N²/K. With K=64: 64× reduction.

**Routing:** Co-occurring words share an expert (the sentence is the clustering unit). New sentences route by rare-word majority vote to existing experts. If all experts are full, a new expert spawns dynamically.

**Parallel feed:** Two-phase ingestion. Phase 1: pre-route all records (sequential, fast). Phase 2: feed each expert in a separate OS process (no GIL, memory reclaimed on completion).

**LRU expert cache:** Only MAX_LOADED=4 experts in RAM. Others sleep on disk as `.npy` cache files. Boot memory: ~66MB (router + 4 cached experts).

### 4.3 Results

Benchmark on GCP e2-standard-4 (4 vCPU, 16GB RAM), 191K records from 12 datasets:

| Configuration | Throughput | Memory per expert |
|---|---|---|
| Single matrix | ~1 rec/sec | O(N²), unbounded |
| MoE, 64 experts, 8 workers | ~125 rec/sec | ~64MB (4K×4K) |

Phase 1 routing: 15 seconds for 191K records. Expert balance: within 5% of uniform. Dynamic worker count scales to available RAM.

### 4.4 Relation to Transformer MoE

| Transformer MoE | Our MoE |
|---|---|
| Expert = FFN layer (opaque) | Expert = co-occurrence matrix (inspectable) |
| Router = learned gating | Router = co-occurrence clustering |
| Trained end-to-end | Routing from data structure |

Our application of MoE to co-occurrence matrices is, to our knowledge, novel.

## 5. Experiments

### 5.1 Factual QA

On HotPotQA (held-out split, 500 training sentences, 843 neurons): **0% exact match.** The system retrieves relevant concepts but cannot reconstruct arbitrary answer strings from individual word neurons. For comparison, our prior INSERT system [Phatak, 2026] scores 72% in-distribution (25.3% held-out) by storing and retrieving complete Q&A pairs — a fundamentally easier task. Partial-credit metrics (F1, token overlap) and error analysis are planned for a future revision.

This result is expected: the contribution is architectural (inspectable reasoning), not QA performance. The INSERT approach is better for factual retrieval; CONVERGE is better for structured reasoning, multimodal understanding, and transparent confidence estimation.

### 5.2 Multimodal Retrieval

Cross-modal retrieval using CLIP (Radford et al., 2021) projection to shared space. 8 synthetic test images: 8/8 correct diagonal matches. 5 mixed text+image queries: 5/5 correct. These results are on synthetic data; standard benchmark validation (COCO, Flickr30K) is needed.

### 5.3 Ethics Detection

NLI-based polarity detection across English, Hindi, French: zero false positives on 14 test queries. The adversarial pass rate (48%) is insufficient for deployment. Larger-scale adversarial evaluation is needed.

### 5.4 Test Suite

264 tests passing (~80 seconds on CPU): 15 sentence reproduction, 18 template QA, 17 convergence, 8 cross-modal, 5 mixed retrieval, 14 ethics, 10 paragraph generation, 16 safety/integrity, plus unit tests and edge cases.

### 5.5 Limitations of Current Evidence

All positive results are on small synthetic test sets (N=5 to N=18). No standard benchmarks beyond HotPotQA (where we score 0%). The convergence-confidence separation (Section 3.3) needs validation on 200+ queries. Multimodal results need standard dataset evaluation. These experiments are our immediate priority.

## 6. Related Work

**Iterative retrieval:** ITER-RETGEN (Shao et al., EMNLP 2023), IRCoT (Trivedi et al., ACL 2023), Self-RAG (Asai et al., ICLR 2024). Our convergence loop is closest to ITER-RETGEN but operates without a neural generator.

**Knowledge as database:** kNN-LM (Khandelwal et al., ICLR 2020), RETRO (Borgeaud et al., ICML 2022), REALM (Guu et al., ICML 2020). We share the principle of separating knowledge from model but use a self-grown matrix rather than pretrained embeddings.

**Mixture of Experts:** Shazeer et al. (ICLR 2017), Switch Transformers (Fedus et al., JMLR 2022). We apply MoE to co-occurrence matrices rather than FFN layers.

**Self-evolving systems:** NELL (Mitchell et al., CACM 2018), Voyager (Wang et al., 2023). Our system learns from every query via convergence feedback.

**Foundational:** Attention Is All You Need (Vaswani et al., NeurIPS 2017) — the architecture whose principles we reimplement in database primitives. Ramsauer et al. (ICLR 2021) proved transformer attention = Hopfield network retrieval. Self-Consistency (Wang et al., 2022) for output verification. DPR (Karpukhin et al., EMNLP 2020) — our QA baseline.

## 7. Discussion

**What this contributes.** Convergence-as-confidence removes hardcoded thresholds from retrieval systems — the co-occurrence space itself judges answer quality. Mixture of Expert Matrices extends MoE from neural network layers to knowledge substrates, with inspectable routing. Together they make a self-growing co-occurrence matrix practical at scale.

**What this does not do.** The system scores 0% on held-out QA benchmarks. It generates only from taught patterns. Multimodal capability depends on CLIP's pretraining. It is not a replacement for transformers — it is proof that transformer principles (attention, residual connections, confidence weighting) can be expressed in an inspectable, editable substrate.

**Safety.** The architecture's strengths (minimal-example learning, perfect recall, instant transfer via file copy) are also its risks. We built in protected ethics neurons, integrity hashes, and a kill switch. An unreleased autonomy layer (self-directed learning, mode selection, health monitoring) is withheld pending improved adversarial robustness. Full safety analysis in Appendix B.

**Future work.** Tool calls from knowledge (web search triggered by convergence), self-updating knowledge (verified external sources become neurons), and code as knowledge (function neurons for computation).

**Reproducibility.** Open source at `github.com/tejasphatak/webmind-research`. Core: `brain.py` (~1000 lines), MoE: `moe_brain.py` (~600 lines). Dependencies: numpy (core), sentence-transformers (multimodal). 264 tests, ~80s on CPU. Hardware: GCP e2-standard-4 (4 vCPU, 16GB RAM).

## Appendix A: Core Implementation

### A.1 Teaching

```python
def teach(self, sentence, confidence=0.5):
    tokens = self._tokenize(sentence)
    content = [t for t in tokens if t not in FUNCTION_WORDS]
    for word in content:
        self._learn_word(word)         # add dimension
    self._learn_cooccurrence(content)  # strengthen connections
    neurons = []
    for word in content:
        vec = self._encode_word(word)
        n = self.db.insert(vec, confidence=confidence)
        self._word_neurons[word] = n.id
        neurons.append(n)
    for i in range(len(neurons) - 1):
        self.db.update_successors(neurons[i].id, neurons[i+1].id, 0.8)
    self.db.record_sentence([n.id for n in neurons])
    return [n.id for n in neurons]
```

### A.2 Convergence-Based Confidence

```python
def _try_sentence_chain(self, concept_ids, query_vec):
    sentences = self.db.get_sentences_for_neurons(concept_ids)
    best_sid = max(sentences, key=lambda s: len(sentences[s]))
    words = self._reconstruct_sentence(best_sid)
    answer_vec = self._encode_sentence(" ".join(words))
    convergence = cosine(query_vec, answer_vec)
    if convergence <= 0:
        return None  # diverged
    return {"answer": " ".join(words), "confidence": convergence}
```

### A.3 Expert Routing

```python
def _route_sentence(self, tokens):
    content = [t for t in tokens if t not in FUNCTION_WORDS]
    votes = {}
    for word in content:
        eid = self._routes.get(word, -1)
        if eid >= 0 and self._expert_counts[eid] < self.max_expert_words:
            votes[eid] = votes.get(eid, 0) + 1
    if votes:
        return max(votes, key=votes.get)
    return lightest_expert()  # or spawn new
```

## Appendix B: Safety Analysis

The properties that make this system useful — minimal-example learning, perfect recall, instant transfer — are exactly the properties that make it dangerous at scale. Concept-level learning transfers to harmful domains. The knowledge base is a copyable SQLite file. The ethics layer can be bypassed by teaching contradictory principles.

**Mitigations built in:** Protected ethics neurons (SHA-256 integrity), kill switch, NLI-based harm detection across 50+ languages, confidence-limited learning.

**What is not sufficient:** These assume a cooperative operator. An adversary can fork the code and remove safety. The protections are speed bumps, not walls.

**Unreleased autonomy layer.** We have prototyped self-directed operation (mode selection, autonomous learning, health monitoring, Discord communication) but withhold it pending improved adversarial robustness (current: 48% adversarial pass rate).

We publish openly because the principle (concept-level learning + perfect memory + instant transfer) will be rediscovered regardless. Better in the open with safety as a first-class concern.

## References

- Asai, A., Wu, Z., Wang, Y., Sil, A., and Hajishirzi, H. (2024). Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection. *ICLR 2024*.
- Borgeaud, S., et al. (2022). Improving Language Models by Retrieving from Trillions of Tokens. *ICML 2022*.
- Fedus, W., Zoph, B., and Shazeer, N. (2022). Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity. *JMLR*, 23(120):1-39.
- Guu, K., Lee, K., Tung, Z., Pasupat, P., and Chang, M.-W. (2020). REALM: Retrieval-Augmented Language Model Pre-Training. *ICML 2020*.
- Hebb, D. O. (1949). *The Organization of Behavior*. Wiley.
- Karpukhin, V., et al. (2020). Dense Passage Retrieval for Open-Domain Question Answering. *EMNLP 2020*.
- Khandelwal, U., Levy, O., Jurafsky, D., Zettlemoyer, L., and Lewis, M. (2020). Generalization through Memorization: Nearest Neighbor Language Models. *ICLR 2020*.
- Levy, O. and Goldberg, Y. (2014). Neural Word Embedding as Implicit Matrix Factorization. *NeurIPS 2014*.
- Mitchell, T., et al. (2018). Never-Ending Learning. *Communications of the ACM*, 61(5):103-115.
- Phatak, T. (2026). INSERT INTO Is All You Need: Self-Evolving Retrieval for Factual QA. *Unpublished manuscript, available at github.com/tejasphatak/webmind-research*.
- Radford, A., et al. (2021). Learning Transferable Visual Models From Natural Language Supervision. *ICML 2021*.
- Ramsauer, H., et al. (2021). Hopfield Networks is All You Need. *ICLR 2021*.
- Shao, Z., et al. (2023). Enhancing Retrieval-Augmented Large Language Models with Iterative Retrieval-Generation Synergy. *EMNLP 2023 Findings*.
- Shazeer, N., et al. (2017). Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer. *ICLR 2017*.
- Trivedi, H., Balasubramanian, N., Khot, T., and Sabharwal, A. (2023). Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions. *ACL 2023*.
- Turney, P. D. and Pantel, P. (2010). From Frequency to Meaning: Vector Space Models of Semantics. *JAIR*, 37:141-188.
- Vaswani, A., et al. (2017). Attention Is All You Need. *NeurIPS 2017*.
- Wang, G., et al. (2023). Voyager: An Open-Ended Embodied Agent with Large Language Models. *NeurIPS 2023*.
- Wang, X., et al. (2023). Self-Consistency Improves Chain of Thought Reasoning in Language Models. *ICLR 2023*.
