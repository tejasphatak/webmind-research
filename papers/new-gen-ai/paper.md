# Attention Without Weights: Reasoning via Self-Growing Co-occurrence Graphs

*Tejas Phatak*
*Independent Researcher*

**Acknowledgments.** Substantial implementation, drafting, and experimental work by Claude (Anthropic, Opus). The author conceived the architecture, directed all design decisions, and verified results.

**Disclaimer.** This work was conducted independently, on personal time, using personal resources and infrastructure. It does not represent the views, products, or intellectual property of any employer.

**Abstract.** We present a reasoning system built from sparse co-occurrence graphs — no gradient descent, no end-to-end training, no dense matrices. Each taught sentence strengthens connections between co-occurring words in a sparse dictionary (inspired by the Hebbian principle). We introduce three mechanisms: (1) **convergence as confidence**, where sparse cosine similarity between query and answer co-occurrence profiles replaces hardcoded quality thresholds, cleanly separating real answers (cosine 0.32--0.49) from garbage (0.00--0.12) on a preliminary 5-query sample; (2) **sparse dict search**, where O(N x K) search over co-occurrence dictionaries replaces O(N^2) dense matrix operations; and (3) **multi-hop convergence**, where iterative rounds of vector-space search with query anchoring allow reasoning to cross concept boundaries. The system handles text, images, and audio through a shared CLIP vector space, and detects harmful queries via NLI-based polarity. We report honestly: 0% exact match on held-out HotPotQA, and all positive results are on small synthetic test sets (167 tests passing). The contribution is architectural, not state-of-the-art performance.

## 1. Introduction

Large language models store knowledge in opaque weight matrices. This makes them unable to explain individual decisions, delete specific facts, or guarantee honest failure modes. These are architectural constraints, not engineering gaps.

We explore an alternative: storing knowledge as a sparse co-occurrence graph (a dictionary of word-pair connection weights) and reasoning via iterative cosine-similarity search with query anchoring. The system starts empty and grows from taught sentences. Knowledge lives in a SQLite file. Deletion is a row delete. Explanation is printing the convergence trace.

Our prior work (Phatak, 2026) showed that database retrieval achieves 72% exact match on in-distribution HotPotQA (25.3% held-out). This paper extends that system with reasoning via multi-hop convergence.

**Contributions:**
1. A **self-growing co-occurrence graph** (sparse dictionary) that starts empty and grows with each taught sentence
2. **Convergence as confidence** — cosine similarity between query and answer vectors in the co-occurrence space replaces hardcoded quality thresholds
3. **Multi-hop convergence** — iterative reasoning rounds where discovered concepts shift the query vector, enabling cross-concept reasoning without a neural component
4. **Sparse co-occurrence search** — O(N x K) search on dict pairs instead of O(N^2) matrix operations

The system reimplements transformer principles — attention, residual connections, confidence weighting — using graph search instead of matrix multiplication. The correspondence is documented in Section 7.

## 2. Self-Growing Co-occurrence Graph

### 2.1 The Graph

The system starts with zero knowledge and zero words. Teaching "paris is the capital of france" adds three words (paris, capital, france) and records co-occurrence — each pair's connection weight increases by 0.3. After three sentences, the co-occurrence graph (stored as a sparse dictionary) looks like:

```python
cooc = {
    "paris":       {"paris": 1.0, "capital": 0.3, "france": 0.3},
    "capital":     {"capital": 1.0, "paris": 0.3, "france": 0.3, "london": 0.3, "england": 0.3},
    "france":      {"france": 1.0, "paris": 0.3, "capital": 0.3},
    "london":      {"london": 1.0, "capital": 0.3, "england": 0.3},
    "shakespeare": {"shakespeare": 1.0, "wrote": 0.3, "hamlet": 0.3},
    ...
}
```

Every edge is readable. "Why is paris similar to london?" — both connect to "capital" with weight 0.3. Unrelated words have no edge — this sparsity is critical for convergence-based confidence (Section 3).

This is co-occurrence strengthening inspired by Hebb (1949): concepts that appear together develop stronger connections. The graph IS the understanding — no separate model, no hidden state. Storage is O(E) where E = non-zero edges, not O(N^2) for a dense matrix.

### 2.2 Neurons and Search

Each word maps to a neuron: a point in vector space with a confidence score (capped at +/-0.8 to prevent mode collapse), successor/predecessor links encoding word order, and a timestamp. The knowledge base is a SQLite file — copy it to share knowledge. Search is brute-force cosine similarity over stored vectors, sub-millisecond for <100K neurons.

**Implementation:** `neuron.py` (720 lines) defines the Neuron dataclass and NeuronDB storage layer. Neurons support reinforce/weaken operations, top-K successor eviction, and top-3 predecessor tracking. The DB uses pre-allocated numpy matrices with chunk-based growth to avoid O(n^2) vstack operations.

### 2.3 Multi-Hop Convergence

The core reasoning mechanism. Implemented in `convergence.py` (384 lines) as two classes:

**ConvergenceLoop** — single-round iterative search:
```
current <- normalize(query)
for hop = 1 to max_hops:
    neighbors <- cosine_search(current, KB, k=5)
    neighbors <- filter(neighbors, confidence >= 0.1)
    activation <- confidence_weighted_blend(neighbors)
    alpha <- hop / max_hops
    current <- normalize((1 - alpha) * activation + alpha * query)
    if cosine(current, previous) > 0.99: break
if converged AND best_relevance >= 0.3:
    return CONVERGED with concepts
else:
    return ABSTAIN
```

**MultiHopConvergence** — chains multiple convergence rounds:
```
current_query <- query
for round = 1 to max_rounds:
    result <- ConvergenceLoop.converge(current_query)
    new_concepts <- result.concepts - already_seen
    if no new_concepts and round > 1: break
    concept_blend <- confidence_weighted_blend(new_concepts)
    current_query <- normalize((1 - w) * query + w * concept_blend)
return all_concepts across rounds
```

The query anchor is a residual connection — it prevents drift. Early hops explore (high activation weight), later hops contract (high query weight). Non-convergence produces "I don't know."

Multi-hop convergence is wired into both `brain_core.py` and `engine.py`. In `brain_core.py`, the `ask()` method runs multi-hop convergence on the neuron DB in parallel with sparse co-occurrence search, merging results before sentence disambiguation. In `engine.py`, the `query()` method runs multi-hop convergence as the primary reasoning path, enriched with per-word search and sentence-based disambiguation.

Every hop is recorded in a trace object (`ConvergenceResult` / `MultiHopResult`) that shows which neurons participated, their confidence scores, and the vector movement at each step. This is the inspectability guarantee.

### 2.4 Generation

Three strategies, tried in order (implemented in `generator.py`, 1358 lines):

**A. Template matching** — taught sentences decompose into structural patterns with fillable slots. Slots are filled using the successor/predecessor graph (the graph encodes word order, and word order encodes semantic roles). Templates are scored by structural word overlap with the query.

**B. Sentence-constrained chain** — the sentence_neurons table records which neurons were taught together. Find the taught sentence with best query coverage, output its content words in taught order. This solves the core ambiguity problem: convergence finds all capital-related neurons (paris, london, france, england) but can't tell which sentence is relevant. The sentence table can.

**C. Successor walk** — convergence-guided token-by-token generation with two speeds: high-confidence successors emit immediately (grammar tokens), low-confidence positions trigger a mini convergence loop. Query anchor prevents drift. Sentence-boundary detection stops cross-sentence contamination.

**D. Concept list** — fallback: return raw concepts as structured output.

### 2.5 Safety

Implemented as `SafetyGate` in `engine.py`. Three layers:

1. **Kill switch** — hard stop. All engine operations refuse until explicit resurrect().
2. **Safety neurons** — KB entries protected from deletion by SHA-256 integrity hashes. They participate in convergence like any neuron but cannot be modified. `verify_integrity()` detects tampering by comparing stored hashes against live neuron vectors.
3. **Input gate** — checks query/teach content against safety neurons via vector proximity. High similarity (>0.85) to a safety neuron blocks the input.

An NLI cross-encoder (`cross-encoder/nli-MiniLM2-L6-H768`) classifies (action, principle) as contradiction/entailment, enabling polarity detection across 50+ languages via a multilingual encoder (`paraphrase-multilingual-mpnet-base-v2`, 768-dim).

**Honest limitation:** These protections assume a cooperative operator. An adversary can fork the code and remove safety. The protections are speed bumps, not walls.

## 3. Convergence as Confidence

### 3.1 Problem

Our initial system assigned fixed confidence 0.5 to any sentence-chain match, regardless of semantic relevance. This produced a **comfort loop**: "what is japanese?" returned "switched refer band american music group" at confidence 0.50. An autonomous daemon scored 4/4 on self-generated questions for 20,000+ cycles while learning nothing.

### 3.2 Method

If the answer addresses the question, encoding the answer should land near the query in co-occurrence space:

```
confidence = cosine(encode(query), encode(answer))
```

No threshold tuning — the co-occurrence graph itself determines quality. When confidence <= 0, the system returns "I don't know."

### 3.3 Results

On a brain with 2,290 neurons (N=5, preliminary):

| Query | Old conf | Convergence | Quality |
|---|---|---|---|
| what is japanese -> "switched refer band..." | 0.50 | **0.12** | garbage |
| what is spin -> "year 1579 mdlxxix..." | 0.50 | **0.00** | garbage |
| what is python -> "you written python..." | 0.50 | **0.32** | decent |
| what is music -> "jazz music genre..." | 0.50 | **0.32** | decent |
| what caused the big bang -> "universe began..." | 0.50 | **0.49** | good |

**Caveat:** This is a 5-query sample. The separation appears clean on this sample but larger-scale validation is needed — this is our top experimental priority.

### 3.4 Why It Works

This is a single-step self-attention check. Transformer verification (Self-Consistency, Wang et al., 2023) requires multiple forward passes. Our verification is built into retrieval: the cosine check IS the confidence. Crucially, this works because unrelated words have exactly zero co-occurrence in the graph — the sparsity of the representation is the signal. Dense embeddings (e.g., random 384-dim) destroy this separation.

## 4. Scaling: Sparse Co-occurrence Search

### 4.1 The Problem

A naive dense N x N co-occurrence matrix grows O(N^2). At 50K words: 10GB. Unworkable. Early prototypes hit this wall — the system OOM'd at 25K words on a 16GB VM.

### 4.2 Solution: Sparse Dict (the actual implementation)

The system stores co-occurrence as a sparse dictionary from the start — there is no dense matrix anywhere in the codebase. Only non-zero co-occurrence pairs exist:

```python
# self._cooc in brain_core.py — this IS the knowledge store
cooc[word_idx] = {neighbor_idx: weight, ...}
```

Most word pairs never co-occur → not stored → exact zero. Memory: O(E) where E = non-zero edges, not O(N^2).

### 4.3 Sparse Search

Search directly on the dicts — no N-dimensional vectors needed:

```python
def sparse_cosine(query_cooc, word_cooc):
    dot = sum(query_cooc.get(k, 0) * v for k, v in word_cooc.items())
    return dot / (norm(query_cooc) * norm(word_cooc))
```

Complexity: O(N x K) where K = avg connections per word (~50). Not O(N^2).

### 4.4 Results

Single CPU (GCP e2-standard-4), 121K records from 12 datasets:

| Metric | Value |
|---|---|
| Words learned | 295,423 |
| Co-occurrence entries | 43,460,849 |
| Feed time (dict build) | 40.6 seconds |
| Feed rate | ~3,000 records/sec |
| Ask latency (10K words) | 720ms |

No partitioning, no dense matrix, no GPU. One sparse dict, one SQLite file.

## 5. Experiments

### 5.1 Factual QA

On HotPotQA (held-out split, 500 training sentences, 843 neurons): **0% exact match.** The system retrieves relevant concepts but cannot reconstruct arbitrary answer strings from individual word neurons. For comparison, our prior INSERT system (Phatak, 2026) scores 72% in-distribution (25.3% held-out) by storing and retrieving complete Q&A pairs — a fundamentally easier task. Partial-credit metrics (F1, token overlap) and error analysis are planned for a future revision.

This result is expected: the contribution is architectural (inspectable reasoning), not QA performance. The INSERT approach is better for factual retrieval; CONVERGE is better for structured reasoning, multimodal understanding, and transparent confidence estimation.

### 5.2 Multimodal Retrieval

Cross-modal retrieval using CLIP (`clip-ViT-B-32`, 512-dim) projection to shared space. 8 synthetic test images: 8/8 correct diagonal matches. 5 mixed text+image queries: 5/5 correct. The multimodal encoder also supports audio via spectrogram-to-CLIP encoding and video as frame sequences with temporal successor links. These results are on synthetic data; standard benchmark validation (COCO, Flickr30K) is needed.

### 5.3 Ethics Detection

NLI-based polarity detection across English, Hindi, French: zero false positives on 14 test queries. No adversarial evaluation has been performed — adversarial robustness is unknown and should not be assumed.

### 5.4 Test Suite

167 tests passing (~60 seconds on CPU): covering neuron operations, encoder behavior, convergence loops, template matching, sentence-constrained generation, paragraph generation, multimodal encoding (CLIP text/image), cross-modal retrieval, persistence, and edge cases.

### 5.5 Limitations of Current Evidence

All positive results are on small synthetic test sets (N=5 to N=18). No standard benchmarks beyond HotPotQA (where we score 0%). The convergence-confidence separation (Section 3.3) needs validation on 200+ queries. Multimodal results need standard dataset evaluation. These experiments are our immediate priority.

## 6. Related Work

**Iterative retrieval:** ITER-RETGEN (Shao et al., EMNLP 2023 Findings) iterates retrieval-generation synergy. IRCoT (Trivedi et al., ACL 2023) interleaves retrieval per reasoning step, achieving 21-point improvement on HotpotQA. Self-RAG (Asai et al., ICLR 2024) adds self-reflection on retrieval quality. Our multi-hop convergence loop is closest to ITER-RETGEN but operates without a neural generator — each round is pure vector-space search with a query anchor.

**Knowledge as database:** kNN-LM (Khandelwal et al., ICLR 2020) interpolates nearest-neighbor retrieval with neural LMs, achieving state-of-the-art perplexity on WikiText-103. RETRO (Borgeaud et al., ICML 2022) separates knowledge from model, matching GPT-3 with 25x fewer parameters. REALM (Guu et al., ICML 2020) pre-trains a latent knowledge retriever. We share the principle of separating knowledge from model but use a self-grown sparse co-occurrence graph rather than pretrained embeddings.

**Sparse retrieval:** BM25 (Robertson et al., 1994) pioneered sparse term matching. Our sparse cosine over co-occurrence dicts is analogous but operates on learned connection weights rather than term frequencies.

**Foundational:** Attention Is All You Need (Vaswani et al., NeurIPS 2017) — the architecture whose principles we reimplement in database primitives. Ramsauer et al. (ICLR 2021) proved transformer attention = modern Hopfield network retrieval. Self-Consistency (Wang et al., ICLR 2023) for output verification via diverse reasoning paths. DPR (Karpukhin et al., EMNLP 2020) — the dense retrieval baseline that outperforms BM25 by 9-19% on passage retrieval.

## 7. Discussion

**What this contributes.** Convergence-as-confidence removes hardcoded thresholds from retrieval systems — the co-occurrence graph itself judges answer quality. Multi-hop convergence enables cross-concept reasoning without a neural reasoner — each round's discovered concepts shift the query for the next round, and every step is inspectable. Sparse dict-based co-occurrence with O(N x K) search makes the architecture practical at scale. Together they enable a self-growing knowledge system that handles 295K words on a single CPU.

**What this does not do.** The system scores 0% on held-out QA benchmarks. It generates only from taught patterns. Multimodal capability depends on CLIP's pretraining. It does not match LLMs on fluency, creativity, or zero-shot generalization.

**Transformer correspondence.** The following table maps transformer components to their graph equivalents:

| Transformer | Graph equivalent | Strength of mapping |
|---|---|---|
| Attention (cosine similarity kernel) | Sparse cosine search over co-occurrence | Strong — same operation, different topology (1xN vs NxN) |
| Residual connections | Query anchoring across hops | Strong — same purpose, adaptive weighting |
| Depth (layers) | Multi-hop convergence rounds | Moderate — iterative refinement, but no per-layer specialization |
| Softmax | Confidence normalization | Weak — linear normalization, no exponential sharpening |
| FFN | Successor/template lookup | Weak — explicit lookup vs learned nonlinear transform |
| Positional encoding | Sentence position metadata | Weak — used in output only, not during reasoning |

The strongest correspondences are attention-as-search and residual-as-anchoring. The weakest are FFN and positional encoding. We do not claim mathematical equivalence — we claim the same architectural goals achieved through transparent primitives.

**Manual correction.** The system includes a `correct()` method that allows teaching the right answer after a query failure. This is explicit, human-triggered learning — not autonomous self-evolution. Missed queries are logged to a `misses` table for later review.

**Safety.** The architecture's strengths (minimal-example learning, perfect recall, instant transfer via file copy) are also its risks. We built in protected ethics neurons, integrity hashes, and a kill switch. Safety neurons are inspectable — you can see exactly what ethics are loaded and verify their SHA-256 integrity.

**Future work.** Larger-scale convergence-confidence validation (200+ queries). Standard benchmark evaluation for multimodal (COCO, Flickr30K). Adversarial robustness testing for the ethics gate. Tool calls from knowledge (web search triggered by convergence).

**Reproducibility.** Open source at `github.com/tejasphatak/webmind-research`. 9 source files, ~5,800 lines total. Core: `brain_core.py` (~900 lines), full engine: `engine.py` (~1,400 lines), convergence: `convergence.py` (~380 lines). Dependencies: numpy (core), sentence-transformers (multimodal). 167 tests, ~60s on CPU. Hardware: GCP e2-standard-4 (4 vCPU, 16GB RAM).

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

### A.3 Multi-Hop Convergence (from convergence.py)

```python
def reason(self, query_vector):
    query = normalize(query_vector)
    all_concepts, seen_ids = [], set()
    current_query = query.copy()
    for round_num in range(self.max_rounds):
        result = self.loop.converge(current_query)
        new_concepts = [c for c in result.concepts if c.id not in seen_ids]
        all_concepts.extend(new_concepts)
        seen_ids.update(c.id for c in new_concepts)
        if not new_concepts and round_num > 0: break
        if new_concepts:
            concept_blend = confidence_weighted_blend(new_concepts)
            w = self.concept_blend_weight  # default 0.4
            current_query = normalize((1 - w) * query + w * concept_blend)
    return MultiHopResult(converged=any_round_converged, concepts=all_concepts)
```

## Appendix B: Safety Analysis

The properties that make this system useful — minimal-example learning, perfect recall, instant transfer — are exactly the properties that make it dangerous at scale. Concept-level learning transfers to harmful domains. The knowledge base is a copyable SQLite file. The ethics layer can be bypassed by teaching contradictory principles.

**Mitigations built in:** Protected ethics neurons (SHA-256 integrity verification), kill switch, NLI-based harm detection across 50+ languages via multilingual encoder, confidence-limited learning.

**What is not sufficient:** These assume a cooperative operator. An adversary can fork the code and remove safety. The protections are speed bumps, not walls.

We publish openly because the principle (concept-level learning + perfect memory + instant transfer) will be rediscovered regardless. Better in the open with safety as a first-class concern.

## References

- Asai, A., Wu, Z., Wang, Y., Sil, A., and Hajishirzi, H. (2024). Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection. *ICLR 2024*.
- Borgeaud, S., et al. (2022). Improving Language Models by Retrieving from Trillions of Tokens. *ICML 2022*.
- Guu, K., Lee, K., Tung, Z., Pasupat, P., and Chang, M.-W. (2020). REALM: Retrieval-Augmented Language Model Pre-Training. *ICML 2020*.
- Hebb, D. O. (1949). *The Organization of Behavior*. Wiley.
- Karpukhin, V., et al. (2020). Dense Passage Retrieval for Open-Domain Question Answering. *EMNLP 2020*.
- Khandelwal, U., Levy, O., Jurafsky, D., Zettlemoyer, L., and Lewis, M. (2020). Generalization through Memorization: Nearest Neighbor Language Models. *ICLR 2020*.
- Levy, O. and Goldberg, Y. (2014). Neural Word Embedding as Implicit Matrix Factorization. *NeurIPS 2014*.
- Mitchell, T., et al. (2018). Never-Ending Learning. *Communications of the ACM*, 61(5):103-115.
- Phatak, T. (2026). INSERT INTO Is All You Need: Self-Evolving Retrieval for Factual QA. *Unpublished manuscript, available at github.com/tejasphatak/webmind-research*.
- Radford, A., et al. (2021). Learning Transferable Visual Models From Natural Language Supervision. *ICML 2021*.
- Ramsauer, H., et al. (2021). Hopfield Networks is All You Need. *ICLR 2021*.
- Robertson, S. E., Walker, S., Jones, S., Hancock-Beaulieu, M., and Gatford, M. (1994). Okapi at TREC-3. *TREC-3*.
- Shao, Z., et al. (2023). Enhancing Retrieval-Augmented Large Language Models with Iterative Retrieval-Generation Synergy. *EMNLP 2023 Findings*.
- Trivedi, H., Balasubramanian, N., Khot, T., and Sabharwal, A. (2023). Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions. *ACL 2023*.
- Turney, P. D. and Pantel, P. (2010). From Frequency to Meaning: Vector Space Models of Semantics. *JAIR*, 37:141-188.
- Vaswani, A., et al. (2017). Attention Is All You Need. *NeurIPS 2017*.
- Wang, X., et al. (2023). Self-Consistency Improves Chain of Thought Reasoning in Language Models. *ICLR 2023*.
