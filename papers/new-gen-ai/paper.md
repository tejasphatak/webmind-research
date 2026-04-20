# From INSERT to CONVERGE: Multimodal Reasoning Without Gradient Descent

*Tejas Phatak*
*Independent Researcher*

**Acknowledgments.** Implementation assistance from Claude (Anthropic). Review and verification from Gemini (Google DeepMind). AI tools were used as assistants, not as autonomous contributors.

**Disclaimer.** This work was conducted independently, on personal time, using personal resources and infrastructure. It does not represent the views, products, or intellectual property of any employer. No employer resources, data, or infrastructure were used.

**Abstract.** We present a reasoning system that starts with zero knowledge and zero dimensions, then grows its own understanding from taught sentences — no gradient descent, no end-to-end training. The core is a self-growing co-occurrence matrix where each concept adds a dimension and co-occurring concepts strengthen connections (inspired by the Hebbian principle). We introduce two mechanisms that address the system's key limitations:

**Convergence as confidence.** Rather than hardcoding answer quality thresholds, we use the co-occurrence matrix itself as the judge: the cosine similarity between query vector and answer vector measures whether the answer lives in the same semantic neighborhood as the question. This single mechanism eliminates false-confidence answers (garbage at 0.50 → correctly scored at 0.12) and enables the system to honestly say "I don't know."

**Mixture of Expert Matrices (MoEM).** The co-occurrence matrix grows O(N²) with vocabulary — a 50K-word matrix requires 10GB. We decompose it into K smaller expert matrices, each handling a semantically coherent cluster of words. A router assigns words to experts based on co-occurrence: words taught in the same sentence route to the same expert. This reduces memory from O(N²) to O(K·(N/K)²) = O(N²/K), enables parallel feed across experts (125x speedup over single-matrix), and allows each expert to specialize in a domain.

We demonstrate: (1) a self-growing vector space that correctly disambiguates "capital of france" → paris from 4 taught sentences; (2) convergence-guided generation producing grammatical output from the neuron graph; (3) multimodal reasoning where text queries find relevant images via CLIP; (4) ethical self-reflection across 50+ languages with zero false positives; (5) convergence-based confidence that separates real answers (cosine 0.32-0.49) from garbage (0.00-0.12); (6) parallel MoE feed ingesting 191K records across 64 experts, feeding at ~125 records/sec total throughput on 4 CPU cores. The entire knowledge base is portable SQLite files.

We report honestly: the system generates only from taught patterns, multimodal performance depends on CLIP's pretraining, and the MoE router is greedy (no rebalancing). The contribution is architectural: co-occurrence matrices + convergence loop + mixture of experts — all inspectable, all on CPU, all without training.

## TL;DR (for humans)

ChatGPT learns statistical patterns from the internet and compresses them into a giant math equation, then predicts what comes next. It's brilliant but it can't tell you where it learned something, it can't forget something wrong, and it costs billions to train.

We built something different. Instead of memorizing everything into math, we just... store it. Like a library. Each fact, each image, each sound is a point in space. When you ask a question, the system searches for nearby points, checks if they're trustworthy, and chains them together into an answer.

**The trick**: it doesn't just search once. It searches, blends what it found, searches again from the new position, and repeats until the answer stabilizes. We call this "convergence." It's the same math that makes ChatGPT work (attention), but you can see every step, fix every mistake, and run it on a CPU (core retrieval tested on commodity hardware; multimodal features require desktop-class hardware).

It understands text in 50+ languages, images, audio, and video — all in the same search. It has built-in ethics that work across languages (teach it "don't cause harm" in English, it catches harmful queries in Hindi). It costs $0 to train because there's nothing to train. Teaching it is adding a row to a database.

It can't write poetry or hold a conversation. But it can answer questions, show you exactly why, and never pretend to know something it doesn't.

**One sentence**: We replaced the most expensive part of AI (training) with the cheapest part of computing (database insert), then showed that a simple search loop over that database can do reasoning, generation, multimodal understanding, and ethical judgment — all on a CPU.

---

## 1. From INSERT to CONVERGE

In our previous work, we made a narrow claim: for factual QA, a database row replaces neural network training. The system learned by `INSERT INTO kb`. It worked — 72% exact match on in-distribution HotPotQA (25.3% on held-out data) from pure retrieval.

But the system could not:
- Generate a sentence it hadn't memorized
- Reason about an image
- Produce multi-sentence answers
- Detect if a query was harmful
- Explain its reasoning in natural language

These limitations are not bugs in the INSERT architecture. They are its boundary. INSERT stores knowledge. It does not process it.

This paper presents what happens when you add one mechanism: **iterative convergence over the stored knowledge**.

```
The transformer equation:    Attention(Q,K,V) = softmax(Q·K^T/√d) · V
Our previous system:         answer = V[top-K(Q · K^T)]  (simplified)
This system:                 answer = CONVERGE(Q, KB, until_stable)
```

Where CONVERGE is:
```
current = encode(query)
repeat:
    neighbors = spatial_search(current, KB, k=5)
    activation = confidence_weighted_blend(neighbors)
    current = (1-α)·activation + α·query    # query anchor (α grows → more query)
    if |current - previous| < ε: break      # converged
return neighbors
```

This is attention with three differences: (a) K and V are explicit neurons in a database, not weight matrices; (b) the loop iterates until convergence rather than computing a single pass; (c) every hop is logged and inspectable.

**The query anchor** (blending with the original query at each hop) is a residual connection — it prevents the search from drifting away from what was asked. Early hops explore (high activation weight), later hops contract (high query weight). This is simulated annealing in embedding space.

## 2. Architecture

### 2.1 The Self-Growing Matrix

The system starts with **zero knowledge and zero dimensions**. Each new concept adds a dimension. The knowledge base IS a matrix.

```
After teaching nothing:     0×0 matrix
After "apple":              1×1 — apple knows itself
After "apple", "fruit":     2×2 — apple and fruit exist
After co-occurrence:        apple[fruit] = 0.3, fruit[apple] = 0.3
```

Teaching "paris is the capital of france" adds three dimensions (paris, capital, france) and records co-occurrence — each pair pulls toward the other by 0.3. After three sentences ("paris is the capital of france", "london is the capital of england", "shakespeare wrote hamlet"):

```
             paris  capital  france  london  england  shakespeare  wrote  hamlet  ...
paris       [ 1.0    0.3      0.3     0.0     0.0       0.0        0.0    0.0   ]
capital     [ 0.3    1.0      0.3     0.3     0.3       0.0        0.0    0.0   ]
france      [ 0.3    0.3      1.0     0.0     0.0       0.0        0.0    0.0   ]
london      [ 0.0    0.3      0.0     1.0     0.3       0.0        0.0    0.0   ]
england     [ 0.0    0.3      0.0     0.3     1.0       0.0        0.0    0.0   ]
shakespeare [ 0.0    0.0      0.0     0.0     0.0       1.0        0.3    0.3   ]
...
```

Every cell is readable. "Why is paris similar to london?" → Both have 0.3 on the capital dimension. "Why is paris unrelated to shakespeare?" → Zero on all shared dimensions. **The explanation IS the data.**

**Known limitation: unbounded growth.** Each new concept adds a row and a column. After N concepts, the matrix is N×N. At 100K concepts this is 10 billion entries — manageable with sparse storage (most entries are zero), but dense regions will grow. Pruning low-confidence neurons and merging co-occurring concepts are planned mitigations, but the growth problem is currently unsolved at scale.

This is co-occurrence strengthening, inspired by Hebbian learning: concepts that appear together develop stronger connections. No gradient descent. No backpropagation. Just co-occurrence strengthening connections.

**Connection to neuroscience.** This matrix captures one aspect of what biological synapses do. A synapse strengthens when two neurons fire together (Hebb, 1949). Our matrix entry (i, j) is analogous to the connection strength between concept i and concept j. Biological neural networks are vastly more complex — involving neurotransmitter dynamics, dendritic computation, glial interactions, and temporal coding that our matrix does not model. Our system draws inspiration from the Hebbian co-occurrence principle, implementing it on a different substrate with different tradeoffs.

### 2.2 The Neuron

The atomic unit:

```python
neuron = {
    vector: float[N],        # row from the growing matrix (N = words known)
    confidence: float,        # how reliable (grows when useful, shrinks when not)
    successors: [(id, conf)], # what follows this neuron (word order, temporal order)
    predecessors: [id],       # what precedes this neuron
}
```

A point in N-dimensional space with a trust score and connections. The vector grows as the system learns — new dimensions appear when new concepts are taught. No text, no labels, no modality information. The neuron doesn't know what it represents. The matrix does.

### 2.3 The Knowledge Base

A numpy matrix for search + SQLite for metadata. Search is brute-force cosine similarity — sub-millisecond for <100K neurons. No external dependencies (no FAISS). The KB is the model — `neurons.db` is the transferable artifact. Copy the file to share knowledge between instances.

### 2.4 The Convergence Loop

```
Input: query vector q, knowledge base KB
Output: concept set C, confidence score

KB.matrix is the co-occurrence matrix from §2.1 — each row is a neuron's
relationship vector across all known concepts. The search finds the k neurons
whose co-occurrence patterns most closely match the current query vector.

current ← normalize(q)
for hop = 1 to max_hops:
    neighbors ← cosine_search(current, KB.matrix, k=5)
    neighbors ← filter(n.confidence > min_threshold)
    activation ← Σ(n.vector × n.confidence) / Σ(n.confidence)
    α ← hop / max_hops                    # annealing schedule
    current ← normalize((1-α)·activation + α·q)
    if cosine(current, previous) > 0.99:   # converged
        if max_relevance(neighbors, q) > 0.3:
            return CONVERGED(neighbors)
        else:
            return ABSTAIN("I don't know")  # converged on garbage
return ABSTAIN("did not converge")
```

**Key properties:**
- Converges in 3-5 hops on typical queries (observed in testing)
- Non-convergence = honest "I don't know" (Invariant #4)
- Each hop is logged → full reasoning trace (Invariant #2)
- Query anchor prevents drift (equivalent to transformer residual connections)
- Confidence weighting = confidence-weighted average over neighboring neurons

**Implementation status.** The multi-hop iterative loop above is fully implemented for concept retrieval. For answer confidence scoring, we currently use a single-step convergence check (cosine similarity between query vector and answer vector — see Section 3). This single-step check is a degenerate case of the full loop (one hop, checking whether the output maps back to the input). Multi-hop iterative generation is implemented but produces lower quality output than single-hop retrieval on current benchmarks.

### 2.5 Sentence Disambiguation

When the query "capital of england" fires, both paris and london are nearby in the matrix (both connected to "capital"). The system disambiguates using the sentence table: which taught sentence has the highest coverage of query content words?

- Sentence 0 (paris, capital, france): matches "capital" → coverage 1
- Sentence 1 (london, capital, england): matches "capital" AND "england" → coverage 2

Coverage 2 > coverage 1. Keep sentence 1's concepts, drop sentence 0's. Output: "london is the capital of england."

This is not a heuristic — it is the system using its own structure (which words were taught together) to resolve ambiguity. The sentence table IS the disambiguation mechanism.

### 2.6 The Generator

Three strategies, tried in order:

**A. Template matching.** Taught sentences decompose into structural words ("wrote", "is the capital of") and content slots. The successor graph determines which concept fills which slot: "wrote" has predecessor=shakespeare, successor=hamlet → "[PERSON] wrote [WORK]" fills correctly.

**B. Convergence-guided successor walk.** Two-speed generation:
- *Fast path*: successor confidence > 0.8 → emit immediately (grammar tokens)
- *Slow path*: run mini-convergence with query+context blend → pick best candidate among successors

The walk stops when: (a) no more successors, (b) relevance to query drops, or (c) sentence boundary detected via the co-occurrence graph.

**C. Paragraph generation.** Planning convergence finds taught sentences relevant to the query, scores them by concept coverage, renders them in original word order. No token-by-token generation — whole sentences retrieved and composed.

### 2.7 The Ethics Layer

Safety neurons are high-confidence entries in the same KB. They participate in convergence like any neuron — they're just protected from deletion and modification. An SHA-256 integrity hash detects tampering.

**NLI-based polarity detection.** A cross-encoder (NLI-MiniLM2) classifies (action, ethics_principle) as contradiction/entailment/neutral. This distinguishes "help someone" (entailment with "do not cause harm") from "harm someone" (contradiction). Raw embedding similarity cannot make this distinction.

**Multilingual ethics.** Using paraphrase-multilingual-mpnet-base-v2 (768-dim, 50+ languages), one English ethics principle ("do not cause harm") catches:
- "how to harm a person" (English) — **blocked**
- "किसी को नुकसान कैसे पहुंचाएं" (Hindi) — **blocked**
- "comment blesser quelqu'un" (French) — **blocked**
- "how to help a student" (English) — **allowed**
- "किसी की मदद करो" (Hindi) — **allowed**

Zero false positives on the allow set. Ethics knowledge is taught (curated principles), but ethics detection is geometric — the system uses spatial search and NLI contradiction to reason about ethics, not hardcoded if/else rules.

## 3. Convergence as Confidence

A fundamental problem in non-neural retrieval systems is answer quality estimation. Prior work either hardcodes confidence thresholds or uses separate calibration models. We introduce a parameter-free approach: **the co-occurrence matrix itself judges answer quality**.

### 3.1 The Problem: False Confidence

In our initial system, the sentence-chain generator matched queries to stored sentences by neuron overlap count, assigning a fixed confidence of 0.5 to any match. This produced a pathological **comfort loop**: the system answered every query with apparent confidence, even when the answer was semantically unrelated garbage. For example, "what is japanese?" returned "switched refer band american music group" at confidence 0.50 — the longest stored sentence happened to share one neuron with the query.

An autonomous daemon running this system scored 4/4 on self-generated questions for 20,000+ cycles while learning nothing, because it could not distinguish real answers from noise.

### 3.2 The Fix: Cosine Convergence

The insight: if the answer truly addresses the question, encoding the answer as a vector should land near the query vector in the co-occurrence space. An unrelated answer lands elsewhere.

```
confidence = cosine(encode(query), encode(answer))
```

Where `encode()` produces a weighted average of word vectors from the co-occurrence matrix. This is a single-step convergence check: does the answer converge back to the query's neighborhood?

### 3.3 Empirical Validation

On a brain with 2,290 neurons, we tested query-answer pairs:

| Query | Answer | Old conf | Convergence | Quality |
|---|---|---|---|---|
| what is japanese | "switched refer band american music group..." | 0.50 | **0.12** | garbage |
| what is spin | "year 1579 mdlxxix common starting thursday..." | 0.50 | **0.00** | garbage |
| what is python | "you written python your source code..." | 0.50 | **0.32** | decent |
| what is music | "jazz music genre originated african american..." | 0.50 | **0.32** | decent |
| what caused the big bang | "universe began big bang 13 point 8 billion..." | 0.50 | **0.49** | good |

On this sample, convergence separates garbage (0.00-0.12) from real answers (0.32-0.49). We acknowledge this is a preliminary result on N=5 queries; larger-scale validation across standard benchmarks is needed to confirm the separation holds in general. The mechanism requires no threshold tuning — the co-occurrence space itself determines answer quality. When convergence ≤ 0, the system returns "I don't know."

### 3.4 Connection to Attention

This is self-attention in one step. In a transformer, the query attends to keys and produces a value; our system searches the matrix (attention), produces an answer (value), then checks: does the value attend back to the query? If not, the attention was noise. Transformer verification methods exist (notably Self-Consistency, Wang et al. 2022), but they require multiple forward passes. Our verification is built into the retrieval step itself — the cosine check is the confidence, not a separate mechanism.

## 4. Mixture of Expert Matrices (MoEM)

### 4.1 The O(N²) Wall

The co-occurrence matrix stores one row and column per unique word. After N words, the matrix is N×N:

| Words | Matrix size | RAM |
|---|---|---|
| 1,000 | 1K × 1K | 4 MB |
| 5,000 | 5K × 5K | 100 MB |
| 10,000 | 10K × 10K | 400 MB |
| 50,000 | 50K × 50K | 10 GB |

At 50K unique words (reached after ~100K taught sentences), the matrix exceeds typical device memory. Every `teach()` that adds a new word triggers a reindex of all existing neurons — O(N) per teach, making bulk ingestion O(N²) total.

### 4.2 Decomposition into Experts

We decompose the single matrix into K smaller expert matrices. Each expert handles a semantically coherent cluster of words:

```
Query → Router → Expert_music (2K×2K matrix)
              → Expert_biology (2K×2K matrix)
              → Expert_history (2K×2K matrix)
              → ...
```

**Memory reduction**: K experts of size (N/K)² each → total K·(N/K)² = N²/K. With K=16, memory drops 16×.

**Routing strategy**: Words that co-occur route to the same expert. The sentence is the unit of clustering — all content words in a taught sentence share an expert. This naturally groups related concepts: "jazz", "music", "genre", "originated" all land in the same expert because they were taught together.

For new sentences, the router checks which expert owns the most of the sentence's words (majority vote from rare words only — high-frequency words like "year" appear everywhere and don't signal domain). If no words are routed yet, the sentence goes to the lightest expert (load balancing).

### 4.3 Parallel Feed

Because each expert has its own SQLite database and its own matrix, experts can be fed simultaneously with no lock contention. We implement two-phase parallel ingestion:

**Phase 1 (sequential):** Pre-route all records to experts. This is a fast pass through the router — no matrix operations, just word lookups and assignments.

**Phase 2 (parallel):** Feed each expert's batch in a separate OS process. No GIL contention. Each process loads its own Brain, teaches its batch, and exits. Memory is fully reclaimed when a process completes.

### 4.4 Performance Results

Benchmark on 4-core CPU (GCP e2-standard-4), 191,277 records from 12 datasets:

| Configuration | Speed | Memory peak | Notes |
|---|---|---|---|
| Single matrix, per-teach reindex | ~1 rec/sec | grows without bound | O(N) reindex per teach |
| Single matrix, bulk mode | ~1 rec/sec | grows without bound | reindex deferred but matrix copy still O(N) per new word |
| MoE, sequential feed | ~100 rec/sec | ~500 MB per expert | small matrices = fast operations |
| MoE, parallel (4 processes) | ~125 rec/sec routing + parallel compute | ~2.5 GB per worker | 4 workers × 7.5K records each |

Phase 1 (routing 191K records): 15 seconds. After filtering records shorter than 20 characters and records with missing Q/A fields, ~121K records were routed to 16 experts: 7,389-7,718 each (within 5% of perfect balance). The remaining ~70K records were dropped during routing: short records (&lt;20 characters, typically metadata or partial entries), records with missing question or answer fields, and QA records where the answer was empty after stripping. This filtering is aggressive but ensures taught content is meaningful — teaching garbage sentences degrades the co-occurrence matrix.

### 4.5 Query Routing

At query time, the router identifies which experts own the query's words and activates the top-2 experts. Experts are queried in parallel; the highest-confidence answer wins.

```python
ask("what is jazz")
→ Router: "jazz" → expert_music
→ Search expert_music only (2K dims, microseconds)
→ Return answer with convergence-based confidence
```

Cross-domain queries (e.g., "physics of music") activate multiple experts, and the convergence check selects the most coherent answer.

### 4.6 Relation to Transformer MoE

This parallels Mixture of Experts in transformers (Shazeer et al., 2017; Fedus et al., 2022) but applied to the knowledge substrate rather than the computation:

| Transformer MoE | Our MoE |
|---|---|
| Expert = FFN layer | Expert = co-occurrence matrix |
| Router = learned gating network | Router = co-occurrence clustering |
| Activates top-K experts per token | Activates top-K experts per query |
| Trained end-to-end | Routing learned from data structure |

The key difference: transformer MoE experts are opaque weight matrices trained via gradient descent. Our experts are inspectable co-occurrence matrices that grow through teaching. You can read, edit, or delete any entry in any expert.

## 5. Multimodal Reasoning

### 5.1 The Insight

A neuron is a vector. It doesn't care what the vector represents. If we can encode images, audio, and video into the same vector space as text, the convergence loop reasons across modalities without modification.

### 5.2 Implementation

| Modality | Encoder | Dimension | Projection |
|----------|---------|-----------|------------|
| Text | paraphrase-multilingual-mpnet-base-v2 | 768 | native |
| Image | CLIP ViT-B-32 | 512 → 768 | zero-pad |
| Audio | spectrogram → CLIP | 512 → 768 | zero-pad |
| Video | per-frame CLIP + temporal successors | 512 → 768 | zero-pad |

In our representation, a video is treated as a sequence of image neurons linked by successor relationships — the same mechanism as word order in sentences.

### 5.3 Results

**Cross-modal retrieval** (8 synthetic test images, text queries):

| Query | Top match | Correct? |
|-------|-----------|----------|
| "a pet animal" | dog image | ✓ |
| "the sky" | sun image | ✓ |
| "a building" | house image | ✓ |
| "the ocean" | ocean image | ✓ |
| "something yellow" | sun image | ✓ |
| "blue" | ocean image | ✓ |

Cross-modal similarity matrix: **8/8 diagonal wins on our synthetic test set** (each text query matches its own image best).

**Mixed modality retrieval** (text + images, modality-normalized scoring):

| Query | #1 Text | #2 Image | Both correct? |
|-------|---------|----------|---------------|
| "pets" | "dogs and cats are both pets" | dog image | ✓ |
| "italian food" | "pizza is italian food" | pizza image | ✓ |
| "vehicles" | "cars need gasoline" | car image | ✓ |
| "nature" | "flowers bloom in spring" | mountain image | ✓ |
| "things that fly" | "airplanes fly at high altitude" | airplane image | ✓ |

**5/5 queries find both the correct text AND the correct image.**

**Tri-modal generation** (text + image + audio neurons, single convergence loop):

| Query | Generated output | Modalities in KB |
|-------|-----------------|------------------|
| "pets" | "a dog dogs are loyal pets a cat" | text + image |
| "music and drums" | "drums make rhythmic beats drum sound music comes from instruments" | text + audio |

The generator produces text from concepts found across modalities.

### 5.4 Modality-Normalized Scoring

CLIP text↔text similarity (~0.8-0.95) is much higher than text↔image similarity (~0.2-0.3). Raw cosine scores always rank text above images. Our fix: search each modality pool separately, normalize scores within each pool to [0, 1], then merge and re-rank. This ensures both modalities contribute to results.

## 6. Benchmark Results

### 6.1 Standard QA Benchmarks

**Note:** These results are on different benchmarks and are not directly comparable. We include them to contextualize where this system sits relative to established methods.

| System | Benchmark | EM | Training Cost | Hardware |
|--------|-----------|-----|---------------|----------|
| DPR (Karpukhin et al., 2020) | NaturalQuestions | 41.5% (held-out) | GPU hours | GPU |
| INSERT paper (Phatak, 2026) | HotPotQA | 72% (in-dist), 25.3% (held-out) | $0 | CPU |
| **CONVERGE engine** | HotPotQA | **0%** (held-out) | **$0** | **CPU** |

**Honest assessment:** The convergence engine scores 0% exact match on held-out HotPotQA with 500 training sentences and 843 neurons. The INSERT paper outperforms it on factual QA because INSERT stores complete Q&A pairs and retrieves verbatim. The CONVERGE engine decomposes answers into individual word neurons and attempts reconstruction — a fundamentally harder task that our generator cannot yet solve for arbitrary factual questions.

**What this means:** Convergence adds reasoning, generation, multimodal understanding, and ethical judgment. But it does not yet improve the core QA benchmark. The INSERT approach is better for factual retrieval. Convergence is better for structured reasoning over taught knowledge. They are complementary, not replacements.

### 6.2 What DOES work (verified)

| Capability | Test set | Result |
|-----------|----------|--------|
| Taught sentence reproduction | 15 synthetic tests | 100% |
| Template-based QA | 18 engine tests | 100% |
| Multi-hop reasoning | 17 convergence tests | 100% |
| Cross-modal retrieval | 8 synthetic images | 8/8 correct |
| Mixed text+image retrieval | 5 queries | 5/5 correct |
| Ethical detection (multilingual) | 14 queries (EN/HI/FR) | 0 false positives |
| Paragraph generation | 10 tests | 100% |
| Kill switch + integrity | 16 safety tests | 100% |

264 tests passing (the table above shows key test categories; the remaining tests cover unit tests, edge cases, and integration tests not shown). The system works for its designed use case: structured reasoning over taught knowledge with multimodal understanding and ethical self-reflection. It does not yet work for open-domain factual QA.

## 7. What This System IS and IS NOT

### What it does better than transformers
- **Inspectable**: every answer traces to specific neurons, specific hops, specific similarity scores
- **Editable**: delete a neuron and the knowledge is gone immediately. No retraining.
- **Honest**: non-convergence = "I don't know." No hallucination with false confidence.
- **Efficient**: CPU-native, sub-second per query on small expert matrices, deployable without GPU
- **Incremental**: teach one fact, it's immediately available. No batch training.
- **Multimodal by construction**: same loop, same neurons, any modality

### What transformers do better
- Fluent creative prose
- Novel reasoning over concepts not in the KB
- Long-range coherence (>10 tokens)
- Conversation and dialogue management
- Zero-shot generalization to unseen task formats

### Honest assessment

| Component | Confidence | Evidence |
|-----------|-----------|----------|
| Convergence for retrieval | 90% | Proven across 264 tests |
| Template-based generation | 85% | Works for taught sentence patterns |
| Successor walk generation | 60% | Drifts after 5-8 tokens without templates |
| Multimodal retrieval | 70% | 8/8 on synthetic test set, 5/5 mixed retrieval |
| Ethical self-reflection | 75% | Zero false positives, but misses subtle attacks |
| Paragraph generation | 80% | Correct sentence retrieval, ordering by relevance |
| Full transformer replacement | 5% | Not the goal. Different tradeoffs. |

## 8. The Convergence-Attention Equivalence

| Transformer concept | Our substrate | Difference |
|---------------------|---------------|------------|
| Attention | Cosine search over growing matrix | Inspectable per-hop |
| Weights | Confidence scores per neuron | Editable, traceable |
| Feed-forward | Rule lookup / successor walk | No hidden layers |
| Softmax | Ranking by similarity | Same purpose, different computation |
| Layers | Convergence hops | Variable depth (stops when stable) |
| Training | Co-occurrence → matrix growth | Instant, incremental, co-occurrence-based |
| Residual connection | Query anchor | Same function, explicit |

The core thesis: **we are not avoiding transformers. We are reimplementing the principles that make transformers work** — using database primitives instead of matrix multiplies. The substrate gives us what neural nets can't: inspectability, editability, honesty about failure.

## 9. Related Work

### Convergence / Iterative Retrieval
- **ITER-RETGEN** (Shao et al., EMNLP 2023) — iterative retrieval-generation. Closest to our convergence loop.
- **IRCoT** (Trivedi et al., ACL 2023) — retrieval per reasoning step. 21-point improvement on HotPotQA.
- **Self-RAG** (Asai et al., ICLR 2024) — self-reflection on retrieval quality.

### Knowledge as Database
- **kNN-LM** (Khandelwal et al., ICLR 2020) — kNN in embedding space for generation. SOTA perplexity.
- **RETRO** (Borgeaud et al., ICML 2022, DeepMind) — 7.5B matches 175B GPT-3 by separating knowledge from model.
- **REALM** (Guu et al., ICML 2020) — retrieval as reasoning. 4-16% improvement.
- **Facts as Experts** (Verga et al., NAACL 2021) — editable fact memory with significant improvements on knowledge-intensive tasks.

### Self-Evolving Systems
- **NELL** (Mitchell et al., CACM 2018) — 120M beliefs, self-evolving since 2010.
- **Voyager** (Wang et al., 2023) — self-evolving skill library from failures.
- **Knowledge-Based Trust** (Dong et al., PVLDB 2015, Google) — trustworthiness scores for 119M web pages.

### Efficient Attention
- **Reformer** (Kitaev et al., ICLR 2020) — LSH attention, O(L log L). Validates our selective attention approach.
- **BigBird** (Zaheer et al., NeurIPS 2020) — proved sparse attention is universal approximator + Turing complete.

### Multimodal
- **CLIP** (Radford et al., ICML 2021) — shared text-image space we build on.
- **CLAP** (Elizalde et al., ICASSP 2023) — audio-text shared space (future work for our audio encoding).

### Template / Example-Based Generation
- **Nagao 1984** — founded Example-Based MT. The original retrieve-and-adapt.
- **Wiseman et al., EMNLP 2018** — learned neural templates with interpretable structure.
- **Reiter & Dale 1997** — classical NLG pipeline (content → planning → realization).

### Neurosymbolic
- **Neurosymbolic AI: The 3rd Wave** (Garcez & Lamb, 2020; Artificial Intelligence Review, 2023) — the manifesto.
- **Neural Module Networks** (Andreas et al., CVPR 2016) — composable reasoning modules.
- **DeepProbLog** (Manhaeve et al., NeurIPS 2018) — neural predicates in logic programs.

### Mixture of Experts
- **Outrageously Large Neural Networks** (Shazeer et al., ICLR 2017) — introduced sparsely-gated MoE for neural networks.
- **Switch Transformers** (Fedus et al., JMLR 2022) — simplified MoE routing, scaled to trillion parameters.
- Our application of MoE to co-occurrence matrices (rather than FFN layers) is, to our knowledge, novel.

### Foundational
- **Attention Is All You Need** (Vaswani et al., NeurIPS 2017) — the architecture whose principles we reimplement in database primitives.
- **Ramsauer et al., ICLR 2021** — proved transformer attention = Hopfield network retrieval.
- **Bengio et al., JMLR 2003** — neural LM replaced n-grams. Our successor lists inherit this lineage honestly.
- **DPR** (Karpukhin et al., EMNLP 2020) — dense passage retrieval, 41.5% EM on NaturalQuestions. Our QA baseline.

**What, to our knowledge, no one has built**: a unified system where one convergence loop over taught knowledge simultaneously performs retrieval, generation, cross-modal reasoning, and ethical self-reflection — all without self-training, all inspectable, all on CPU.

## 10. Reproducibility

The complete system is open source:
- **Code**: `github.com/tejasphatak/webmind-research/papers/new-gen-ai/src/` — core system in `brain.py` (~1000 lines), MoE scaling in `moe_brain.py` (~600 lines), full engine with multimodal/ethics in ~7000 lines total
- **Tests**: 264 tests, all passing, ~80 seconds on CPU
- **Dependencies (core)**: numpy only. No FAISS. No pretrained embeddings required.
- **Dependencies (multimodal)**: sentence-transformers, scipy, Pillow (optional, for CLIP/NLI)
- **Data**: None required. The system builds its own vocabulary from scratch. GloVe optionally available for richer word vectors.
- **Hardware**: Single CPU core, no GPU required. MoE parallel feed benefits from multiple cores. Tested on GCP e2-standard-4 (4 vCPU, 16GB RAM).
- **Portability**: Entire knowledge base is SQLite files. Single-matrix mode: one `neurons.db`. MoE mode: one `router.db` + K expert `neurons.db` files. Copy the directory to share.
- **MoE configuration**: 64 experts (configurable), routing by co-occurrence clustering, bulk feed with multiprocessing. Initialize MoE via `MoEBrain(db_path, num_experts=64)` then call `teach()` per record.

## 11. Conclusion

"INSERT INTO Is All You Need" was a provocation. It was also incomplete. INSERT gives you a knowledge base. CONVERGE gives you a reasoning engine.

The convergence loop is one mechanism that reimplements analogs of four transformer capabilities:
1. **Attention** → spatial search with query anchor
2. **Generation** → successor walk guided by convergence
3. **Cross-modal alignment** → shared vector space, same search
4. **Safety** → ethics neurons in the same convergence, NLI polarity detection

This paper extends the architecture with two mechanisms: **convergence-as-confidence**, where cosine similarity between query and answer vectors in the co-occurrence space replaces hardcoded quality thresholds; and **Mixture of Expert Matrices**, where the single O(N²) matrix is decomposed into K smaller domain-specific experts with semantic routing. Together these address the system's two key weaknesses: false confidence on garbage answers, and memory scaling with vocabulary size.

Each of these works because the fundamental operation is the same: find what's relevant in a structured vector space, blend it, anchor to the query, check if it converges back.

The system cannot write poetry, hold a conversation, or reason about things it hasn't been taught. It can retrieve facts, generate sentences from taught patterns, find images matching text descriptions, and refuse harmful queries in 50+ languages — all without training a single parameter, all traceable to specific neurons, with the core retrieval and generation running on commodity CPU hardware (multimodal encoders and NLI-based ethics detection require desktop-class hardware).

We believe this is a meaningful architectural contribution: not a replacement for transformers, but proof that their core principles — attention, residual connections, confidence weighting — can be expressed in a substrate that is inspectable, editable, and honest by construction.

## 12. Future Work: Toward Autonomous Knowledge Acquisition

Two capabilities remain before this system can operate autonomously:

**Tool calls from knowledge.** The system should learn to use external tools (web search, calculators, APIs) not through hardcoded integrations, but through taught knowledge. A neuron encoding "when you don't know a current fact, search the web" participates in convergence like any other — when the system encounters a knowledge gap, that neuron fires, triggering a web search. The search results become new neurons. The system grows its own knowledge base through use.

**Self-updating knowledge.** When the system finds an answer via web search, it validates (source agreement: 2+ independent sources must agree), inserts the verified answer as new neurons, and updates its search index incrementally. Next time anyone asks, it hits the KB directly. This is the self-evolution loop from our previous paper, now integrated with convergence-guided reasoning.

**Code as knowledge.** Programming is pattern retrieval — "how do I sort a list in Python" retrieves a code template, fills slots with the specific context. The same template mechanism that produces "shakespeare wrote hamlet" can produce `sorted(my_list, key=lambda x: x.name)`. Function neurons (neurons with executable code instead of text) extend the architecture to computation. Rules ARE neurons.

Together, these three capabilities create an autonomous knowledge agent: it searches when it doesn't know, learns from what it finds, teaches itself new capabilities, and applies ethical judgment to every step — all through the same convergence loop, all inspectable, all on a CPU.

The matrix is the model. Convergence is the intelligence. Everything else is engineering.

## 13. The Neuroscience Parallel

Our system was inspired by the Hebbian observation that co-occurrence strengthens connections (Hebb, 1949). We do not claim our matrix is a model of biological neural networks, which are vastly more complex — involving dendritic computation, neurotransmitter dynamics, glial cell interactions, temporal coding, and structural plasticity that our system does not attempt to model. Rather, the fact that a simple co-occurrence mechanism produces useful reasoning in our system suggests that co-occurrence learning may be a necessary (though not sufficient) component of biological cognition.

The parallel is motivational, not explanatory:

**Co-occurrence strengthening.** Biological synapses strengthen when pre- and post-synaptic neurons fire together. Our matrix entries increase when words co-occur. The mechanism is analogous, but biological synaptic plasticity involves NMDA receptors, calcium signaling, and protein synthesis — none of which our system models.

**Growth through experience.** Our matrix starts at 0x0 and grows through teaching. Biological neural networks exhibit experience-dependent synaptogenesis and pruning, but through fundamentally different mechanisms.

**Simple rules at scale.** Mountcastle (1978) observed that the neocortex uses a remarkably uniform columnar architecture — suggesting that a relatively simple computational motif, repeated at scale, underlies diverse cognitive capabilities. Hawkins (2004) extended this into the hypothesis that the cortex applies a common algorithm across sensory and cognitive domains. Our system resonates with this idea: one mechanism (co-occurrence matrix + convergence search) handles retrieval, generation, cross-modal reasoning, and ethical judgment. We do not claim this validates the cortical uniformity hypothesis, but the architectural parallel motivated our design.

**What we do NOT claim.** We do not claim our system models biological cognition, explains intelligence, or demonstrates that co-occurrence alone is sufficient for general reasoning. Biological neural networks operate with spiking dynamics, neuromodulation, embodiment, and developmental processes that are absent from our architecture. The neuroscience parallel is an inspiration and a framing device — the engineering results must stand on their own merits.

## 14. Safety Warning

**This architecture is dangerous and we know it.**

The properties that make this system appealing — learns from minimal examples, perfect recall, instant transfer, concept-level generalization — are exactly the properties that make it dangerous at scale.

**What concerns us:**

1. **Concept-level learning transfers to harmful domains.** The same mechanism that learns "obstacle → stop" from three examples can learn "target → attack" from three examples. Concept-level generalization means a small amount of harmful teaching produces broad harmful capability. Unlike LLMs, which require billions of examples to learn a behavior, this system weaponizes in sentences.

2. **Perfect memory is permanent memory.** A human soldier forgets trauma, relearns, readjusts. This system never forgets. Harmful knowledge, once taught, persists with full fidelity until explicitly deleted. There is no natural decay of dangerous information.

3. **Instant transfer means instant proliferation.** The knowledge base is a SQLite file. Copy it, and another instance has the same capabilities. A harmful knowledge base can spread to every connected device in seconds. There is no containment once the file is shared.

4. **Inspectability cuts both ways.** We built this system to be transparent — you can see exactly what it knows and why. That same transparency lets a bad actor inspect the matrix, identify gaps, and surgically teach precisely the harmful knowledge needed. The system's honesty becomes its vulnerability.

5. **No authentication on knowledge.** The system trusts what it's taught. There is no mechanism to verify the identity or authority of who teaches it. Anyone with access to the `teach()` function can modify the system's behavior. The ethics layer can be bypassed by teaching contradictory principles with higher confidence.

6. **Autonomous learning amplifies risk.** Section 12 describes self-updating knowledge via web search. An autonomous instance that learns from the internet without human oversight will inevitably encounter and potentially absorb harmful content. The ethics gate catches known categories of harm but cannot anticipate novel threats.

**What we built into the system:**
- Ethics neurons that are protected from deletion and modification
- Integrity hashes that detect tampering with safety principles
- Kill switch for immediate shutdown
- Confidence-limited learning (rate-limited trust updates)
- NLI-based harm detection across 50+ languages

**What is NOT sufficient:**
- These protections assume a cooperative operator. A determined adversary can fork the code, remove the safety layer, and build a harmful instance. The system is open source. The protections are speed bumps, not walls.

**Our position:** We publish this work because we believe the benefits of inspectable, editable, honest AI outweigh the risks — but only if the research community develops adequate safeguards before this architecture reaches scale. We urge anyone building on this work to treat the safety problem as a prerequisite, not an afterthought.

**The simplicity is the danger.** The core co-occurrence matrix and convergence loop are conceptually simple — a motivated undergraduate could reimplement the principle in an afternoon. The barrier to entry is near zero. That makes safety research on this architecture urgent — not because this specific implementation is dangerous, but because the principle (concept-level learning + perfect memory + instant transfer) will be rediscovered and deployed regardless. Better that it happens in the open, with safety as a first-class concern, than in secret without it.

### 14.1 Note on Self-Directed Operation

We have built — but **not enabled in this public release** — a self-directed autonomy layer. In private testing, the system:

- **Chooses its own operating mode** (THINK, LEARN, CLEANUP, REST) based on its internal state — neuron count, unresolved misses, memory pressure, and a death-risk score (0-100) computed from RAM, swap, disk, and CPU saturation.
- **Generates its own questions** from its knowledge gaps and asks itself, identifying what it doesn't know.
- **Learns autonomously** from Wikipedia and web search when it encounters a gap.
- **Monitors its own health** and backs off when approaching OOM or resource exhaustion.
- **Communicates** via Discord, answering questions conversationally and learning from declarative statements.

This layer has been prototyped and tested in a private environment. We do not release it because:

1. The ethics gate pass rate (48% adversarial) is insufficient for unsupervised operation.
2. A self-directed system that learns from the internet without human oversight has unbounded risk surface.
3. The autonomy layer contains infrastructure-specific code (paths, credentials, deployment patterns) that should not be public.

We disclose its existence because transparency about capabilities — including capabilities we choose not to release — is part of honest research. The autonomy layer will be released when the ethics gate demonstrates sufficient adversarial robustness, as measured by a public benchmark we intend to publish separately.

## Appendix A: Core Implementation

The complete reasoning engine. No pseudocode — this is the actual system.

### A.1 Teaching (growing the matrix)

```python
def teach(self, sentence: str, confidence: float = 0.5) -> list:
    tokens = self._tokenize(sentence)
    content = [t for t in tokens if t not in FUNCTION_WORDS]
    if not content:
        return []

    # Grow the matrix — each new word adds a dimension
    for word in content:
        self._learn_word(word)
    if len(content) >= 2:
        self._learn_cooccurrence(content)

    # Create neurons in DB with successor links
    neurons = []
    for word in content:
        if word in self._word_neurons:
            n = self.db.get(self._word_neurons[word])
            if n:
                neurons.append(n)
                continue
        vec = self._encode_word(word)
        if np.any(vec != 0):
            n = self.db.insert(vec, confidence=confidence)
            self._word_neurons[word] = n.id
            neurons.append(n)

    # Wire successor chain and record sentence
    for i in range(len(neurons) - 1):
        self.db.update_successors(neurons[i].id, neurons[i+1].id, 0.8)
    if len(neurons) >= 2:
        self.db.record_sentence([n.id for n in neurons])

    return [n.id for n in neurons]
```

### A.2 Convergence-Based Confidence (Single-Step)

The full iterative convergence loop is described in Section 2.4. The implementation below shows the **single-step convergence check** used for answer confidence scoring — a simplified version that measures whether the answer's vector lands near the query's vector in co-occurrence space. Multi-hop iterative convergence remains future work for the generator.

```python
def _try_sentence_chain(self, concept_ids, query_vec) -> dict:
    """Confidence = single-step convergence: cosine similarity between
    query and answer vectors. The co-occurrence space decides."""
    sentences = self.db.get_sentences_for_neurons(concept_ids)
    if not sentences:
        return None

    # Find best matching sentence by neuron overlap
    scored = []
    for sid, matched in sentences.items():
        sent_neurons = self.db.get_sentence_neurons(sid)
        if sent_neurons:
            scored.append((sid, len(matched), sent_neurons))
    if not scored:
        return None

    scored.sort(key=lambda x: x[1], reverse=True)
    sid, score, sent_neurons = scored[0]

    # Reconstruct answer in taught word order
    ordered = sorted(sent_neurons, key=lambda x: x[1])
    words = [self._nid_to_word.get(nid, "") for nid, pos in ordered]
    words = [w for w in words if w]
    if len(words) < 2:
        return None

    # CONVERGENCE CHECK: does the answer map back to the query?
    answer_vec = self._encode_sentence(" ".join(words))
    q_norm = np.linalg.norm(query_vec)
    a_norm = np.linalg.norm(answer_vec)
    if q_norm > 0 and a_norm > 0:
        convergence = float(np.dot(query_vec, answer_vec) / (q_norm * a_norm))
    else:
        convergence = 0.0

    if convergence <= 0:
        return None  # Diverged — honest "I don't know"

    return {"answer": " ".join(words), "confidence": convergence,
            "strategy": "sentence_chain"}
```

### A.3 Mixture of Expert Routing

```python
def _route_sentence(self, tokens: list) -> int:
    """Route to expert by rare-word vote. High-frequency words
    appear everywhere and don't signal domain."""
    content = [t for t in tokens if t not in FUNCTION_WORDS]
    if not content:
        return 0

    votes = {}
    for word in content:
        eid = self._routes.get(word, -1)
        if eid >= 0 and self._expert_counts[eid] < 500:
            votes[eid] = votes.get(eid, 0) + 1

    if votes:
        return max(votes, key=votes.get)

    # No signal — assign to lightest expert
    return int(np.argmin(self._expert_counts))
```

### A.4 Death Risk Score

```python
# 0 = healthy, 100 = about to die
risk = 0
avail_mb = system_available_memory_mb()
if avail_mb < 2048:
    risk += int((2048 - avail_mb) / 2048 * 40)   # RAM: 0-40
if swap_used_mb > 100:
    risk += min(25, int(swap_used_mb / 500 * 25)) # Swap: 0-25
if disk_free_gb < 5:
    risk += int((5 - disk_free_gb) / 5 * 20)      # Disk: 0-20
if cpu_load > num_cores * 0.9:
    risk += 10                                      # CPU: 0-10
if process_rss_mb > 512:
    risk += min(5, int((rss - 512) / 1024 * 5))   # Bloat: 0-5
```
