# High-Level Design: New-Gen-AI

## What We're Building

A reasoning engine that reimplements transformer principles using database, spatial search, and convergence primitives. Not a neural network. Not trained with gradient descent. Every answer traceable, every fact editable, every failure honest.

**Target users:** Anyone who needs trust over fluency — doctors, lawyers, teachers, regulators, offline schools.

---

## The Atomic Unit: Neuron

```
neuron = {
    id:           int           # unique identifier
    vector:       float[384]    # position in concept space (from pretrained embeddings)
    confidence:   float         # how reliable (grows when useful, shrinks when not, capped ±0.8)
    successors:   [(id, conf)]  # top-10 observed successors (what follows this neuron)
    predecessors: [id, id, id]  # top-3 predecessors (what came before — derives context)
    timestamp:    int           # when last verified
    temporal:     bool          # true = time-sensitive knowledge, false = timeless
    level:        enum          # character | word | concept
}
```

- A point in space with a trust score. No text labels stored.
- Confidence is persisted — grows on useful fire (×1.1), shrinks on useless fire (×0.9).
- Successors are evicted top-K: new successor competes for slot, replaces lowest if better. Fixed 80 bytes per neuron.

---

## System Architecture

```
                    ┌─────────────────────────────────┐
                    │           QUERY INPUT            │
                    └──────────────┬──────────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────────┐
                    │            ENCODER               │
                    │   text → vector(384)              │
                    │   (pretrained word embeddings)    │
                    └──────────────┬──────────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────────┐
                    │       CONVERGENCE LOOP            │
                    │                                   │
                    │   1. Search nearest neurons       │
                    │   2. Fire → collect activations    │
                    │   3. Blend with query anchor       │
                    │   4. Repeat until stable or abort  │
                    │                                   │
                    │   Converged = answer found         │
                    │   Not converged = "I don't know"   │
                    └──────────────┬──────────────────┘
                                   │
                          ┌────────┴────────┐
                          ▼                 ▼
                   ┌────────────┐    ┌────────────┐
                   │  CONCEPTS  │    │  NO ANSWER  │
                   │  (neurons  │    │  (honest    │
                   │  that      │    │   abstain)  │
                   │  converged)│    └────────────┘
                   └─────┬──────┘
                         │
                         ▼
                   ┌─────────────────────────────────┐
                   │          GENERATOR                │
                   │                                   │
                   │   Two-speed output:               │
                   │   • Grammar tokens: successor     │
                   │     lookup (fast, conf > 0.8)      │
                   │   • Content tokens: convergence    │
                   │     loop per token (slow, accurate)│
                   │                                   │
                   │   Template matching for fluency:   │
                   │   concepts → closest template →    │
                   │   fill slots → output              │
                   └──────────────┬──────────────────┘
                                  │
                                  ▼
                   ┌─────────────────────────────────┐
                   │         FEEDBACK LOOP             │
                   │                                   │
                   │   Layer 1: Self-consistency (10%) │
                   │   Layer 2: User behavior (always) │
                   │   Layer 3: External verify (idle) │
                   │                                   │
                   │   Updates neuron confidence        │
                   └─────────────────────────────────┘
```

---

## Component Details

### 1. Encoder

Converts input text to vector(384).

- **Word-level:** Pretrained embeddings (GloVe 400K or FastText). Each word maps to a point in 384-dim space.
- **Sentence-level:** Average of word vectors, weighted by position (later: attention-weighted).
- **No training.** Embeddings are loaded, not learned. Like using a calculator — the map is borrowed, confidence is earned.

### 2. Neuron Database

All neurons stored in:
- **FAISS index** — spatial search, O(log N) for nearest neighbors
- **SQLite** — metadata (confidence, successors, predecessors, timestamps)

Operations:
- `search(vector, k)` → k nearest neurons
- `insert(vector, metadata)` → add neuron
- `delete(id)` → gone. Immediately. No retraining.
- `update_confidence(id, delta)` → adjust trust score

**Proximity IS connection.** No explicit wiring graph. Two neurons are "connected" if they're close in vector space. The spatial structure IS the network topology.

### 3. Convergence Loop (The Core)

This is what replaces attention in transformers. Multi-hop reasoning:

```
def converge(query_vector, max_hops=10):
    current = query_vector
    for hop in range(max_hops):
        neighbors = db.search(current, k=5)
        
        # Blend neighbors weighted by confidence
        activation = weighted_average(neighbors, weights=confidences)
        
        # Anchor to query (prevents drift)
        # Early hops: explore (more activation)
        # Later hops: contract (more query)
        alpha = hop / max_hops  # 0→1 as hops increase
        current = (1 - alpha) * activation + alpha * query_vector
        
        # Check convergence: has the vector stopped moving?
        if cosine_sim(current, previous) > 0.99:
            return current, neighbors  # CONVERGED
    
    return None  # DID NOT CONVERGE → "I don't know"
```

**Key properties:**
- Query anchor = residual connection (transformers). Keeps the original signal alive.
- Convergence check = stopping criterion. No convergence = honest abstention.
- Each hop is inspectable — you can trace exactly why the answer was found.

### 4. Generator

Converts converged concepts into text output. Three strategies, tried in order:

#### Strategy A: Template Matching (primary, 75% confidence)
```
template_neuron = {
    vector:     embedding of the pattern
    pattern:    "[PERSON] [VERB-past] [WORK] in [YEAR]"
    confidence: learned from frequency
    slots:      {PERSON: "proper_noun", VERB: "verb", WORK: "noun", YEAR: "number"}
}
```

Flow: concepts → search for closest template → fill slots with converged neurons → output.
Templates extracted from corpus: NER identifies entities, remainder = template.

#### Strategy B: Successor Walk (secondary, 20% confidence)
Walk through successor lists, one token at a time:
- Current neuron → pick highest-confidence successor → emit → repeat
- Two-speed: if successor confidence > 0.8 → emit immediately (grammar token). Else → full convergence loop (content token).

#### Strategy C: Concept List (fallback, 95% confidence)
Return raw concept neurons as structured output. No fluency, but always correct.
```
{concepts: ["Shakespeare", "wrote", "Hamlet", "1600"], confidence: 0.92}
```

### 5. Context (Mini-KB)

Every fired neuron in a conversation becomes a temporary KB entry. Context = spatial query against all previous neurons, ranked by relevance — not recency.

- Topic-shift detection: sliding window centroid over last N queries. Old entries pruned when centroid drifts.
- Unreferenced neurons age per query, drop after 5.
- No fixed context window. No decay. Scales with conversation length.

### 6. Feedback Loop

Three layers, time-stratified:

| Layer | What | When | Cost |
|-------|------|------|------|
| Self-consistency | Ask same thing differently, compare | 10% sampling | Low |
| User behavior | Follow-up = insufficient, new topic = accepted | Always | Free |
| External verify | Re-check high-confidence neurons against sources | Idle time only | Moderate |

Confidence changes:
- Useful fire → confidence × 1.1
- Useless fire → confidence × 0.9
- Capped at ±0.8 to prevent mode collapse
- Only sticks if neuron participates in converging chains in subsequent queries

### 7. Error Correction

- Divergence in convergence loop at token N+1 = token N was wrong
- Backtrack: drop token N, penalize its neuron (×0.9), try second-best candidate
- Depth-1 only. If second-best also fails → honest abort, emit partial answer
- Backtrack > 3 times → "I'm not confident enough"

### 8. Rule Engine (Function Neurons)

Rules live in the same space as data neurons. They compete on confidence, fire through the same network.

```
function_neuron = {
    vector:     embedding of the operation (e.g., "addition")
    confidence: earned through use
    function:   (input) → output  # inspectable source code, NOT weights
}
```

- Route by query type: numeric → rule engine, not KB
- Rules ARE neurons. Same search, same confidence, same space.
- Example: `def add(a, b): return a + b` — readable, editable, deletable.

---

## Data Flow Example

Query: "Who wrote Hamlet?"

```
1. ENCODE: "Who wrote Hamlet?" → vector [0.23, -0.41, 0.87, ...]

2. CONVERGE:
   Hop 0: search → [Shakespeare(0.91), Marlowe(0.44), theater(0.38)]
   Hop 1: blend → Shakespeare-heavy vector, anchor back to query
   Hop 2: search → [Shakespeare(0.93), Hamlet(0.89), playwright(0.71)]
   Hop 3: stable (cosine > 0.99) → CONVERGED
   
   Converged concepts: [Shakespeare, Hamlet, playwright, English]

3. GENERATE:
   Template search → "[PERSON] was a [NATIONALITY] [OCCUPATION] who wrote [WORK]"
   Fill slots → "Shakespeare was an English playwright who wrote Hamlet"

4. TRACE:
   "Found Shakespeare because query matched 'wrote' + 'Hamlet' (0.91).
    Connected to 'playwright' (0.71) and 'English' (0.68).
    Template: person-occupation-work (confidence 0.85)."
```

---

## Storage & Scale

| Scale | Neurons | Storage | Search Time |
|-------|---------|---------|-------------|
| MVP | 400K (GloVe vocab) | ~600MB | <1ms |
| Production | 1M | ~1.5GB | <1ms |
| Year 1 | 10M | ~16GB | <2ms |

- FAISS IVF index: O(log N) search
- Low-confidence neurons pruned periodically
- Co-occurring neurons merged (self-compression)

---

## Transformer Equivalence Map

| Transformer | Our Substrate | Why Ours |
|-------------|---------------|----------|
| Attention | Spatial query (cosine sim) | Each hop inspectable |
| Weights | Confidence scores | Readable, editable |
| Feed-forward | Rule/function neurons | Source code, not matrices |
| Softmax | Rank by similarity | Same math |
| Layers | Convergence hops | Traceable chain |
| Training | Insert + feedback | No gradient descent |
| Residual connection | Query anchor | Prevents drift |

Same math. Different substrate. The substrate gives us: inspectability, editability, honesty.

---

## MVP Scope

**Goal:** Test the core hypothesis — can convergence + successor lists + templates produce coherent answers?

### MVP Components (build in order)

```
new-gen-ai/
├── HLD.md              # this document
├── CLAUDE.md            # full design spec (17 rounds)
├── src/
│   ├── neuron.py        # Neuron dataclass + NeuronDB (FAISS + SQLite)
│   ├── encoder.py       # text → vector (load pretrained embeddings)
│   ├── convergence.py   # convergence loop with query anchor
│   ├── generator.py     # template matching + successor walk + concept fallback
│   ├── feedback.py      # confidence updates (layer 1: self-consistency)
│   └── engine.py        # wires everything together, CLI interface
├── data/
│   └── (embeddings downloaded here)
└── tests/
    ├── test_convergence.py   # does it converge on known facts?
    ├── test_generation.py    # does it produce readable text?
    └── test_honesty.py       # does it abstain on unknown?
```

### MVP Tests (pass/fail, binary)

| Test | Pass Condition |
|------|---------------|
| Convergence on known fact | "Who wrote Hamlet?" → concepts include Shakespeare |
| Honest abstention | "What is glorpnax?" → "I don't know" |
| Successor walk | 5-word sequence that is grammatically valid |
| Template fill | Concept set → readable sentence |
| Delete = gone | Delete Shakespeare neuron → query returns "I don't know" |
| Confidence update | Useful answers → higher confidence on next query |

### What MVP Does NOT Include
- Multi-level neurons (character/word/concept routing)
- External verification (feedback layer 3)
- Creativity mode / temperature
- Error correction / backtracking
- Mini-KB / conversation context
- Function neurons / rule engine

These are post-MVP. Build them only after the core loop works.

---

## Dependencies

- Python 3.10+
- numpy
- faiss-cpu (already installed)
- sqlite3 (stdlib)
- Pretrained embeddings: GloVe 6B 300d or FastText (download at build time)

No GPU. No training. No external APIs.

---

## 10 Invariants (non-negotiable, checked before every commit)

1. No opaque training. Learning is fine if inspectable, editable, traceable.
2. Every answer has a source — fact, rule, or retrieval trace.
3. Delete = gone. No retraining.
4. Honest about failure. Non-convergence = "I don't know."
5. No GPU required. CPU-native. Sub-millisecond target.
6. Rules first, retrieval second, then miss.
7. Minimal examples → induced rules (8 examples enough).
8. Known limits stated, not hidden.
9. Verified failures published.
10. Reimplement transformer principles — same math, transparent substrate.

---

## Risks

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Successor walks produce broken grammar | HIGH (80%) | Template matching as primary strategy |
| Convergence loop drifts on open-ended queries | MEDIUM (40%) | Query anchor + hop limit + abort |
| 400K vocabulary too small for real use | LOW (20%) | FastText handles unseen words via subword |
| Templates too rigid for varied output | MEDIUM (50%) | Template composition via successor lists |
| System is just a sophisticated Markov chain | HONEST | Query anchor + confidence + mini-KB differentiate it |

---

## What Success Looks Like

The MVP succeeds if:
1. Convergence loop finds the right concepts for factual queries
2. The system says "I don't know" for things it doesn't know
3. Template-based generation produces at least one grammatically correct sentence
4. A fact can be deleted and the system immediately stops using it
5. All of the above runs on CPU in <100ms per query
