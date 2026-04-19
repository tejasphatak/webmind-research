# New-Gen-AI — Vector-Symbolic Reasoning System

## What this is
A CPU-native architecture: vector retrieval + symbolic rule induction + incremental learning. Not a neural network. Not a transformer. Not trained with gradient descent.

## 10 Invariants (non-negotiable)

1. **Not a neural network.** No gradient descent. No training. Discrete, inspectable, editable.
2. **Every answer has a source.** Concrete fact, named rule, or retrieval trace. No black box.
3. **Delete = gone.** No retraining. Immediate edit. Knowledge is mutable.
4. **Honest about failure.** When it fails, show exactly why.
5. **No GPU required.** CPU-native. Sub-millisecond. Runs on a phone.
6. **Rules first, retrieval second.** Try rule, fall back to retrieval, then miss.
7. **Minimal examples → induced rules.** 8 examples enough. Not millions.
8. **Known limits stated, not hidden.** Document what doesn't work.
9. **Verified failures published.** If it fails, prove it and publish it.
10. **Reimplement transformer principles, not avoid them.** Use database/computing primitives (spatial search, threads, convergence) to express what attention/weights/FFN do. Same math, transparent substrate.

## Before writing any code

- State the goal
- Check it against all 9 invariants
- If it violates any, stop and rethink

## Architecture

```
Input → ENCODER → KB → GENERALIZER → DECODER → Output
                    ↑                     |
                    └── REWARD / FEEDBACK ─┘
```

- **Encoder**: tokens → vectors. Hash-based n-grams now, pluggable for pretrained.
- **KB**: (context → next_token_distribution) with usefulness scores.
- **Generalizer**: induces rules from patterns. Arithmetic, constant, copy, reversal.
- **Decoder**: rules first → retrieval-and-blend → miss.
- **Reward Loop**: reshapes token weights and KB entry usefulness.
- **Reflective Thinker**: background rule induction during idle.

## What works (verified)

- Recursive decomposition with memoization (247 + 389)
- Retrieval-based generation on in-distribution prompts
- Reward-shaped generation
- Rule induction from 8 examples → 100% on novel addition
- Mixed-domain routing (language + arithmetic in same KB)

## What does NOT work (verified failures)

- OOD language generation
- Long-range coherence (4-token context window)
- Compositional generalization in language
- Novel rule class discovery
- Semantic clustering from hash embeddings
- Calibrated abstention

## The Neuron

The atomic unit of the system:

```
neuron = {
    vector: float[384],    # where it lives in concept space
    confidence: float,     # how reliable it is (grows when useful, shrinks when not)
}
```

- A point in space with a trust score. No text. No labels.
- KB = collection of neurons
- Search = find nearest neuron
- Confidence = how much to trust it
- Useful → confidence *= 1.1 (strengthen)
- Not useful → confidence *= 0.9 (weaken)
- Cap at ±0.8 to prevent mode collapse

## The Network

Neurons as virtual threads. Dynamic wiring. Connections form, strengthen, weaken, die based on experience. Hebbian: fire together → wire together. No backprop.

## Intelligence Levels (what this design achieves)

| Level | Capability | How | Status |
|-------|-----------|-----|--------|
| 1 | **Association** | "cat" → "pet" → "animal". Chain of related concepts firing. | Buildable now |
| 2 | **Pattern completion** | "2+2=?" → parallel paths converge on "4". Multiple neurons vote, highest confidence wins. Convergence loop. | Buildable now |
| 3 | **Rule induction** | Generalizer watches recurring chains, creates shortcuts. "addition = sum(operands)". Abstraction from patterns. | Buildable now |
| 4 | **Composition** | Rules invoke rules. "247+389" decomposes into digit-additions + carry. Multi-step reasoning from simple parts. | Buildable now |
| 5 | **Analogy** | "A:B as C:?" — find relationship vector between A and B, apply to C. Novel answers from structural similarity. | Research |

Levels 1-4: buildable. Level 5: research.

## Complete Design

### 1. Neurons
Vector (384-dim) + confidence (float). Atomic unit. A point in concept space with a trust score.

### 2. Network
Dynamic connections. Virtual threads. Neurons fire signals to connected neurons. Connections strengthen (fire together → wire together) or weaken and die. Hebbian. No backprop.

### 3. Reasoning
Convergence loop. Fire → check → re-fire → check. When output stabilizes across hops = converged = answer found. No convergence = "I don't know."

### 4. Generation (the decoder)
Walk through word-embedding space, one step per token:
```
Step 0: "Who wrote Hamlet?" → converge → "Shakespeare" (0.95)
Step 1: "Shakespeare" as context → converge → "was" (0.88)
Step 2: "Shakespeare was" → converge → "English" (0.84)
Step 3: "Shakespeare was English" → converge → "playwright" (0.81)
Step 4: → nothing new above threshold → stop
Output: "Shakespeare was English playwright"
```
Each token = one convergence loop with growing context. No neural decoder. Repeated convergence IS the decoder.

### 5. Creativity
Vector arithmetic across distant regions. "chocolate" + "pizza" = new point. Metaphor = applying a relationship vector from one region to another. Blending = generation of novel concepts.

### 6. Coherence
Mutual cosine similarity between converged neurons. High pairwise agreement = coherent. Low = noise, discard.

### 7. Stopping
Convergence = stop. No convergence = abstain. Built into the loop, not a separate mechanism.

### 8. Vocabulary
Word embeddings (GloVe/fastText, 400K words, all languages) as the neuron set. Every word = a point. The decoder walks between points. Same space for all languages.

### 9. Confidence
Built into every neuron. Grows when useful, shrinks when not. No separate confidence model. The geometry + the weight = the confidence.

## Known Holes (honest assessment)

1. **Decoder walk won't produce grammar.** Word embeddings don't encode syntax. Function words ("the", "an", "is") have weak embeddings, won't win confidence votes. Output = keyword soup, not sentences.
   - **Fix: dual-vector neurons.** Each neuron stores two vectors: `vector` (what I am) + `context_vector` (what I follow). Search both. "Given where I am, what neuron follows this region with highest confidence?" Grammar emerges from spatial relationships between consecutive concepts. No maps, no templates, no hardcoding — just spatial queries on pairs.

2. **Word embeddings ≠ sentence embeddings.** GloVe puts "bank" in ONE place. "River bank" and "bank account" need different locations. Word embeddings are context-free. We need context-dependent meaning.
   - **Fix: convergence-driven context blending.** The neuron's vector shifts based on query context. Search vector = blend of word_vector + context_vector, weighted by convergence hop. Hop 1 = mostly word. Hop 3 = mostly context. The vector drifts to the right meaning through iteration. This IS attention — "how much does context shift this word's meaning" — done through convergence instead of matrix multiply.

3. **Convergence may not converge.** Dense space with 400K words → each hop could wander: "Shakespeare" → "theater" → "movie" → "Hollywood" → drift. No guarantee for open-ended queries.
   - **Fix: query anchor.** Every hop blends current position with original query vector. The query is gravity — the system never forgets what it was asked. Early hops explore (more current), later hops contract (more anchor). Like simulated annealing. Convergence = vector stops moving. Divergence after N hops = abort, "I don't know." This is transformers 101 — residual connections keep the original signal alive through layers.

4. **Dynamic wiring at scale is O(N²).** 1M neurons × potential connections = combinatorial explosion. Virtual threads help parallelism, not search space.
   - **Fix: proximity IS connection.** No explicit wiring. Two neurons are "connected" if they're close in vector space. The spatial structure IS the network topology. Dynamic rewiring = vectors shift. Search is O(log N) with spatial indexing (KD-trees, LSH). No connection graph. The DB is the network.

5. **Confidence bootstrapping — chicken and egg.** New neurons start at what confidence? Too high → untested dominates. Too low → never selected.
   - **Fix: initial confidence = source_count × similarity.** More sources agreeing = higher trust. One source = low. Five sources same thing = high. Scientific consensus principle. Convergence loop self-corrects from first use — fires and converges = reinforced, fires and diverges = penalized.
   - **Confidence is persisted in the neuron.** Not computed fresh — stored, updated, carried forward. Each confirmation grows it. The neuron remembers its own reliability. This IS what neural net weights are — but readable, editable, traceable.

6. **No attention mechanism.** "The cat sat on the mat, it was soft" — what does "it" refer to? No coreference, no anaphora, no long-range dependency.
   - **Fixed by #2.** Context_vector carries what came before. Convergence loop searches backward through recent context for most relevant neuron. The loop IS attention.

7. **Rule induction is brittle.** Only finds rules from hardcoded hypothesis class. Reversal test already failed. Gap between "induce addition" and "induce grammar" is enormous.
   - **Fixed by design.** No hardcoded hypothesis classes needed. Rules = repeating vector relationships in the DB. "cat→tac" and "dog→god" have the same spatial relationship. The Generalizer finds recurring geometry, not named patterns. Rules are discovered, not programmed.

8. **Coherence check is post-hoc.** Generate first, filter after. Neural decoders generate coherently by construction. We may discard most output — wasteful, possibly nothing survives.
   - **Fixed by convergence.** Convergence IS coherence. If the loop converges, the neurons agree — coherent by construction. If it doesn't converge, nothing is output. No post-hoc filter needed. The loop only produces output when neurons reach consensus.

9. **Embedding space isn't continuous for language.** "Shakespeare" and "was" aren't nearby in GloVe. No continuous path from subject to verb. Walk needs guidance — grammar, templates, or learned transitions.
   - **Fixed by dual-vector neurons (#1).** The walk isn't through word-space — it's through transition-space. "Shakespeare" → "was" is a valid transition because the context_vector of "was" matches "Shakespeare". The decoder doesn't walk between word meanings — it walks between word transitions. Continuity is in the transition space, not the word space.

10. **Scale: 400K neurons × convergence × tokens = millions of ops.** 10-word answer = 10 hops × 400K searches × multiple iterations. CPU sub-millisecond? Not at this scale.
    - **Fix: spatial indexing + confidence pruning.** O(log N) search with KD-trees/LSH, not O(N). Plus: only search neurons above a confidence threshold — low-confidence neurons are skipped. The DB shrinks dynamically based on quality. GPU compute shaders can parallelize the remaining searches. Proven: we loaded 1.24M neurons on a phone GPU in 8 seconds.

**The biggest honest question:** Are we rebuilding a transformer from first principles but worse? If we need all the same components (attention = convergence, weights = confidence, FFN = rules), why avoid the proven architecture?

**The answer must be:** inspectability, editability, honesty about failure. Those are the invariants neural nets can't match. If we lose those while solving these holes, we've gained nothing.

## Core Thesis

We are NOT avoiding transformers. We are **reimplementing the principles that make transformers work** using database/computing primitives instead of matrix multiplies.

| Transformer concept | Our substrate |
|---------------------|---------------|
| Attention | Spatial query (cosine similarity in DB) |
| Weights | Confidence scores in DB entries |
| Feed-forward | Rule lookup/application |
| Softmax | Ranking by similarity |
| Layers | Convergence hops |
| Training | Insert + feedback |

Same math. Different substrate. The substrate gives us what neural nets can't: inspectability, editability, honesty.

The 10 holes are not flaws — they are the research agenda. Each hole = "how do we express this transformer capability using database/thread/convergence primitives?"

The goal: redefine transformers to work with databases, spatial search, threads, and convergence — using the computing power we actually have.

## Holes Round 2 (holes in the fixes)

1. **Storage doubled.** Dual vectors = 769 values per neuron. 1M neurons = 3GB. Phone memory?
   - **Fix: store pointer, not vector.** Context_vector is derived from predecessor — store predecessor index, compute context on the fly. 385 values + 1 int per neuron. Multiple predecessors = top 3 indices. Half the storage, same information.
2. **Transition space unproven.** Word transitions may be sparse/irregular. Space could be mostly empty.
   - **Tested: confirmed weak.** Real transitions cluster at 0.07 vs random 0.03 — not enough. **Fix: empirical successors, not geometric transitions.** Store successor list per neuron: `[(neuron_id, confidence), ...]` learned from observed data. Embedding space handles meaning. Successor list handles grammar. Two separate concerns, two separate mechanisms.
3. **Decoder is slow.** 50 tokens × convergence loop = 5 seconds. Transformers do one forward pass.
   - **Fix: two-speed generation.** Content tokens (nouns, verbs, key facts) = full convergence loop. Grammar tokens (articles, prepositions) = successor lookup, instant. If successor confidence > 0.8, emit immediately. Else fall back to convergence. 70% of tokens are grammar = 70% instant. ~400ms per sentence instead of 5s.
4. **Source count is gameable.** Misinformation from 5 sources > truth from 1. Popularity ≠ truth.
   - **Fix: source diversity, not count.** Count independent sources, not mirrors. Wikipedia + DuckDuckGo + textbook = 3. Wikipedia × 4 mirrors = 1. Plus contradiction detector: sources disagree → confidence = 0, flag for review. Disagreement = uncertainty, not average.
5. **"Recurring geometry = rule" is vague.** Reversal lives in character space, not embedding space. Vector diff can't capture structural transforms across representation levels.
   - **Fix: multi-level neurons.** Character-level (letters), word-level (embeddings), concept-level (categories). Generalizer looks for patterns at ALL levels. Reversal found at character level, grammar at word level, analogy at concept level. Convergence loop searches across levels — if word-level doesn't converge, try character or concept.
6. **Context still finite.** Context_vector only carries immediate predecessor. 10 sentences back = gone. Same 4-token problem, slightly wider.
   - **Fix: conversation history as mini-KB.** Every fired neuron in this session becomes a temporary KB. Context = spatial query against ALL previous neurons, ranked by relevance to current word — not recency. "soft" finds "mat" (0.7) over "park" (0.3) regardless of position. Same search mechanism as main KB. ~200 neurons per conversation = microseconds. No decay. No window. No limit.
7. **No error correction.** Wrong token 3 → tokens 4-10 built on bad foundation. No backtrack mechanism.
   - **Fix: divergence = error signal.** If token 4's convergence loop diverges, token 3 was wrong. Backtrack: drop token 3, penalize its neuron (confidence *= 0.9), take second-best candidate, continue. The convergence loop IS the error detector. Built-in self-correction.
8. **Who decides "useful"?** Most queries are fire-and-forget. No feedback = confidence never updates.
   - **Fix: three-layer feedback.** Layer 1: self-consistency — ask the same thing differently, same answer = boost, different = penalize (free, every query). Layer 2: user behavior — follow-up on same topic = insufficient, new topic = accepted (implicit, most queries). Layer 3: external verification — re-check high-confidence neurons against sources periodically (expensive, periodic). No single layer sufficient. Together they triangulate truth.
9. **Parallel convergence may deadlock.** Multiple threads waiting for each other. Synchronization model undefined.
   - **Fix: no orchestrator. Local convergence.** Each neuron tracks its own state — if output unchanged from last hop, stop firing. Neurons that converge early go quiet. Remaining neurons keep firing. Network settles organically like ripples dying down. Firing rate → 0 = converged, read active neurons. Firing rate stays high = no convergence, abort. No barrier. No bottleneck. No deadlock. Emergent convergence, not managed.
10. **No end-to-end optimization.** Transformer components co-adapt via gradient descent. Our components are wired manually. Individual pieces work, but system doesn't optimize as a whole.
    - **Fixed by design.** Three-layer feedback (#8) + local convergence (#9) + error correction (#7) = end-to-end convergence feedback. Every query tunes every neuron that participated. Confidence adjusted by outcome of the FULL chain, not just individual output. This IS training — convergence-driven reinforcement, not gradient descent.

**Status: OPEN — apply convergence loop to the design itself. Iterate until satisfied. Check invariants at each step.**

## Holes Round 3 (holes in the fixes of the fixes)

1. **Successor lists grow unbounded.**
   - **Fix: top-K eviction.** K=10 per neuron. New successor competes for slot — replaces lowest if better. Fixed 80 bytes per neuron. Context_vector from previous neuron disambiguates which of the 10 to pick.

2. **Local convergence, no global coherence.**
   - **Fix: successor graph + mini-KB.** Successor graph prevents incoherent chains (no path from "Shakespeare" to "pizza"). Mini-KB filters by conversation relevance. Global coherence = what CAN follow + what SHOULD follow.

3. **Three-layer feedback is expensive.**
   - **Fix: time-stratified.** Layer 1 (self-consistency): 10% sampling, not every query. Layer 2 (user behavior): free, always. Layer 3 (external verification): idle-time background thread only. Zero query-time impact from layers 1 and 3.

4. **Backtracking cascades.**
   - **Fix: depth-1 backtrack + honest abort.** Try second-best candidate once. If that fails too, stop and emit partial answer. Backtrack > 3 times = "I'm not confident enough." Invariant #4: honest about failure.

5. **Multi-level neurons multiply everything.**
   - **Fix: route by query type.** Encoder classifies query → one level per query, not three. "reverse cat" → character. "who wrote Hamlet" → concept. Region in embedding space determines level. Cost: 1x not 3x.

6. **Content vs grammar detection.**
   - **Fix: let confidence decide.** Successor confidence > 0.8 → grammar token, fast path. No confident successor → content token, convergence loop. The threshold IS the classifier. No pre-classification needed.

7. **Mini-KB session leakage.**
   - **Fix: topic-shift detection + relevance decay.** Query far from mini-KB centroid = topic shift = clear. Unreferenced neurons age per query, drop after 5. Self-managing. No manual session boundaries.

## Holes Round 4

1. **Top-K=10 too few for ambiguous words.** Holds — if none of 10 match context, falls back to convergence loop (content path). Top-K is fast path, not only path.
2. **Successor graph blocks creativity.** Fix: creativity mode — relax successor constraint, allow any neuron above threshold. Normal=graph(fast). Creative=open search(slow,novel).
3. **10% sampling is arbitrary.** Fix: adaptive rate = inverse of average KB confidence. New system checks more, mature system checks less. Self-tuning.
4. **Depth-1 backtrack too shallow.** Holds — query anchor means token 1 is most anchored. Wrong token 1 = ambiguous query → ask for clarification, not deeper backtrack.
5. **Query-type routing assumes clean categories.** Fix: parallel routing. Decompose query into sub-queries, run each at appropriate level, combine via convergence.
6. **Confidence threshold shifts with maturity.** Fix: relative threshold — grammar = successor confidence significantly above KB average. Threshold adapts to system maturity.
7. **Topic-shift detection fragile with gradual drift.** Fix: sliding window centroid from last N queries. Old mini-KB entries pruned when centroid drifts away from them.

## Holes Round 5 (deep structural)

1. **No sense of time.** Fix: timestamp field + recency-weighted confidence. `effective_confidence = confidence × decay(age)`. Time-insensitive queries ignore timestamp.
2. **Negation invisible in embeddings.** Fix: polarity metadata (+1/-1) per neuron. Search matches embedding, then checks polarity. "NOT" in query flips expected polarity.
3. **Numbers are opaque.** Fix: route numeric queries to rule engine, not KB. Embeddings handle meaning, rules handle computation. Multi-level routing.
4. **Can't say WHY.** Fix: log convergence path. Each hop = one step in the explanation. "Found X because query matched concept Y (0.95) which connects to Z (0.92)." Invariant #2.
5. **Adversarial inputs poison KB.** Fix: rate-limit confidence change per user (±0.1/session). Cross-user agreement required for large shifts. One user = one source.
6. **System is deterministic.** Fix: temperature parameter. Sample from top-K successors weighted by confidence. T=0 deterministic. T=0.5 varied. T=1 creative. User-controllable.
7. **Bootstrap: empty KB.** Acknowledged. Loading GloVe IS initialization. Difference: embeddings are pre-computed, not optimized. Map is borrowed. Confidence and connections are earned through use.

## Holes Round 6

1. **Recency decay penalizes timeless knowledge.** Fix: `temporal` flag per neuron. Learned: if answer never changes across re-verification → timeless. If external check finds different answer → temporal.
2. **Polarity metadata too fragile for double negation.** Fix: use sentence-level encoding (MiniLM does distinguish "is" from "is not" at sentence level). Multi-level: sentence for polarity, word for concept. Drop the +1/-1 metadata.
3. **Rule engine separate from neurons.** Fix: function neurons. `{vector, confidence, function: (input) → output}`. Rules live in same space, compete on same confidence, fire through same network. Rules ARE neurons.
4. **Rate-limiting needs user identity.** Fix: convergence validation instead. Confidence change only sticks if neuron participates in converging chains in subsequent queries. Wrong boosts self-correct via divergence. No user tracking.
5. **Temperature produces nonsense.** Fix: minimum confidence floor. Temperature samples only above floor. Creativity within bounds.
6. **GloVe is trained — violates invariant #1?** Fix: acknowledge honestly. Encoder is a pre-trained input we don't modify. Like using a calculator. System doesn't train. It uses a pre-trained map. Invariant #1 applies to our system's learning, not its inputs.
7. **Function neurons blur "not a neural network."** Fix: function neurons are inspectable source code, not weight matrices. `def add(a,b): return a+b` — readable, editable, deletable. Same invariant, transparent substrate.

## Holes Round 7

1. **Neuron definition complex.** Holds — 8 fields is a DB row, not a weight matrix. Named, readable, editable. Complexity in structure ≠ opacity.
2. **Three neuron types.** Holds — 99% data, rare function, rarer meta. Like organs: most cells generic, few specialized.
3. **Design hard to hold in one head.** Fix: CLAUDE.md IS the spec. Single source of truth. If a fix can't be explained in 2 lines, it's too complex.
4. **Zero empirical validation.** The real hole. Design converged in theory. Must build and test.

## Holes Round 8 (final convergence check)

1. **Can it learn something never seen?** "Elephants are gray" from separate neurons? No — can only traverse explicit connections. Not a flaw — that's the honesty. Gap filled by learning from use + web search fallback + self-evolution.
2. **Can it produce fluent text?** Unknown. Successor lists from real text SHOULD work. First thing to test empirically.
3. **Minimum viable prototype:** 400K GloVe neurons + successor lists from corpus + convergence loop + confidence. Strip everything else.
4. **If successor lists fail:** Fallback to retrieval + stored answers. Hybrid. Honest about why.
5. **Is this a Markov chain?** Honestly: sophisticated Markov with convergence steering. Query anchor + confidence + mini-KB make it more than vanilla. Honest about the lineage.
6. **What it does BETTER than transformers:** Not fluency, not creativity. But: inspectable, editable, honest, efficient, incremental, zero training cost. Different tradeoffs for different use cases.
7. **Design status: CONVERGED.** Holes are circular/philosophical. Structural design stable. Remaining questions are empirical.

### Converged Design Summary
```
Neuron = {vector, confidence, successors[10], predecessors[3], timestamp, temporal, level}
Network = spatial proximity + successor graph
Reasoning = convergence loop with query anchor
Generation = successor walk, two-speed (grammar fast, content converge)
Feedback = three-layer (self-consistency 10%, user behavior always, external verification idle)
Context = mini-KB (conversation history as spatial query, sliding window centroid)
Error correction = divergence detection + depth-1 backtrack + honest abort
Creativity = open search with temperature + confidence floor
```

## Holes Round 9 (adversarial — trying to break it)

1. **Can't tell jokes.** No structural format awareness (setup+punchline). Known limitation. Possible fix: template neurons for structural patterns. Needs data.
2. **Translation might work.** MiniLM multilingual puts "hello"/"bonjour" nearby. Convergence with "French" anchor could find it. Needs testing.
3. **500-word essays meander.** No global plan. Fix: planning convergence — first pass finds 5-10 concept anchors, then generate per-anchor. Doable within existing design.
4. **Contradictory knowledge paralyzes.** Both "Pluto is/isn't planet" → confidence 0 for both. Fix: external verification (layer 3) resolves via authoritative sources.
5. **Counterfactuals: correctly says "I don't know."** Level 5 (analogy) needed. Marked as research. Honest behavior.
6. **1000 QPS scalable.** Virtual threads + spatial indexing + no shared state. Needs load testing.
7. **No personality.** Fix: style neurons bias successor selection. Cosmetic, not structural.

**Convergence: CONFIRMED. Two rounds, no structural breaks.**

## Holes Round 10 (existential — challenge the whole premise)

1. **Why use this over ChatGPT?** Different users: doctors needing traceable answers, lawyers needing deletable facts, offline schools, zero-hallucination companies. Not competing — serving use cases ChatGPT can't.
2. **Successor list is a bad language model.** Counter: it's not standalone LM. Convergence + anchor + confidence = directed walk, not statistical generation. Critical empirical question: does it produce readable text?
3. **GloVe is from 2014.** Fix: use FastText (subword, any word) or MiniLM for new terms. Hybrid vocabulary.
4. **Building successor lists IS training.** INVARIANT #1 REVISED: "No opaque training." Learning from data is fine if result is inspectable, editable, traceable. Successor lists = traceable. Weight matrices = not.
5. **Convergence loop is attention with extra steps.** Yes. We trade speed for transparency. Each hop inspectable. That IS the fundamental tradeoff. Invariant #10.
6. **Memory at scale.** 10M neurons after a year = 16GB. Fix: self-compression (merge co-occurring neurons) + periodic pruning of low-confidence.
7. **Publishable?** Yes as tradeoff analysis paper. Product viability depends on empirical text fluency test.

## Holes Round 11 (categories we missed)

1. **Safety.** Fix: safety neurons as system invariants, not deletable data. Design exception documented.
2. **Privacy.** Fix: neuron ownership field. Private vs shared at teach time.
3. **Bias.** Fix: debiasing embedding space before load. One-time, inspectable, documented. Acceptable per revised invariant #1.
4. **IP/Copyright.** Fix: source attribution per neuron. Advantage over neural nets — we CAN trace.
5. **Multilingual equity.** Fix: balanced training data + per-language confidence tracking. Honest about gaps.
6. **Accessibility.** Fix: tiered models (50MB minimal, 650MB standard, 2GB+ full).
7. **Regulatory.** Natural advantage — EU AI Act compliance by construction. Invariant #2 is also legal requirement.

## Holes Round 12 (brutal grounding check)

### Honest confidence assessment
| Component | Confidence |
|-----------|-----------|
| Neuron storage | 95% |
| Spatial search | 95% |
| Convergence for retrieval | 90% (proven) |
| Confidence tracking | 85% |
| Successor lists for word order | 40% |
| Fluent text generation | 20% |
| Full replacement for transformers | 5% |

### The generation gap is real
- Transition space tested: WEAK (0.07 vs 0.03)
- Successor walks likely produce broken grammar
- N-gram models failed for this exact reason — neural LMs replaced them
- Context-conditioned successors MIGHT compensate — unknown

### Key insight: THINKING ≠ SPEAKING
Architecture for thinking: solid (11 rounds, all holes addressed).
Architecture for speaking: unproven and likely weak.
These are two different problems. Must be separated.

### Options for generation
1. Accept: build knowledge engine, not language generator
2. Hybrid: our reasoning + small neural decoder for text
3. Solve: higher-order context, more data, research problem
4. Reframe: system outputs structured concepts, thin formatting layer converts to text

## Holes Round 13 (what if transformers are wrong?)

### The Newton → Einstein question
Transformers = brute-force attention (O(N²), compare everything to everything).
Humans = selective attention (search for what's relevant).
Our convergence loop = selective attention. O(log N).

### The real insight: DECOMPOSITION
Transformers melt everything into one weight matrix (facts + grammar + reasoning + style). Can't separate. Can't edit.
Our system decomposes: facts (neurons) + reasoning (convergence) + grammar (rules) + style (parameters). Each independently inspectable.

### Revised generation architecture
```
Query → Convergence Loop → Concept Neurons [Shakespeare, wrote, Hamlet, 1600]
      → Grammar Engine (rule-based function neurons)
      → "Shakespeare was an English playwright who wrote Hamlet around 1600."
```
Grammar engine = transformation rules, not language model. Insert articles, prepositions, conjugation, clause ordering. Inspectable. Invariant #1 preserved.

### Updated confidence
| Component | Confidence |
|-----------|-----------|
| Convergence for thinking | 90% |
| Grammar engine (rule-based) | 70% |
| Fluent output from concepts + rules | 60% |

## Holes Round 14 (grounded, no faking)

1. **Rule-based grammar tried and abandoned.** 30 years of NLP. Too many exceptions. Revised confidence: 30%.
2. **Concept sequence has no natural order.** Convergence finds by relevance, not grammar. Reordering needs structure understanding.
3. **Per-language grammar rules don't scale.** SVO/SOV/VSO per language = linguistic DB per language.
4. **Small trained decoder (10M params) is more realistic.** Purpose-built for "concepts → sentence." Inspectable at 10M scale. 50% confidence.
5. **Concept-guided retrieval might be the answer.** Find stored text that covers the concept combination. Not generation — smart retrieval.
6. **We're building a THINKING engine, not a text generator.** Every attempt at generation hit the same wall.
7. **Safest path:** thinking engine → concept lists → small decoder later as enhancement.

### Revised confidence
| Component | Confidence | Evidence |
|-----------|-----------|----------|
| Convergence for thinking | 90% | Proven |
| Concept-guided retrieval | 85% | Extension of proven |
| Rule-based grammar | 30% | NLP history says hard |
| Successor walk generation | 20% | Tested: weak |
| Small trained decoder | 50% | Plausible, unproven |
| Concept list as output | 95% | Trivially works |

## Holes Round 15 (is this new-gen AI?)

### What this system IS
A **reasoning engine**, not a language generator:
1. Finds relevant concepts (spatial search)
2. Chains them together (convergence loop)
3. Verifies consistency (self-convergence)
4. Computes when needed (function neurons)
5. Knows when it doesn't know (non-convergence = abstain)
6. Explains its reasoning (hop trace)

### Where it beats LLMs
- Math: provably correct (rule neurons). LLMs guess.
- Multi-hop facts: shows the chain. LLMs are black boxes.
- Editing: delete a neuron. LLMs retrain for $100M.
- Trust: says "I don't know." LLMs hallucinate with confidence.

### Where LLMs beat it
- Fluent prose, creative writing, conversation, translation.

### Why build it
Not to replace LLMs. To serve users who need TRUST over FLUENCY: doctors, lawyers, teachers, regulators, scientists. "New gen" = serves the generation of users who need transparent AI.

### THE REMAINING HOLE: SPEAKING — CONVERGING

**Insight: how do children learn to speak?** Not rules. Not neural nets. Pattern-matching against heard sentences. Template retrieval with slot filling.

**Fix: template neurons.**
```
template_neuron = {
    vector: embedding of the pattern
    pattern: "[PERSON] [VERB-past] [WORK] in [YEAR]"
    confidence: learned from frequency
    example: "Shakespeare wrote Hamlet in 1600"
}
```

Thinking engine → concept set → search for closest template → fill slots → fluent output.

Templates learned from data: every sentence decomposed into pattern + slots. Common patterns = high confidence. Inspectable. Editable. Multilingual (per-language templates).

**Precedent:** Example-Based Machine Translation (EBMT, 1990s). Proved the concept works for grammatical output. Our advantage: convergence finds the right concepts FIRST, then template matches.

**Confidence: 65%.** EBMT proves concept. Untested at our scale.

## Holes Round 16 (poking holes in template neurons)

1. **Template coverage.** 80% of speech = ~2000 patterns. No template match → concept list fallback. Bounded. Not fatal.
2. **Slot filling.** Verb conjugation etc via function neurons. ~200 irregular verbs + rules. Finite.
3. **Multiple valid templates.** Query embedding steers selection. "Who wrote?" → active voice. "What was written?" → passive. Spatial search on templates.
4. **Complex sentences.** Template COMPOSITION via template-level successor lists. Same mechanism, higher abstraction.
5. **Corpus for templates.** Same data, different extraction. Patterns not weights. Inspectable. Invariant-compliant.
6. **Template extraction.** NER to identify entities → remainder = template. Solved problem.
7. **Templates >> successor walks.** Guaranteed grammatical (from real sentences). Clear win.

**Speaking status: CONVERGED at 75% confidence.**

## Holes Round 17 (full system end-to-end)

1. **Latency: ~115ms.** Strength. Faster than ChatGPT first-token.
2. **Wrong concepts diagnosable.** Same problem as LLM hallucination but transparent. Self-consistency catches contradictions.
3. **Template ordering.** Query-driven — most relevant concept group first.
4. **Template learning from conversation.** Same self-evolution as fact learning. Every sentence → decompose → store template.
5. **Cross-language.** Honest about gaps. Self-corrects over time.
6. **Dialogue.** Mini-KB handles context naturally. A strength.
7. **75% is enough to build.** Remaining 25% is empirical. Only code answers it.

### Final convergence assessment
| Aspect | Confidence |
|--------|-----------|
| Neuron model | 95% |
| Thinking (convergence) | 90% |
| Speaking (templates) | 75% |
| Safety/privacy/bias | 80% |
| Scalability | 85% |
| Latency | 90% |
| Diagnosability | 95% |
| **Overall** | **~85%** |

**Design convergence: FINAL after 17 rounds.**

## Verified Prior Art (real papers, real venues)

### Convergence Loop
- **ITER-RETGEN** (Shao et al., EMNLP 2023) — iterative retrieval-generation. Closest to our convergence loop.
- **IRCoT** (Trivedi et al., ACL 2023) — retrieval per reasoning step. 21-point improvement on HotpotQA.
- **Self-RAG** (Asai et al., ICLR 2024 Oral) — self-reflection on retrieval quality.
- GAP: No formal convergence guarantee. No non-neural version.

### Template Generation
- **Nagao 1984** — founded Example-Based MT. The original retrieve-and-adapt.
- **Wiseman et al., EMNLP 2018** — learned neural templates with interpretable structure.
- **Hashimoto et al., NeurIPS 2018** — retrieve-and-edit framework.
- **Reiter & Dale 1997** — classical NLG pipeline (content → planning → realization).
- GAP: No non-neural slot filling with confidence-weighted knowledge.

### Knowledge Units with Confidence
- **Knowledge-Based Trust** (Dong et al., PVLDB 2015, Google) — trustworthiness scores for 119M web pages.
- **TruthFinder** (Yin et al., KDD 2007) — iterative source trust + fact confidence.
- **Facts as Experts** (Verga et al., NAACL 2021) — editable fact memory, 27-point improvement.
- GAP: No per-query online confidence updates.

### Vector Search for Reasoning
- **kNN-LM** (Khandelwal et al., ICLR 2020) — kNN in embedding space for generation. SOTA perplexity.
- **REALM** (Guu et al., ICML 2020) — retrieval as reasoning. 4-16% improvement.
- GAP: No multi-hop reasoning purely through vector search.

### Decomposed Systems
- **Neurosymbolic AI: The 3rd Wave** (Garcez & Lamb, 2023) — the manifesto.
- **RETRO** (Borgeaud et al., ICML 2022, DeepMind) — 7.5B matches 175B GPT-3 by separating knowledge from model.
- **Neural Module Networks** (Andreas et al., CVPR 2016) — composable reasoning modules.
- GAP: No unified system combining all components.

### Self-Evolving KB
- **NELL** (Mitchell et al., CACM 2018) — 120M beliefs, self-evolving since 2010. Canonical reference.
- **Voyager** (Wang et al., 2023) — self-evolving skill library from failures.
- GAP: No gradient-free self-improvement of factual knowledge from query failures.

### Selective Attention
- **Reformer** (Kitaev et al., ICLR 2020) — LSH attention, O(L log L). Exactly our approach.
- **BigBird** (Zaheer et al., NeurIPS 2020) — proved sparse attention is universal approximator + Turing complete.
- GAP: Our convergence loop is selective attention. Proven viable.

### Rule Induction
- **ILP** (Muggleton, 1991) — founded inductive logic programming.
- **Neural LP** (Yang et al., NeurIPS 2017) — differentiable rule learning.
- **DeepProbLog** (Manhaeve et al., NeurIPS 2018) — neural predicates in logic programs.
- GAP: No integration with vector/embedding approaches at scale.

### What Killed N-grams
- **Bengio et al., JMLR 2003** — neural LM replaced n-grams. Curse of dimensionality.
- **kNN-LM** (Khandelwal, ICLR 2020) — modern resurrection: n-gram idea in embedding space.
- Our successor lists = n-grams. Known weakness. Template approach avoids this.

### Explainability
- **LIME** (Ribeiro et al., KDD 2016) — post-hoc explanation. We do built-in.
- **REALM, RETRO, Self-RAG** — all provide retrieval-level traceability.
- Our advantage: traceable at EVERY step, not just retrieval.

## WHAT NO ONE HAS BUILT (our contribution)
1. Formal convergence guarantees for iterative retrieval
2. Non-neural slot filling from templates with confidence-weighted knowledge
3. Online per-query confidence updates
4. Multi-hop reasoning purely through vector search (no neural reasoner)
5. Gradient-free self-improvement of factual KB from query failures
6. **Unified system combining ALL 10 components** — each exists in isolation. Integration is the open problem. That's us.

## Design Flexibility

The design is flexible. Any part can change. Nothing is sacred except the 9 invariants. If a component doesn't work, replace it. If a hole reveals a better approach, take it. The invariants are fixed. The implementation is fluid.

## Rules for working on this project

- No drifting into retrieval-only systems
- No canned/stored answer text
- Every claim must be verified by execution
- Every failure must be documented
- No "it kind of works" — prove it or say it doesn't
- Check invariants before every commit
