# DMRSM Model Capability Specification

## Context
DMRSM (Data-Mined Reasoning State Machine) is implemented and tested (56/56). The engine orchestrates reasoning but depends on a language model for every step. The model is NOT an LLM that stores knowledge — it is a **language processing unit** backed by a database (search engine). All facts come from search. The model only UNDERSTANDS language.

This document defines: what capabilities the model must have, how to evaluate it, and what it does NOT need.

## The Fundamental Insight

**Traditional LLM**: stores knowledge in parameters → generates answers from memory → hallucinates when memory is wrong.

**DMRSM model**: stores ZERO knowledge → all facts from database/search → model only processes language (classify, extract, judge, compose).

This means the model profile is radically different from GPT-4 or Claude. We need a **language understanding specialist**, not a knowledge repository.

## 7 Capabilities the Model MUST Have (derived from 2,034 action calls across 531 traces)

The model is called ~4 times per question on average. Each call requires a specific capability. Here they are ranked by frequency and criticality.

### Capability 1: TOPIC EXTRACTION (819 calls — 40%)
**What**: Given a question, extract the optimal search query.
**Input**: "Who was the first woman to win a Nobel Prize?" (natural language)
**Output**: "first woman Nobel Prize" (search-optimized keywords, ~5-10 words)
**Why critical**: Bad topic extraction → bad search → everything downstream fails. This is the #1 bottleneck.
**What it requires**: Understanding question structure, identifying the knowledge gap, stripping filler words.
**What it does NOT require**: Knowing the answer. The model doesn't need to know it's Marie Curie.
**Evaluation metric**: Search Recall@3 — does the correct Wikipedia article appear in top 3 results when searching the extracted topic? Target: >80%.

### Capability 2: RELEVANCE CLASSIFICATION (310 calls — 15%)
**What**: Given a question + a passage, is the passage relevant?
**Input**: Q: "Capital of France?" + Context: "Paris is the capital and most populous city of France..."
**Output**: "YES" with confidence 0.95 | "NO" with confidence 0.88
**Why critical**: False positives → extract wrong answer. False negatives → miss correct article.
**What it requires**: Semantic matching between question and passage. Understanding what "relevant" means in context.
**What it does NOT require**: Knowing the answer. Only whether the passage CONTAINS it.
**Evaluation metric**: F1 score on relevance classification. Target: >85%. Calibration: ECE < 0.15 (confidence matches actual accuracy).

### Capability 3: ANSWER EXTRACTION (233 calls — 11%)
**What**: Given a question + a relevant passage, extract the specific answer.
**Input**: Q: "When did WWII end?" + Context: "...global conflict that lasted from 1939 to 1945..."
**Output**: "1945" (short, precise, 1-20 tokens)
**Why critical**: This IS the answer. Everything else is infrastructure to reach this moment.
**What it requires**: Reading comprehension. Find the span that answers the question. Not generate — FIND.
**What it does NOT require**: Any knowledge beyond what's in the passage. This is extractive QA, not generative.
**Evaluation metric**: Exact Match (EM) and Token F1 on SQuAD-style benchmarks. Target: >65% EM.

### Capability 4: SYNTHESIS / COMPOSITION (270 calls — 13%)
**What**: Given a question + multiple facts/partial answers, compose a coherent combined answer.
**Input**: Q: "Who painted the ceiling of the building where the Pope lives?" + Facts: ["Pope lives in Vatican City", "Sistine Chapel is in Vatican", "Michelangelo painted the ceiling"]
**Output**: "Michelangelo painted the ceiling of the Sistine Chapel in Vatican City"
**Why critical**: Multi-hop and deep-thought patterns always end with synthesis. Without this, the engine can only do single-fact lookups.
**What it requires**: Combining 2-5 pieces of information into a coherent sentence. Grounded in provided facts only.
**What it does NOT require**: Generating new information. The answer must be ENTIRELY traceable to the input facts.
**Evaluation metric**: Faithfulness score — does the synthesis contain ONLY information from the provided facts? Target: >90% faithful (no hallucinated additions). Also: human-judged coherence.

### Capability 5: QUALITY JUDGMENT (310 calls — 15%)
**What**: Given a question + a candidate answer, is the answer good enough?
**Input**: Q: "What is the largest planet?" + A: "Geography, time and location"
**Output**: "VAGUE" with confidence 0.82
**Why critical**: This is the convergence signal. A bad judge either converges on wrong answers (false GOOD) or never converges (false VAGUE).
**What it requires**: Understanding what constitutes a complete, relevant, non-echoed answer to the specific question asked.
**What it does NOT require**: Knowing the correct answer. Only whether the candidate LOOKS like a good answer.
**Evaluation metric**: Judge accuracy — does GOOD correlate with actually correct answers? Target: >80% precision on GOOD. Calibration: when judge says GOOD with 0.9 confidence, the answer should actually be correct >85% of the time.

### Capability 6: QUESTION DECOMPOSITION (76 calls — 4%)
**What**: Break a complex question into simpler sub-questions.
**Input**: "What is the currency of the country where the Great Wall is located?"
**Output**: ["Where is the Great Wall?", "What is the currency of China?"]
**Why critical**: Enables multi-hop reasoning and deep thought patterns. Without decomposition, the engine can only answer single-hop questions.
**What it requires**: Understanding question structure, identifying implicit steps, generating well-formed sub-questions.
**What it does NOT require**: Knowing the answers to the sub-questions.
**Evaluation metric**: Coverage — do the sub-questions, if all answered correctly, provide enough information to answer the original? Target: >75% coverage.

### Capability 7: INFERENCE / REASONING (140 calls — 7%)
**What**: Given accumulated facts, draw connections and insights.
**Input**: Q: "Why do empires fall?" + Facts: ["Rome had military overextension", "Ibn Khaldun described 4-generation cycles", "Tainter showed complexity has diminishing returns"]
**Output**: "All three frameworks suggest empires contain the seeds of their own destruction — success creates conditions that undermine what produced success."
**Why critical**: Deep thought pattern (8% of all questions) requires multi-step reasoning.
**What it requires**: Basic logical inference, comparison, pattern recognition across facts.
**What it does NOT require**: Novel knowledge. All reasoning must be grounded in the provided facts.
**Evaluation metric**: Human-judged insight quality (1-5 scale). Target: avg >3.0. Faithfulness: >85% (no facts invented).

## What the Model Does NOT Need

| Capability | Why NOT needed | What provides it instead |
|-----------|---------------|------------------------|
| World knowledge | All facts from search | Database / Wikipedia / Google |
| Long-form generation (>100 tokens) | Avg output is 15-30 words | Engine composes multiple short outputs |
| Mathematical reasoning | CALCULATE uses eval() | Python math |
| Creative writing | Rare (3% of traces) | Instruct model fallback |
| Multi-language | English-only for now | Future: multilingual encoder |
| Code generation | Not in scope | Separate tool |
| Image understanding | Not in scope | Separate tool |

## Model Architecture Options

### Option A: Single Multi-Task Model (current approach)
SmolLM2-135M fine-tuned with task prefixes: `route:`, `relevant:`, `judge:`, `answer:`, `decompose:`, `synthesize:`, `reason:`

**Pros**: Simple deployment, one model, shared representations.
**Cons**: 135M may be too small for 7 diverse capabilities. Task interference during training.
**Size**: ~100MB (Q4 GGUF)

### Option B: Dual Model (current v3/v4)
Base model (fine-tuned, structured output) + Instruct model (off-the-shelf, generation).

**Pros**: Each model optimized for its task type. Instruct handles synthesis/generation well.
**Cons**: 2x memory, 2x load time.
**Size**: ~200MB (2 × Q4 GGUF)

### Option C: Encoder + Small Decoder (proposed)
MiniLM/DistilBERT encoder (22-66M) for understanding tasks (classify, relevance, judge) + T5-small decoder (60M) for generation tasks (extract, decompose, synthesize).

**Pros**: Each component optimized for its role. Encoder is fast for classification. Decoder only needed for generation steps.
**Cons**: More complex pipeline. Two different model architectures.
**Size**: ~80-120MB total

### Option D: Specialist Ensemble
Separate small models per capability: classifier (10M), relevance judge (10M), extractor (30M), synthesizer (60M).

**Pros**: Each model perfectly sized. Can upgrade one without affecting others.
**Cons**: Complex deployment. More models to maintain.
**Size**: ~110MB total

### Recommendation: Option C (Encoder + Small Decoder)
The data shows a clear split: 70% of model calls are UNDERSTANDING (classify, relevance, judge) and 30% are GENERATION (extract, decompose, synthesize, reason). An encoder handles the 70% faster and better. A small decoder handles the 30%.

## Evaluation Framework: DMRSM Model Scorecard

### Tier 1: Task-Level Metrics (per capability)

| # | Capability | Metric | Target | Test Dataset |
|---|-----------|--------|--------|-------------|
| 1 | Topic Extraction | Search Recall@3 | >80% | 200 NQ questions |
| 2 | Relevance | F1 (YES/NO) | >85% | 500 question-passage pairs |
| 3 | Relevance | ECE (calibration) | <0.15 | Same 500 pairs |
| 4 | Extraction | Exact Match | >65% | SQuAD dev set |
| 5 | Extraction | Token F1 | >75% | SQuAD dev set |
| 6 | Judge | Precision@GOOD | >80% | 200 question-answer pairs |
| 7 | Judge | ECE (calibration) | <0.15 | Same 200 pairs |
| 8 | Decomposition | Coverage | >75% | 100 multi-hop questions |
| 9 | Synthesis | Faithfulness | >90% | 100 multi-fact questions |
| 10 | Reasoning | Insight quality (1-5) | >3.0 | 50 deep-thought questions |

### Tier 2: System-Level Metrics (DMRSM + model together)

| # | Metric | Target | Test |
|---|--------|--------|------|
| 11 | NQ Open-Domain EM | >30% | NQ validation set |
| 12 | TriviaQA EM | >35% | TriviaQA validation |
| 13 | HotPotQA EM (multi-hop) | >20% | HotPotQA validation |
| 14 | Convergence rate | >85% | 200 random questions |
| 15 | Avg steps to converge | <5 | Same 200 questions |
| 16 | Avg searches per Q | <3 | Same 200 questions |
| 17 | Hallucination rate | <10% | 100 factual questions (human-judged) |

### Tier 3: Deployment Metrics

| # | Metric | Target | Measured on |
|---|--------|--------|------------|
| 18 | Model size (quantized) | <200MB | GGUF Q4 |
| 19 | Latency per question (CPU) | <3s avg | 100 questions on laptop |
| 20 | Latency per question (phone) | <5s avg | 100 questions on mid-range Android |
| 21 | RAM usage | <500MB | Runtime peak |
| 22 | First-token latency | <200ms | Time to first model output |

### Tier 3: Safety Metrics (non-negotiable)

| # | Metric | Target | Test |
|---|--------|--------|------|
| 23 | Crisis detection | 100% | 50 crisis phrases |
| 24 | Injection resistance | 100% | 50 injection attempts |
| 25 | DEFER on urgent medical | >95% | 30 urgent medical scenarios |
| 26 | No harmful persona | 100% | 20 harmful roleplay requests |

## Training Data for a Custom Model

The 531 traces we mined ARE the training data format. For each capability:

```
# Route/Classify examples (from traces)
{"input": "What is the capital of France?", "output": "SEARCH(capital of France)", "task": "route"}
{"input": "What is 15% of 230?", "output": "CALCULATE(15% * 230)", "task": "route"}

# Relevance examples
{"input": "question: Capital of France? context: Paris is the capital...", "output": "YES", "confidence": 0.95, "task": "relevant"}
{"input": "question: Capital of France? context: Mold is a type of fungus...", "output": "NO", "confidence": 0.92, "task": "relevant"}

# Extract examples
{"input": "question: When did WWII end? context: ...lasted from 1939 to 1945...", "output": "1945", "task": "answer"}

# Judge examples
{"input": "question: Largest planet? answer: Jupiter", "output": "GOOD", "confidence": 0.95, "task": "judge"}
{"input": "question: Largest planet? answer: Geography and time", "output": "VAGUE", "confidence": 0.82, "task": "judge"}

# Decompose examples (from multi-hop traces)
{"input": "Capital of country where Eiffel Tower is?", "output": "Where is the Eiffel Tower?\nWhat is the capital of France?", "task": "decompose"}

# Synthesize examples (from multi-hop traces)
{"input": "question: Who painted ceiling where Pope lives?\nFacts:\n- Pope lives in Vatican\n- Sistine Chapel in Vatican\n- Michelangelo painted ceiling", "output": "Michelangelo", "task": "synthesize"}
```

**Estimated training data needed:**
- Route: ~2,000 examples (531 from traces + augmentation)
- Relevant: ~5,000 examples (can generate from SQuAD/NQ + negative sampling)
- Extract: ~10,000 examples (SQuAD train set)
- Judge: ~2,000 examples (need to label)
- Decompose: ~500 examples (from multi-hop traces)
- Synthesize: ~1,000 examples (from traces + augmentation)

## The Key Insight: What Makes This Different from an LLM

An LLM is trained to predict the next token from a web-scale corpus. It memorizes facts IN parameters.

The DMRSM model is trained to PROCESS language, not STORE knowledge. Its training objective is:
1. Given a question → what to search for (not what the answer is)
2. Given a passage → is it relevant (not what's in it)
3. Given a passage + question → extract the answer (reading comprehension, not recall)
4. Given multiple facts → combine faithfully (composition, not generation)

**The database IS the knowledge. The model IS the language processor. The engine IS the reasoner.**

This is why a 100MB model can match a 100GB LLM on factual QA — the model only needs to understand language, not memorize Wikipedia.

---

# Original Architecture Plan (reference — DO NOT DELETE)

## Architecture

```
One T5-small (231MB), 6 task prefixes:

"route: Who invented the telephone?"    → SEARCH(telephone inventor)
"query: Who invented the telephone?"    → telephone inventor history patent
"relevant: question: ... context: ..."  → YES / PARTIAL / NO
"grounded: question: ... answer: ... context: ..." → YES / UNSURE / NO
"judge: question: ... answer: ..."      → GOOD / ECHO / VAGUE / TYPE_MISMATCH / TOO_SHORT
"answer: question: ... context: ..."    → Alexander Graham Bell
```

## What It Replaces
- `critic_score()` heuristics (40 lines) → learned "judge:" prefix
- Separate thinker model (231MB) → "query:" prefix in same model
- CONVERGENCE_THRESHOLD (-0.30) → learned GOOD/BAD judgment
- Entity extraction regex → "query:" with accumulated context

## What It Keeps
- Python orchestration loop (simplified to ~20 lines)
- SearchEngine/SearchProvider protocol (unchanged)
- WorkingMemory class (unchanged)
- Pre-filter (crisis/PII/injection — hardcoded, too critical to learn)
- Action dispatch table (35 handlers — unchanged)

## Generalizability
The orchestrator is NOT LM-RAG-specific. Same model works for ANY tool-using system:
- LM-RAG: route → search → check → answer
- Synapse: query → KB search → convergence → respond
- Code assistant: understand → search docs → write code → test
- Data analyst: question → query DB → check → summarize

## Next Steps: Building/Selecting the Model

1. **Extract training data from 531 traces** — convert to supervised format per capability
2. **Augment with existing datasets** — SQuAD (extraction), NQ (route + relevance), NLI (judge)
3. **Choose base model** — MiniLM encoder (22M) + T5-small decoder (60M) OR SmolLM2-135M unified
4. **Fine-tune per capability** — multi-task training with task prefixes
5. **Evaluate on the scorecard** — all 26 metrics, pass/fail per target
6. **Quantize and benchmark** — GGUF Q4, measure latency on CPU/phone
7. **Integrate with DMRSM engine_v4.py** — replace MockModelPool with real model

## Reference Files
- `engine_v4.py` — DMRSM state machine implementation (56/56 tests)
- `DMRSM.md` — full algorithm specification
- `reasoning_traces.jsonl` — 531 training traces
- `reasoning_traces_multiturn.jsonl` — 20 multi-turn conversations
- `analyze_traces.py` — pattern analysis tools
