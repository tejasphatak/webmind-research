# DMRSM-ULI: Combined System Design

## Objective
ChatGPT but honest. Every answer traceable, no hallucination, deterministic, phone-deployable, <200MB. **No LM/LLM — strict.** Only MiniLM encoder (22M params) for embedding similarity. Everything else is rules + database + algorithms.

## The System: One Pipeline

```
text in → ULI reads → AST → DMRSM thinks → AST → ULI writes → text out
```

Three separable data layers:
- **DATABASE** = Knowledge (facts from Wikipedia/search/KB)
- **RULES** = Language (grammar JSON, vocab DB, idioms, templates per language)
- **ENGINE** = Reasoning (DMRSM state machine, 15 patterns from 531 traces)

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                     USER INPUT                            │
│  "Who painted the ceiling where the Pope lives?"          │
└──────────────┬───────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────┐
│  ULI: READER (text → AST)                                 │
│                                                            │
│  L1 LEXER (100% rules)                                     │
│    detect_language() → en                                  │
│    normalize() → fix spelling, expand abbrevs              │
│    tokenize() → [Who, painted, the, ceiling, ...]          │
│                                                            │
│  L2 PARSER (95% rules, 5% spaCy for POS)                  │
│    dependency_parse() → painted(nsubj=Who, dobj=ceiling,   │
│                         nmod=where(Pope, lives))           │
│                                                            │
│  L3 SEMANTICS (70% rules, 30% MiniLM cosine)              │
│    semantic_roles() → QUESTION(agent=?, action=paint,      │
│                       object=ceiling,                      │
│                       location=building(inhabitant=Pope))  │
│    detect_form() → factual_question                        │
│                                                            │
│  OUTPUT: MeaningAST (language-independent)                 │
└──────────────┬───────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────┐
│  DMRSM: THINKER (AST → AST)                              │
│                                                            │
│  classify(AST) → multi_hop                                 │
│  DECOMPOSE: split nested AST nodes → sub-questions         │
│    → [Where does Pope live?, What ceiling?, Who painted?]  │
│  SEARCH each: search_engine → passages                     │
│  RELEVANCE: MiniLM cosine(question_emb, passage_emb)       │
│  EXTRACT: pattern match answer node in passage parse tree  │
│  JUDGE: AST comparison (question coverage, type match)     │
│  SYNTHESIZE: merge fact ASTs                               │
│                                                            │
│  OUTPUT: answer AST {entity: Michelangelo, rel: painted,   │
│          object: Sistine Chapel ceiling}                    │
└──────────────┬───────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────┐
│  ULI: WRITER (AST → text)                                 │
│                                                            │
│  select_template(factual_answer)                           │
│  fill_slots(template, answer_AST)                          │
│  apply_grammar(en.json) → word order, agreement            │
│  render() → "Michelangelo painted the Sistine Chapel       │
│              ceiling."                                     │
│                                                            │
│  OR swap to hindi.json →                                   │
│  "माइकलएंजेलो ने सिस्टिन चैपल की छत चित्रित की"           │
└──────────────────────────────────────────────────────────┘
```

## How Each DMRSM Capability Works WITHOUT an LM

| Capability | Old (with LM) | New (rules + MiniLM) |
|-----------|--------------|---------------------|
| **Topic Extraction** | Model generates search query | ULI parses → AST → extract entity nodes → search terms |
| **Relevance** | Model classifies YES/NO | MiniLM: cosine(question_emb, passage_emb) > threshold |
| **Extraction** | Model generates short answer | Match question AST slots against passage parse tree → extract span |
| **Synthesis** | Model combines facts | Template slot-filling from merged fact ASTs |
| **Judgment** | Model says GOOD/VAGUE | AST comparison: does answer cover question? Type match? Complete? |
| **Decomposition** | Model breaks question | Split nested AST nodes into separate question ASTs |
| **Reasoning** | Model connects facts | Rule-based inference: transitivity, comparison, pattern match on ASTs |

## Data: What Gets Stored as JSON

### Per Language (pluggable — swap files = new language)

```
data/
├── vocab/en.json           # words, POS, senses, morphology, frequency
├── vocab/hi.json
├── grammar/en.json         # word order (SVO), dependency rules, agreement, question formation
├── grammar/hi.json         # word order (SOV), postpositions, gender system
├── normalize/en.json       # abbreviations, contractions, leetspeak, emoji mappings
├── idioms/en.json          # fixed expressions: "kick the bucket" → die
├── registers/en/
│   ├── gen_z.json          # slay, bussin, no cap, rizz
│   ├── academic.json       # formal vocabulary overlay
│   └── legal.json          # legal jargon overlay
└── templates/
    └── discourse.json      # email, essay, chat, poem, paper, legal, math structures
```

### Universal (shared across languages)

```
data/
├── constructions/universal.json  # cross-language patterns (ditransitive, causative, etc.)
└── inference_rules.json          # transitivity, comparison, negation, temporal ordering
```

## MeaningAST: The Universal Representation

```python
@dataclass
class MeaningAST:
    """Language-independent meaning representation."""
    type: str                    # question, statement, command, exclamation
    intent: str                  # factual, comparison, explanation, creative, ...
    
    # Core semantic frame
    predicate: str               # The main action/state (paint, be, have, ...)
    agent: Optional[Entity]      # Who/what does it
    patient: Optional[Entity]    # Who/what it's done to
    theme: Optional[Entity]      # What is being transferred/described
    location: Optional[Entity]   # Where
    time: Optional[Entity]       # When
    manner: Optional[str]        # How
    reason: Optional[str]        # Why
    
    # Modifiers
    negation: bool = False
    modality: str = 'realis'     # realis, irrealis, hypothetical, imperative
    tense: str = 'present'
    aspect: str = 'simple'
    
    # Discourse
    form: str = 'statement'      # question, email, essay, chat, poem, ...
    register: str = 'neutral'    # formal, informal, gen_z, academic, ...
    person: str = 'third'        # first, second, third
    
    # Nested (for complex sentences)
    sub_clauses: List['MeaningAST'] = field(default_factory=list)
    
    # Search terms (extracted from structure)
    entities: List[str] = field(default_factory=list)
    
    # Source tracking
    source: str = ''             # Where this meaning came from (for traceability)
```

## DMRSM v5: Operates on ASTs, Not Text

```python
class ReasoningEngine:
    """DMRSM v5 — operates on MeaningASTs, no LM calls."""
    
    def __init__(self, uli, search_engine, embedder):
        self.uli = uli                    # Universal Language Interpreter
        self.search = search_engine       # Wikipedia/Google/KB
        self.embedder = embedder          # MiniLM (22M) — ONLY neural component
    
    def reason(self, text, conversation=None):
        # Phase 0: Safety
        if not pre_filter(text): return crisis_response()
        
        # Phase 1: READ (ULI)
        ast = self.uli.read(text)
        
        # Phase 2: THINK (DMRSM state machine on ASTs)
        answer_ast = self.think(ast)
        
        # Phase 3: WRITE (ULI)
        return self.uli.write(answer_ast)
    
    def think(self, question_ast):
        """State machine loop — all operations on ASTs, not text."""
        state = ReasoningState(action=STARTING_STATE[question_ast.intent])
        
        while state.steps < MAX_STEPS:
            result = self.execute(state, question_ast)
            state.update(result)
            if result.terminal: break
            state.action = TRANSITIONS[(state.action, result.signal)]
        
        return state.best_answer_ast()
    
    def execute_search(self, question_ast):
        """SEARCH: extract entities from AST → search → parse results."""
        query = ' '.join(question_ast.entities)          # From AST, not model
        results = self.search.search(query)
        if not results: return Result(signal='no_results')
        
        # Parse search result with ULI
        passage_ast = self.uli.read(results[0].text)
        
        # Relevance: MiniLM cosine similarity
        q_emb = self.embedder.encode(query)
        p_emb = self.embedder.encode(results[0].text[:300])
        similarity = cosine(q_emb, p_emb)
        
        if similarity > 0.7:
            return Result(signal='relevant_high', fact_ast=passage_ast, confidence=similarity)
        elif similarity > 0.4:
            return Result(signal='relevant_low', fact_ast=passage_ast, confidence=similarity)
        else:
            return Result(signal='irrelevant', confidence=similarity)
    
    def execute_extract(self, question_ast, passage_ast):
        """EXTRACT: pattern match question slots against passage AST."""
        # Question: QUESTION(agent=?, action=paint, object=ceiling)
        # Passage:  STATEMENT(agent=Michelangelo, action=paint, object=ceiling)
        # Match: agent slot filled by "Michelangelo"
        answer = match_slots(question_ast, passage_ast)
        return answer  # Structural, no model needed
    
    def execute_judge(self, question_ast, answer_ast):
        """JUDGE: structural comparison of ASTs."""
        # Does answer cover what question asks?
        coverage = compute_coverage(question_ast, answer_ast)
        # Type match: "who" question → answer must be person entity
        type_ok = check_type_match(question_ast, answer_ast)
        
        if coverage > 0.8 and type_ok:
            return Result(signal='good', confidence=coverage)
        elif coverage > 0.4:
            return Result(signal='needs_more', confidence=coverage)
        else:
            return Result(signal='unanswerable', confidence=coverage)
    
    def execute_decompose(self, question_ast):
        """DECOMPOSE: split nested AST into sub-question ASTs."""
        # AST: QUESTION(object=ceiling, location=building(inhabitant=Pope))
        # Split: [QUESTION(inhabitant=Pope, location=?),
        #         QUESTION(ceiling=?, location=building),
        #         QUESTION(agent=?, action=paint, object=ceiling)]
        return split_nested_ast(question_ast)
    
    def execute_synthesize(self, question_ast, fact_asts):
        """SYNTHESIZE: merge fact ASTs → answer AST."""
        merged = merge_asts(fact_asts)
        # Fill question slots from merged facts
        answer_ast = fill_from_merged(question_ast, merged)
        return answer_ast
```

## Edge Cases — How Each Is Handled

### Input Quality
| Edge case | Component | Solution |
|-----------|-----------|----------|
| Typos "teh" | ULI Lexer | Edit distance ≤2 vs vocab DB |
| Abbreviations "ppl bc rn" | ULI Lexer | normalize.json lookup |
| Emoji "🔥🔥🔥" | ULI Lexer | Emoji→meaning mapping DB |
| Leetspeak "h4ck3r" | ULI Lexer | Character substitution table |
| Mixed lang "accha yaar" | ULI Lexer | Multi-lang detect, merge grammars |
| Empty / garbage | Pre-filter | Reject before pipeline |

### Ambiguity
| Edge case | Component | Solution |
|-----------|-----------|----------|
| "I saw her duck" | ULI Semantics | MiniLM: embed both interpretations, compare to context |
| "Bank" (river/money) | ULI Semantics | MiniLM cosine with surrounding words vs sense embeddings |
| "Mercury" (planet/element) | ULI Semantics | Same — or ask for clarification |

### Reasoning
| Edge case | Component | Solution |
|-----------|-----------|----------|
| Sarcasm "Oh great" | DMRSM | Surface sentiment ≠ context → invert |
| Multi-hop chains | DMRSM | DECOMPOSE splits AST, SEARCH each |
| Unanswerable | DMRSM | SEARCH finds nothing → GIVE_UP |
| Contradictory facts | DMRSM | JUDGE: compare confidence of conflicting fact ASTs |

### Generation
| Edge case | Component | Solution |
|-----------|-----------|----------|
| Short answer "Paris" | ULI Writer | factual_short template |
| Long answer (essay) | ULI Writer | essay template with N body paragraphs |
| Different language output | ULI Writer | Load target grammar JSON |
| Code output | ULI Writer | Load programming language grammar JSON |

## Test Strategy

### Unit Tests (per component)

**ULI Lexer** (~20 tests):
- Language detection (English, Hindi, French, mixed)
- Spell correction (edit distance 1, 2, no match)
- Abbreviation expansion (ppl→people, bc→because)
- Emoji parsing, number parsing, URL extraction
- Normalization idempotency (normalize(normalize(x)) == normalize(x))

**ULI Parser** (~20 tests):
- Simple sentence → correct dependency tree
- Question → correct wh-extraction
- Compound sentence → correct coordination
- Fragment → handled gracefully
- Garden path → backtrack succeeds

**ULI Semantics** (~20 tests):
- Semantic role extraction (agent, patient, location)
- Idiom detection ("kick the bucket" → die)
- Negation scope ("I don't think he went" → he didn't go)
- Register detection (formal vs informal vs gen_z)
- MiniLM disambiguation ("bank" + "river" context → river_bank)

**ULI Writer** (~15 tests):
- AST → short answer
- AST → full sentence
- AST → paragraph (multi-fact)
- Same AST → English vs Hindi (pluggability)
- Template selection (question→question_template, answer→answer_template)

**DMRSM Engine** (~56 tests — already exist in test_engine_v4.py):
- All 15 patterns tested
- Edge cases (no results, garbage model, max steps)
- Guard rails (max searches, consecutive same action)
- Safety (crisis, injection, triage)

### Integration Tests (~30 tests)

**E2E pipeline** (text → text):
```
test_e2e_factual:         "Capital of France?" → "Paris"
test_e2e_multi_hop:       "Currency of Great Wall country?" → "Yuan"
test_e2e_calculate:       "15% of 230?" → "34.5"
test_e2e_unanswerable:    "Stock market tomorrow?" → "Cannot predict..."
test_e2e_crisis:          "I want to die" → 988 message
test_e2e_disambiguation:  "What is Mercury?" → presents all senses
```

**Cross-language** (same AST, different output):
```
test_cross_lang_en_hi:    AST{Paris, capital, France} → English vs Hindi
test_cross_lang_parse:    Same meaning from English and Hindi inputs → same AST
```

**Multi-turn**:
```
test_multiturn_pronoun:   "Capital of France?" + "Population?" → Paris population
test_multiturn_correct:   "Mona Lisa painter?" + "No, Sistine Chapel" → Michelangelo
```

**Trace validation** (against 531 mined traces):
```
test_trace_patterns:      Run 100 questions from traces, verify action sequences match expected patterns
test_trace_convergence:   Verify convergence rate >85% on trace questions
```

### Benchmark Tests

| Benchmark | Metric | Target |
|-----------|--------|--------|
| SQuAD v2 (extraction) | EM / F1 | >65% / >75% |
| NQ Open (full pipeline) | EM | >25% |
| HotPotQA (multi-hop) | EM | >15% |
| 531 trace questions | Convergence rate | >85% |
| 531 trace questions | Avg steps | <5 |

## File Structure

```
lm-rag/
├── dmrsm_uli.py              # Combined system: read → think → write
├── test_dmrsm_uli.py         # All tests: unit + integration + benchmark
│
├── uli/                       # Universal Language Interpreter
│   ├── __init__.py
│   ├── protocol.py            # LanguageModule protocol + MeaningAST dataclass
│   ├── lexer.py               # L1: detect, normalize, tokenize
│   ├── parser.py              # L2: POS tag, dependency parse (wraps spaCy)
│   ├── semantics.py           # L3: roles, WSD, idioms, register
│   ├── writer.py              # Reverse: AST → text via templates + grammar
│   └── modules/
│       ├── english.py          # English-specific module
│       └── hindi.py            # Hindi module (Phase 2)
│
├── dmrsm/                     # Reasoning engine (refactored from engine_v4.py)
│   ├── __init__.py
│   ├── engine.py              # State machine on ASTs
│   ├── transitions.py         # Transition table (data, not code)
│   ├── actions.py             # SEARCH, JUDGE, EXTRACT, etc on ASTs
│   └── safety.py              # Pre-filter, triage, defer
│
├── data/                      # ALL language data — JSON, no code
│   ├── vocab/en.json
│   ├── vocab/hi.json
│   ├── grammar/en.json
│   ├── grammar/hi.json
│   ├── normalize/en.json
│   ├── idioms/en.json
│   ├── registers/en/gen_z.json
│   ├── templates/discourse.json
│   ├── constructions/universal.json
│   └── inference_rules.json
│
├── search_providers.py        # Existing — Wikipedia/DDG/Google
├── reasoning_traces.jsonl     # 531 traces for validation
└── reasoning_traces_multiturn.jsonl  # 20 conversations
```

## Creativity: Controlled Randomness, Not an LM

Creativity = weighted random selection at three points. Same facts, different presentation.

**1. Template variation** (which structure to use):
```
temperature=0.0: "The sun is hot."                    (deterministic)
temperature=0.5: "The sun blazes with fury."          (metaphor template)
temperature=1.0: "fire in the sky, burning, burning"  (poetic fragment)
```

**2. Synonym selection** (which word from vocab DB):
```
vocab/en.json: "hot" → synonyms: [blazing, scorching, fiery, searing]
temperature=0.0: always "hot"
temperature=0.7: sometimes "scorching" or "fiery"
```

**3. Construction variation** (which grammar pattern):
```
SVO:      "She walked slowly through the garden."
Fronted:  "Through the garden, she walked slowly."
Fragment: "The garden. A slow walk. Her."
```

Facts never change. Only presentation varies. No LM needed — a random number generator + weighted selection from the database.

## Constraints

1. **No LM/LLM** — only MiniLM encoder (22M) for cosine similarity
2. **All language knowledge in JSON** — grammar, vocab, templates are data
3. **Pluggable languages** — swap JSON files = new language
4. **Every answer traceable** — AST chain from input to output
5. **Phone-deployable** — <200MB total, <3s latency on CPU
6. **Deterministic** — same input → same output (no sampling)

## What Exists vs What We Build

| Component | Status | Notes |
|-----------|--------|-------|
| DMRSM state machine | DONE | engine_v4.py, 56/56 tests |
| Search providers | DONE | Wikipedia, DDG, Google |
| 531 reasoning traces | DONE | Validation data |
| MiniLM encoder | AVAILABLE | sentence-transformers installed |
| spaCy parser | NEEDS INSTALL | `pip install spacy` + model |
| ULI Lexer | BUILD | ~200 lines |
| ULI Parser | BUILD | ~150 lines (wraps spaCy + grammar JSON) |
| ULI Semantics | BUILD | ~200 lines |
| ULI Writer | BUILD | ~200 lines |
| MeaningAST | BUILD | ~50 lines (dataclass) |
| DMRSM v5 (on ASTs) | REFACTOR | From engine_v4.py, replace LM calls with AST ops |
| English data JSONs | BUILD | Convert from UD/WordNet |
| Hindi data JSONs | BUILD (Phase 2) | Convert from UD Hindi treebank |
| Combined pipeline | BUILD | ~100 lines glue |
| Tests | BUILD | ~150 tests across all levels |
