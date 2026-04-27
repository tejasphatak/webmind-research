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

## Learning: Training by Talking / Feeding Data

### The Insight
Neural networks learn by updating weights via gradient descent.
DMRSM-ULI learns by **updating JSON databases via INSERT/UPDATE operations**.

Same result: the system gets smarter. Different mechanism: no backprop, no GPU, no retraining.

### What "Training" Means for Each Layer

| Layer | What's learned | How it's learned | Storage |
|-------|---------------|-----------------|---------|
| Vocabulary | New words, senses, forms | Encounter unknown word → add to vocab DB | vocab/{lang}.json |
| Normalization | New abbreviations, slang | User uses unknown abbrev → add to normalize DB | normalize/{lang}.json |
| Idioms | New fixed expressions | Phrase doesn't compose literally → add to idiom DB | idioms/{lang}.json |
| Register | New slang, jargon | Detect register mismatch → add to register overlay | registers/{lang}/{reg}.json |
| Grammar | New constructions | Encounter unparseable-but-valid structure → add rule | grammar/{lang}.json |
| Templates | New discourse forms | Analyze document structure → add template | templates/discourse.json |
| Knowledge | New facts | Verified search result → add to KB | knowledge DB (Synapse) |
| Inference | New reasoning patterns | Trace successful reasoning → add to transition table | transitions.json |

### 3 Learning Modes

#### Mode 1: Learn from Conversation (interactive)
```
User: "ngl that's bussin fr fr"
System: [doesn't know "ngl", "bussin", "fr"]
         → flags unknown tokens
         → if user explains or context clarifies:
           normalize/en.json: "ngl" → "not gonna lie"
           vocab/en.json: "bussin" → {senses: ["excellent"], register: "gen_z"}
           normalize/en.json: "fr" → "for real"
```

```
User: "What is the capital of France?"
System: "Paris"
User: "That's correct!"
         → fact verified → add to KB:
           {question: "capital of France", answer: "Paris", confidence: 1.0, source: "verified"}
```

```
User: "No, 'tabling a motion' in the US means postpone, not bring forward"
System: → correction detected
         → idioms/en.json: update "tabling a motion" with locale tag
           {meaning_us: "postpone", meaning_uk: "bring forward"}
```

#### Mode 2: Learn from Bulk Data (batch)
```python
engine.learn_from_documents([
    {"text": "Dear John, ...", "type": "email"},
    {"text": "Abstract: We present...", "type": "paper"},
    {"text": "yo wya lol", "type": "chat"},
])
# → Extracts: email templates, paper templates, chat patterns
# → Extracts: new vocabulary, new abbreviations
# → Updates: templates/discourse.json, normalize/en.json, vocab/en.json
```

```python
engine.learn_vocabulary("path/to/marathi_corpus.txt", lang="mr")
# → Scans text for words not in vocab/mr.json
# → Infers POS from context (most frequent POS for each unknown word)
# → Adds to vocab/mr.json with frequency counts
```

#### Mode 3: Explicit Teaching (direct)
```python
engine.teach("slay", {
    "pos": ["verb"],
    "senses": ["excel", "impress"],
    "register": "gen_z",
    "formal_equivalent": "did excellently"
})
# → Direct insert into vocab/en.json

engine.teach_idiom("spill the tea", {
    "meaning": "share gossip",
    "literal": False,
    "register": "gen_z"
})
# → Direct insert into idioms/en.json

engine.teach_template("code_review", {
    "structure": ["summary", "issues+", "suggestions", "verdict"],
    "register": "technical"
})
# → Direct insert into templates/discourse.json
```

### How Each Learning Mode Works Mechanically

#### Unknown Word Detection (automatic during read)
```python
def read(self, text):
    tokens = self.tokenize(text)
    for token in tokens:
        if token.text.lower() not in self.vocab:
            self._flag_unknown(token)  # Add to learning queue
    # ... rest of pipeline
```

#### Learning Queue (buffer before committing)
```python
class LearningQueue:
    """Buffer unknown items. Commit after threshold encounters."""
    
    def __init__(self, threshold=3):
        self.threshold = threshold
        self.unknown_words = {}  # word → {count, contexts}
        self.corrections = []
        self.verified_facts = []
    
    def flag_unknown(self, word, context):
        if word not in self.unknown_words:
            self.unknown_words[word] = {'count': 0, 'contexts': []}
        self.unknown_words[word]['count'] += 1
        self.unknown_words[word]['contexts'].append(context[:100])
        
        if self.unknown_words[word]['count'] >= self.threshold:
            self._learn_word(word)
    
    def _learn_word(self, word):
        """Word seen N times → infer POS from contexts → add to vocab."""
        contexts = self.unknown_words[word]['contexts']
        pos = self._infer_pos(word, contexts)  # Most common POS in contexts
        # Add to vocab DB
        self._update_vocab(word, pos)
```

#### Correction Detection (during conversation)
```
User says "No" or "Wrong" or "Actually, ..." or "I meant ..."
→ Previous answer was incorrect
→ If user provides correction:
  1. Update fact in KB (if knowledge correction)
  2. Update vocab/idiom/normalize (if language correction)
  3. Log the correction in learning trace
```

#### Bulk Document Learning
```python
def learn_from_documents(self, docs):
    for doc in docs:
        # 1. Parse with ULI
        ast = self.read(doc['text'])
        
        # 2. Extract new vocabulary
        for entity in ast.entities:
            if entity not in self.vocab:
                self.learning_queue.flag_unknown(entity, doc['text'])
        
        # 3. Extract discourse template
        if doc.get('type'):
            template = self._extract_template(ast, doc['type'])
            self._update_templates(doc['type'], template)
        
        # 4. Extract new abbreviations/slang
        for token in self.tokenize(doc['text']):
            if self._looks_like_abbreviation(token):
                self.learning_queue.flag_unknown(token.text, doc['text'])
```

### What Gets Updated (and what DOESN'T)

| Updated by learning | NOT updated by learning |
|---|---|
| vocab/{lang}.json (new words) | Grammar rules (too risky — manual only) |
| normalize/{lang}.json (new abbreviations) | Safety filters (hardcoded, too critical) |
| idioms/{lang}.json (new expressions) | Transition table (requires trace analysis) |
| registers/{lang}/{reg}.json (new slang) | Core parsing logic (code, not data) |
| templates/discourse.json (new forms) | |
| Knowledge DB (verified facts) | |

Grammar rules are NOT auto-learned — a bad grammar rule breaks parsing for everything. Grammar changes are manual (or from UD treebank updates).

### Self-Evolution (same as Synapse SAQT)

This is exactly Synapse's self-evolution mechanism:
1. User asks question → system answers
2. User confirms (explicitly or implicitly) → fact verified → add to KB
3. System answers a question it previously couldn't → KB grew
4. Over time, system needs fewer searches because answers are in KB

**The KB IS the training data. Every conversation makes it smarter.**

### Implementation

New file: `uli/learner.py`

```python
class Learner:
    """Learning layer — updates JSON databases from conversation and data."""
    
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.queue = LearningQueue(threshold=3)
        self.corrections = []
    
    # Mode 1: Conversation learning
    def on_unknown_word(self, word, context, lang='en'): ...
    def on_correction(self, wrong, right, category): ...
    def on_verified_answer(self, question, answer): ...
    
    # Mode 2: Bulk learning
    def learn_from_documents(self, docs, lang='en'): ...
    def learn_vocabulary(self, corpus_path, lang='en'): ...
    
    # Mode 3: Explicit teaching
    def teach_word(self, word, definition, lang='en'): ...
    def teach_idiom(self, phrase, meaning, lang='en'): ...
    def teach_template(self, name, structure): ...
    
    # Persistence
    def save(self): ...  # Write updated JSONs to disk
```

### Verification

Tests for learning:
```
test_unknown_word_flagged:     Unknown word → added to learning queue
test_threshold_triggers_learn: Word seen 3x → added to vocab
test_correction_updates_db:    User corrects → DB updated
test_teach_word_adds:          Explicit teach → vocab updated
test_teach_idiom_adds:         Teach idiom → idiom DB updated
test_bulk_learns_vocab:        Feed corpus → new words learned
test_bulk_learns_templates:    Feed documents → templates extracted
test_grammar_not_auto_updated: Grammar rules NOT changed by learning
test_safety_not_changed:       Safety filters NOT changed by learning
test_persist_across_restart:   Learned data survives restart (JSON on disk)
```

## Honest Gaps vs LLM — And How to Close Them

### Gap 1: Long-form Generation Feels Mechanical

**The problem:** Templates produce correct but flat text. No rhythm, no flow, no voice. A human writer varies sentence length, uses transitions, builds to a point.

**Why it happens:** We have templates with slots. LLMs have 175B params of "how humans write."

**How to close WITHOUT an LM:**

The "how humans write" knowledge is codifiable — it's called **rhetoric**. 2,400 years of documented rules:

```json
{
  "rhetorical_devices": {
    "anaphora": {"pattern": "{X}. {X}. {X}.", "effect": "emphasis"},
    "antithesis": {"pattern": "Not {A}, but {B}", "effect": "contrast"},
    "rhetorical_question": {"pattern": "Is it not {claim}?", "effect": "persuasion"},
    "tricolon": {"pattern": "{A}, {B}, and {C}", "effect": "completeness"}
  },
  "rhythm_rules": [
    "After 3 long sentences, use a short one.",
    "Vary length: 8, 15, 6, 20, 10 words.",
    "End paragraphs with short punchy sentences."
  ],
  "transition_library": {
    "addition": ["Moreover", "Furthermore", "Additionally"],
    "contrast": ["However", "Nevertheless", "On the other hand"],
    "consequence": ["Therefore", "As a result", "Consequently"],
    "example": ["For instance", "Consider", "Take the case of"]
  },
  "voice_profiles": {
    "conversational": {"sentence_avg": 10, "contractions": true, "questions": "frequent"},
    "academic": {"sentence_avg": 22, "contractions": false, "passive_voice": "allowed"},
    "storytelling": {"sentence_avg": 14, "vary_length": true, "sensory_words": true}
  }
}
```

**The writer composes paragraphs, not just sentences:**
```python
def write_paragraph(facts, profile='conversational'):
    # 1. Pick topic sentence template
    # 2. Pick 2-3 evidence sentence templates  
    # 3. Add transitions between them (from transition_library)
    # 4. Apply rhythm rules (vary sentence length)
    # 5. Optionally insert rhetorical device
    # 6. End with short concluding sentence
```

**Honest remaining gap:** Even with all this, it won't have the emergent "voice" of a human writer. It'll read like good technical writing, not like a conversation with a friend. That said — most practical Q&A doesn't need literary voice. It needs clear, accurate, traceable answers. We optimize for that.

**Closeable: ~80%.** Rhetoric rules + rhythm + transitions + voice profiles get most of the way there.

---

### Gap 2: Implicit Reasoning / Deep Context

**The problem:** "I saw her duck" — is it a noun (bird) or verb (dodge)? MiniLM cosine helps but it's one vector comparison, not deep contextual understanding.

**Why it happens:** We compare two embeddings. LLMs attend across the entire context window with 96+ layers of transformers.

**How to close WITHOUT an LM:**

**Context chain** — maintain a sliding window of topic embeddings:

```python
class ContextChain:
    """Track discourse topic as a running embedding vector."""
    
    def __init__(self, embedder, window=5):
        self.embedder = embedder
        self.window = window
        self.history = []  # List of (text, embedding) pairs
    
    def add(self, text):
        emb = self.embedder.encode(text)
        self.history.append((text, emb))
        if len(self.history) > self.window:
            self.history.pop(0)
    
    def context_embedding(self):
        """Average of recent embeddings = topic vector."""
        if not self.history:
            return None
        return np.mean([emb for _, emb in self.history], axis=0)
    
    def disambiguate(self, word, senses):
        """Pick sense closest to current context."""
        ctx_emb = self.context_embedding()
        best_sense, best_sim = None, -1
        for sense, description in senses.items():
            sense_emb = self.embedder.encode(description)
            sim = np.dot(ctx_emb, sense_emb)
            if sim > best_sim:
                best_sim = sim
                best_sense = sense
        return best_sense
```

**Example:**
```
Context: "We walked to the pond. The children were feeding bread to the birds."
Ambiguous: "I saw her duck"

context_embedding ≈ average(pond, children, feeding, birds)
cosine(context, "bird species") = 0.72
cosine(context, "dodge/crouch") = 0.31
→ duck = bird (noun) ✓
```

This is NOT attention in the transformer sense. It's a running average of MiniLM embeddings — simpler but surprisingly effective for topic tracking.

**Honest remaining gap:** Multi-hop implicit reasoning where the disambiguating context is 20 sentences back, or where it requires world knowledge (not in the text). Example: "The pen is mightier than the sword" — knowing this is metaphorical requires cultural knowledge that a context chain doesn't capture. But idiom DB handles this specific case.

**Closeable: ~70%.** Context chain handles most discourse-level ambiguity. Deep implicit reasoning remains a genuine gap.

---

### Gap 3: Creative Synthesis — Novel Ideas, Not Just Variation

**The problem:** Temperature-controlled randomness produces variation (different words, different templates), not genuine creativity (new ideas that didn't exist in the inputs).

**Why it happens:** We select from what's in the database. LLMs interpolate in latent space, sometimes producing genuinely novel combinations.

**How to close WITHOUT an LM:**

Creativity = **structured combination of existing concepts**. Research shows even human creativity follows patterns (Boden's 3 types):

**1. Combinational creativity** — merge two unrelated concepts:
```python
def combine(concept_a, concept_b, embedder):
    """Find structural bridge between two concepts."""
    # "cloud" + "computing" → "cloud computing"
    # "artificial" + "intelligence" → "AI"
    
    # Find shared properties via embedding
    emb_a = embedder.encode(concept_a)
    emb_b = embedder.encode(concept_b)
    
    # Midpoint in embedding space = the blend
    blend = (emb_a + emb_b) / 2
    
    # Find words near the blend that aren't either original
    # This is word2vec analogy: king - man + woman = queen
    return find_nearest_words(blend, exclude=[concept_a, concept_b])
```

**2. Exploratory creativity** — push boundaries of a pattern:
```python
def explore(template, temperature=0.8):
    """Fill template with increasingly unusual slot fillers."""
    # Normal: "The {ADJ} {NOUN}" → "The big house"
    # Creative: → "The whispering house" (synesthesia — houses don't whisper)
    # More creative: → "The house that remembered" (personification)
    
    # At high temperature, select from semantically distant but 
    # structurally valid fillers
```

**3. Analogical creativity** — map structure from one domain to another:
```python
def analogize(source_ast, target_domain, embedder):
    """Map relationships from source to target domain."""
    # Source: "The atom has a nucleus orbited by electrons"
    # Target domain: "solar system"
    # Output: "The solar system has a sun orbited by planets"
    
    # Find structural alignment between ASTs
    # Map entities: nucleus↔sun, electrons↔planets, orbit↔orbit
```

All three use MiniLM embeddings + AST structure. No generation needed.

**Honest remaining gap:** These produce structured novelty, not the fluid, surprising leaps that human creativity (or LLM hallucination, sometimes productively) achieves. The system will never write a poem that makes you cry. But it can write one that's structurally interesting.

**Closeable: ~50%.** Combinational and analogical creativity are achievable. Genuine artistic voice is not.

---

### Gap 4: Pragmatics — Sarcasm, Implicature, Reading Between Lines

**The problem:** Rule-based DMRSM will miss cases where meaning depends on social context, power dynamics, emotional state, shared history.

**Why it happens:** Pragmatics requires a model of the OTHER PERSON'S mind (theory of mind). Rules can capture patterns but not the full flexibility.

**How to close WITHOUT an LM:**

**Pragmatic pattern library** — most pragmatic phenomena are catalogued in linguistics:

```json
{
  "sarcasm_patterns": [
    {"pattern": "Oh {POSITIVE}, {NEGATIVE_CONTEXT}", "meaning": "invert_sentiment"},
    {"pattern": "Yeah, {UNLIKELY_CLAIM}", "meaning": "disbelief"},
    {"pattern": "Thanks for {INCONVENIENCE}", "meaning": "complaint"}
  ],
  "implicature_patterns": [
    {"pattern": "Can you {PHYSICAL_ACTION}?", "meaning": "request", "not": "ability_question"},
    {"pattern": "Do you know {INFORMATION}?", "meaning": "request_info", "not": "yes_no"},
    {"pattern": "It's cold in here", "context": "window_open", "meaning": "close_window"}
  ],
  "understatement_patterns": [
    {"pattern": "not {NEGATIVE}", "meaning": "positive", "example": "not bad = good"},
    {"pattern": "a bit {EXTREME}", "meaning": "very", "example": "a bit dangerous = very dangerous"}
  ],
  "politeness_markers": [
    {"pattern": "Would you mind...", "meaning": "request", "register": "formal"},
    {"pattern": "I was wondering if...", "meaning": "request", "register": "hedged"}
  ]
}
```

**Context-aware pragmatic inference:**
```python
def detect_sarcasm(ast, context_chain):
    """Check if surface sentiment contradicts context."""
    surface = ast_sentiment(ast)  # positive/negative from words
    context = context_sentiment(context_chain)  # recent topic sentiment
    
    # Sarcasm signal: positive words + negative context
    if surface == 'positive' and context == 'negative':
        # Check for sarcasm pattern match
        if matches_sarcasm_pattern(ast):
            return True, 'invert'
    return False, None
```

**Conversational history as pragmatic context:**
```python
def resolve_implicature(ast, conversation):
    """Use conversation history to resolve indirect speech acts."""
    # "Can you pass the salt?" after dinner context → request
    # "Can you pass the salt?" after "what can you lift?" → ability question
    
    recent_topic = conversation[-3:]  # Last 3 turns
    topic_embedding = embedder.encode(' '.join(t['text'] for t in recent_topic))
    
    # Compare to known implicature patterns
    for pattern in IMPLICATURE_PATTERNS:
        if pattern_matches(ast, pattern):
            context_fit = embedder.similarity(topic_embedding, pattern['context_embedding'])
            if context_fit > 0.5:
                return pattern['meaning']
    
    return None  # No implicature detected → literal interpretation
```

**Honest remaining gap:** Subtle social dynamics. "Nice shirt" from your friend vs from someone you just beat in a competition. Same words, opposite meaning — resolved by relationship history and power dynamics. Rules can't capture every social nuance.

**Closeable: ~60%.** Common patterns (sarcasm, implicature, politeness, understatement) are well-catalogued. Subtle social inference is a genuine gap.

---

### Summary: What's Closeable vs What's a Genuine Limitation

| Gap | Closeable | Method | Genuine limitation |
|-----|-----------|--------|-------------------|
| Long-form generation | ~80% | Rhetoric rules, rhythm, transitions, voice profiles | Emergent personal voice |
| Implicit reasoning | ~70% | Context chain (MiniLM sliding window) | Deep multi-hop implicit context |
| Creative synthesis | ~50% | Combinational + analogical creativity via embeddings | Artistic surprise, emotional resonance |
| Pragmatics | ~60% | Pattern library + context-aware inference | Subtle social dynamics, power relationships |

**The meta-answer:** These gaps exist because we separated language from knowledge from reasoning. An LLM smears all three together, which gives it flexibility at the cost of traceability and size. Our system is traceable and small at the cost of some flexibility.

The question isn't "can we match an LLM?" — it's "can we handle 80-90% of real-world queries with a traceable, phone-deployable system?" The answer is yes, especially for factual, procedural, and comparison tasks. For creative writing, emotional conversation, and subtle social dynamics — those are genuine gaps we should be honest about.

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
