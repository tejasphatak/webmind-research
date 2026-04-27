# Universal Language Interpreter (ULI)

## The Three-Layer Architecture

```
Database = Knowledge (facts, Wikipedia, search results)
Rules    = Language  (grammar JSON, vocabulary DB, templates, idioms)
Engine   = Reasoning (DMRSM state machine)
```

Swap rules → new language. Swap database → new domain. Engine stays the same.

## The Pipeline

```
Input Text (any language, any form, any quality)
    │
    ▼
┌─────────────────────────────────────────────────┐
│  LAYER 1: LEXER (100% rules + DB, 0% model)     │
│                                                   │
│  1. Language detection (n-gram, script analysis)  │
│  2. Load lang rules: grammar/{lang}.json          │
│  3. Normalize: spelling fix (edit distance ≤2),   │
│     abbreviations (DB lookup), emoji (mapping),   │
│     number parsing, URL/mention extraction        │
│  4. Tokenize using language-specific rules        │
│                                                   │
│  Output: clean token stream + language ID         │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│  LAYER 2: PARSER (95% rules, 5% model)           │
│                                                   │
│  1. POS tagging (from UD rules)                   │
│  2. Dependency parsing (grammar rules JSON)       │
│  3. Handle: garden paths (backtrack parser),       │
│     ellipsis (parallel structure recovery),        │
│     fragments (context-aware completion)           │
│  4. Code-switching detection (multi-lang merge)   │
│                                                   │
│  Output: dependency tree                          │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│  LAYER 3: SEMANTICS (70% rules, 30% model)       │
│                                                   │
│  1. Word sense disambiguation (frequency + ctx)   │
│  2. Idiom detection (idiom DB lookup)             │
│  3. Semantic role labeling (who did what to whom) │
│  4. Negation scope resolution (scope rules)       │
│  5. Register/formality detection                  │
│                                                   │
│  Output: Meaning AST (language-independent)       │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│  LAYER 4: PRAGMATICS (= DMRSM reasoning)        │
│                                                   │
│  1. Intent detection (question? command? opinion?)│
│  2. Sarcasm/irony (surface vs context mismatch)  │
│  3. Implicature ("Can you pass the salt?" = req) │
│  4. Multi-turn context (pronoun resolution, etc) │
│  5. Discourse form detection (email/essay/chat)  │
│                                                   │
│  This IS the reasoning engine. Not language.      │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
              Meaning AST
         (structured, universal,
          language-independent)
```

## What's Stored as Data (JSON / Database)

### Per Language (pluggable — swap file = new language)

**vocab/{lang}.json** — Word database
```json
{
  "words": {
    "run": {
      "pos": ["verb", "noun"],
      "forms": {"past": "ran", "participle": "running", "gerund": "running"},
      "senses": ["move_fast", "operate", "manage", "sequence"],
      "frequency": 0.0023
    }
  },
  "morphology": {
    "plural": [{"match": ".*s$", "add": "es"}, {"match": ".*y$", "sub": "ies"}, {"default": "+s"}],
    "past_tense": [{"match": ".*e$", "add": "d"}, {"default": "+ed"}]
  }
}
```

**grammar/{lang}.json** — Syntax rules (from Universal Dependencies)
```json
{
  "word_order": "SVO",
  "dependencies": [
    {"rel": "nsubj", "head": "verb", "dep": "noun", "position": "before_head"},
    {"rel": "dobj", "head": "verb", "dep": "noun", "position": "after_head"},
    {"rel": "amod", "head": "noun", "dep": "adj", "position": "before_head"}
  ],
  "question_formation": {
    "yes_no": "AUX SUBJ VERB OBJ",
    "wh": "WH AUX SUBJ VERB"
  },
  "agreement": [
    {"between": ["subj", "verb"], "feature": "number"},
    {"between": ["det", "noun"], "feature": "number"}
  ]
}
```

**registers/{lang}/{register}.json** — Slang, dialect, jargon overlays
```json
{
  "register": "gen_z",
  "parent": "english",
  "words": {
    "slay": {"senses": ["excel"], "formal": "did excellently"},
    "bussin": {"senses": ["very_good"], "formal": "excellent"},
    "no cap": {"senses": ["truthfully"], "formal": "honestly"}
  },
  "constructions": [
    {"pattern": "it's giving {NOUN}", "meaning": "resembles"},
    {"pattern": "lowkey {ADJ/VERB}", "meaning": "moderate_degree"}
  ]
}
```

**idioms/{lang}.json** — Fixed expressions
```json
{
  "kick the bucket": {"meaning": "die", "literal": false},
  "raining cats and dogs": {"meaning": "raining heavily", "literal": false},
  "tabling a motion": {"meaning_us": "postpone", "meaning_uk": "bring forward", "locale_dependent": true}
}
```

**templates/discourse.json** — Text structure patterns (universal)
```json
{
  "email": ["greeting", "context?", "body", "closing", "signature"],
  "essay": ["introduction(hook,context,thesis)", "body_paragraph(topic,evidence,analysis)+", "conclusion"],
  "research_paper": ["abstract", "intro", "related_work?", "method", "results", "discussion", "conclusion"],
  "poem_haiku": {"lines": 3, "syllables": [5, 7, 5]},
  "chat": {"turns": "short", "features": ["informal", "contractions", "emoji"]},
  "math_proof": ["given", "to_prove", "proof_steps+", "qed"]
}
```

## Edge Cases — Complete Analysis

### Handled by rules + DB alone (no model)

| Edge Case | Layer | Solution |
|-----------|-------|----------|
| Typos / spelling | Lexer | Edit distance ≤2 against vocab DB |
| Abbreviations (ppl, bc, rn) | Lexer | normalize.json lookup |
| Emoji | Lexer | Emoji → meaning mapping DB |
| Number formats ($5.99, 3rd) | Lexer | Regex patterns per locale |
| URLs, @mentions, #hashtags | Lexer | Regex extraction before parse |
| Slang (bussin, rizz, no cap) | Lexer | Register overlay JSON |
| Historical language (thou art) | Lexer | Period register JSON |
| RTL / mixed direction | Lexer | Unicode bidi algorithm |
| Idioms (kick the bucket) | Semantics | Idiom DB lookup |
| Negation scope | Semantics | Scope rules in grammar JSON |
| Agreement (he goes / they go) | Parser | Agreement rules |
| Word order (SVO vs SOV) | Parser | Per-language rule |
| Locale variants (US vs UK) | Semantics | Locale-tagged DB entries |
| Morphology (ran←run) | Lexer | Irregular forms in vocab DB |
| Garden path sentences | Parser | Backtracking parser |
| Language evolution | Lexer | Update JSON rows |

### Needs small model (~22M params, disambiguation only)

| Edge Case | Layer | Why |
|-----------|-------|-----|
| "I saw her duck" (verb/noun) | Semantics | Context-dependent POS |
| "Bank" (river/money) | Semantics | Word sense needs sentence context |
| Ellipsis recovery | Parser | Infer deleted verb from structure |
| Code-switching boundaries | Lexer | Where does Hindi end, English begin? |
| Register shift mid-text | Semantics | Detect formal→informal boundary |

### Handled by DMRSM (reasoning, not language)

| Edge Case | Why it's reasoning |
|-----------|-------------------|
| Sarcasm ("Oh great, another meeting") | Surface ≠ intent. Context reasoning. |
| Implicature ("Can you pass salt?") | Literal question = implied command. |
| Metaphor ("time is money") | Non-literal mapping. Conceptual reasoning. |
| Passive-aggressive "..." | Tone from context. Social reasoning. |
| "Not bad" = quite good | Scale reasoning. |

## Language Evolution = Database Updates

```
1600: "thou art"  → register: standard
1800: "thou art"  → register: archaic. "you are" = standard.
2020: "u r"       → register: digital_informal
2025: "ur"        → register: gen_z

All parse to: {subject: 2nd_person, verb: be}
```

New slang = new JSON row. Grammar shift = rule update. No retraining.

## Implementation Plan

### Phase 1: Protocol + AST format
```python
class LanguageModule(Protocol):
    def detect(self, text: str) -> str: ...
    def normalize(self, text: str) -> str: ...
    def tokenize(self, text: str) -> List[Token]: ...
    def parse(self, tokens: List[Token]) -> DependencyTree: ...
    def to_ast(self, tree: DependencyTree) -> MeaningAST: ...
    def from_ast(self, ast: MeaningAST) -> str: ...  # generator
```

### Phase 2: English MVP (lexer + parser using spaCy/UD)
### Phase 3: Mine discourse templates (same as DMRSM trace mining)
### Phase 4: Connect to DMRSM engine_v4.py
### Phase 5: Add Hindi module (prove pluggability)

## File Structure

```
lm-rag/
├── engine_v4.py              # DMRSM reasoning (done, 56/56)
├── language/
│   ├── protocol.py           # LanguageModule protocol + MeaningAST
│   ├── lexer.py              # Universal lexer
│   ├── parser.py             # Universal parser (UD-based)
│   ├── semantics.py          # WSD, roles, idioms
│   ├── generator.py          # AST → text
│   └── modules/
│       ├── english.py
│       └── hindi.py
├── data/
│   ├── vocab/{lang}.json
│   ├── grammar/{lang}.json
│   ├── normalize/{lang}.json
│   ├── registers/{lang}/{register}.json
│   ├── idioms/{lang}.json
│   ├── templates/discourse.json
│   └── constructions/universal.json
```

## The Key Insight

LLMs conflate three things into one parameter matrix:
1. **Vocabulary** (words, meanings) → should be DATABASE
2. **Grammar** (how words combine) → should be RULES
3. **Pragmatics** (what speaker means) → should be REASONING

We separate them. That's why:
- New slang → add DB row (no retrain)
- New language → add JSON files (no retrain)
- Language evolves → update rules (no retrain)
- Sarcasm → reasoning engine handles it (not language layer)
- Runs on phone → rules + DB + small model = ~100MB
