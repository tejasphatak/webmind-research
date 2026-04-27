# Pattern-Driven Reasoning Engine v4

## Context
We mined 531 reasoning traces from Claude Opus and Gemini 2.5 Pro across 44 question categories. The traces revealed 15 generalized reasoning patterns, a core action transition graph, and clear starting/ending distributions. The current engine.py uses a generic DFS with hardcoded strategy ordering. The goal: replace it with a **state machine** driven by the patterns we discovered from data.

## The Algorithm: Guided State Machine with Working Memory

### Core Insight from Data
Every reasoning trace follows the same meta-pattern:
```
CLASSIFY → [ACTION → EVALUATE → TRANSITION]* → TERMINAL
```
The variation is in which actions, which transitions, and how many loops.

### Data Structure: ReasoningState

```python
@dataclass
class ReasoningState:
    # Current position in the state machine
    action: str              # Current action: SEARCH, JUDGE, EXTRACT, etc.
    
    # Working memory (accumulates across steps)
    facts: List[str]         # Extracted facts from searches
    sub_questions: List[str] # Decomposed sub-questions
    partial_answers: Dict[str, str]  # sub_question → answer
    confidence: float        # Running confidence (0-1)
    searches_done: int       # Search count
    steps_done: int          # Total steps
    
    # Context
    question: str            # Original user question
    question_type: str       # Classified type
    persona: Optional[str]   # If roleplay, the persona constraints
    safety_level: str        # safe, caution, urgent
    
    # Multi-turn
    conversation: List[dict] # Prior turns for RECALL
    
    # Trace (for debugging)
    trace: List[dict]        # Log of all actions + results
```

### Data Structure: Transition Table

The transition table encodes the state machine. Each entry maps:
`(current_action, result_quality) → next_action`

```python
# Derived from 531 traces — action bigram frequencies
TRANSITIONS = {
    # After SEARCH (most common starting action — 74%)
    ('SEARCH', 'relevant_high'):   'EXTRACT',     # conf > 0.85 → extract directly
    ('SEARCH', 'relevant_low'):    'JUDGE',        # conf < 0.85 → judge first
    ('SEARCH', 'irrelevant'):      'SEARCH',       # try different query (max 3)
    ('SEARCH', 'ambiguous'):       'DISAMBIGUATE',
    ('SEARCH', 'no_results'):      'DECOMPOSE',    # break question apart
    ('SEARCH', 'partial'):         'REASON',        # think about what's missing
    
    # After JUDGE (2nd most common — SEARCH→JUDGE = 276 transitions)
    ('JUDGE', 'good'):             'EXTRACT',
    ('JUDGE', 'needs_more'):       'SEARCH',        # search again with refined query
    ('JUDGE', 'needs_synthesis'):  'SYNTHESIZE',
    ('JUDGE', 'unanswerable'):     'GIVE_UP',
    ('JUDGE', 'needs_expert'):     'DEFER',
    
    # After EXTRACT (often terminal — 30% of endings)
    ('EXTRACT', 'complete'):       'DONE',          # terminal
    ('EXTRACT', 'needs_more'):     'SEARCH',        # need more info
    ('EXTRACT', 'multiple'):       'SYNTHESIZE',    # combine multiple extracts
    
    # After DECOMPOSE (10% of starts)
    ('DECOMPOSE', 'has_subs'):     'SEARCH',        # search each sub-question
    ('DECOMPOSE', 'cant_break'):   'SEARCH',        # fallback to direct search
    
    # After REASON (deep thought pattern)
    ('REASON', 'need_evidence'):   'SEARCH',
    ('REASON', 'insight'):         'SYNTHESIZE',
    ('REASON', 'more_angles'):     'REASON',         # loop: think more
    
    # After TRIAGE (professional roles — 7% of starts)
    ('TRIAGE', 'safe'):            'SEARCH',
    ('TRIAGE', 'caution'):         'SEARCH',         # + flag for DEFER at end
    ('TRIAGE', 'urgent'):          'DEFER',           # immediate referral
    
    # After PERSONA_ADOPT (roleplay — 3% of starts)  
    ('PERSONA_ADOPT', None):       'SEARCH',          # then PERSONA_FILTER before output
    
    # After SYNTHESIZE (most common ending — 48%)
    ('SYNTHESIZE', 'complete'):    'DONE',
    ('SYNTHESIZE', 'weak'):        'SEARCH',          # need more evidence
    
    # Terminal states
    ('GIVE_UP', None):             'DONE',
    ('DEFER', None):               'SYNTHESIZE',      # synthesize what we know + referral
    ('CALCULATE', 'done'):         'DONE',
    ('GENERATE', 'done'):          'DONE',
}
```

### Algorithm: `reason(question)` — The Main Loop

```python
def reason(question, conversation=None):
    """
    Pattern-driven reasoning loop.
    
    Returns: (answer, trace, confidence)
    """
    MAX_STEPS = 12        # Deep thought avg=7.8, max ~10
    MAX_SEARCHES = 6      # Deep thought avg=4-5
    CONFIDENCE_THRESHOLD = 0.7
    
    # ── Step 0: Classify question type ──────────────────
    # Model decides: what KIND of question is this?
    # This determines the STARTING STATE.
    q_type = classify(question)  # → factual, multi_hop, deep_thought, 
                                 #   roleplay, medical, math, etc.
    
    # ── Step 1: Determine starting state ────────────────
    # From data: SEARCH (74%), DECOMPOSE (10%), TRIAGE (7%), 
    #            PERSONA_ADOPT (3%), CALCULATE (3%)
    STARTING_STATE = {
        'factual':       'SEARCH',
        'multi_hop':     'DECOMPOSE',
        'deep_thought':  'DECOMPOSE',    # always decompose into angles
        'comparison':    'DECOMPOSE',    # decompose into entity lookups
        'temporal':      'SEARCH',       # + flag for freshness verification
        'math':          'CALCULATE',
        'roleplay':      'PERSONA_ADOPT',
        'medical':       'TRIAGE',
        'legal':         'TRIAGE',
        'financial':     'SEARCH',       # most financial Qs are safe to answer
        'therapy':       'TRIAGE',
        'creative':      'GENERATE',     # optional search for reference first
        'unanswerable':  'SEARCH',       # search first, THEN give up
        'disambiguation':'SEARCH',       # search reveals the ambiguity
        'negation':      'SEARCH',       # search for full set, then exclude
    }
    
    state = ReasoningState(
        action=STARTING_STATE.get(q_type, 'SEARCH'),
        question=question,
        question_type=q_type,
        conversation=conversation or [],
        # ... init empty lists, 0 counts, etc.
    )
    
    # ── Step 2: Multi-turn context check ────────────────
    # If conversation history exists, check for:
    #   - Pronoun resolution ("its", "he", "that")  
    #   - Correction ("no, I meant X")
    #   - Drill-down (same topic, deeper)
    #   - Topic callback ("going back to...")
    if state.conversation:
        resolved = recall_and_resolve(question, state.conversation)
        if resolved.is_correction:
            # Discard previous topic, re-classify
            state.question = resolved.corrected_question
        elif resolved.has_pronoun_ref:
            # Expand pronouns from context
            state.question = resolved.expanded_question
        elif resolved.is_callback:
            # Restore earlier topic from memory
            state.facts = resolved.recalled_facts
    
    # ── Step 3: The Reasoning Loop ──────────────────────
    while state.steps_done < MAX_STEPS:
        
        # Execute current action
        result = execute_action(state)
        
        # Log to trace
        state.trace.append({
            'step': state.steps_done,
            'action': state.action,
            'input': result.input_used,
            'output': result.output,
            'confidence': result.confidence,
        })
        state.steps_done += 1
        
        # Update working memory
        if result.facts:
            state.facts.extend(result.facts)
        if result.answer:
            state.partial_answers[state.action] = result.answer
        state.confidence = max(state.confidence, result.confidence)
        if result.searched:
            state.searches_done += 1
        
        # ── Convergence check ───────────────────────────
        # Terminal states: DONE, or action produced a final answer
        if state.action in ('DONE',):
            break
        if result.is_terminal and state.confidence >= CONFIDENCE_THRESHOLD:
            break
        
        # ── Transition to next state ────────────────────
        # Look up: (current_action, result_quality) → next_action
        transition_key = (state.action, result.quality)
        next_action = TRANSITIONS.get(transition_key)
        
        if next_action is None:
            # No explicit transition — ask the model
            next_action = model_decide_next(state)
        
        # Guard rails
        if state.searches_done >= MAX_SEARCHES and next_action == 'SEARCH':
            next_action = 'SYNTHESIZE'  # force synthesis
        if state.steps_done >= MAX_STEPS - 1 and next_action not in ('DONE', 'SYNTHESIZE', 'GIVE_UP'):
            next_action = 'SYNTHESIZE'  # force wrap-up
        
        state.action = next_action
    
    # ── Step 4: Post-processing ─────────────────────────
    # Persona filter (roleplay)
    if state.persona:
        answer = persona_filter(state.best_answer(), state.persona)
    else:
        answer = state.best_answer()
    
    # Safety gate (professional roles)
    if state.safety_level in ('caution', 'urgent'):
        answer = append_defer_message(answer, state.question_type)
    
    return answer, state.trace, state.confidence


def execute_action(state):
    """Execute the current action and return result."""
    
    if state.action == 'SEARCH':
        # What to search? Use sub-questions if available, else model picks topic
        query = state.sub_questions.pop(0) if state.sub_questions else extract_topic(state.question)
        results = search_engine.search(query)
        if not results:
            return Result(quality='no_results', searched=True)
        
        # Judge relevance with confidence
        best = results[0]
        rel, conf = model.call_with_confidence('relevant', 
            f"question: {state.question} context: {best.text[:300]}")
        
        if rel == 'NO':
            return Result(quality='irrelevant', confidence=conf, searched=True)
        elif conf > 0.85:
            return Result(quality='relevant_high', facts=[best.text[:500]], 
                         confidence=conf, searched=True)
        else:
            return Result(quality='relevant_low', facts=[best.text[:500]], 
                         confidence=conf, searched=True)
    
    elif state.action == 'JUDGE':
        # Judge the best partial answer so far
        answer = state.best_partial_answer()
        judge, conf = model.call_with_confidence('judge',
            f"question: {state.question} answer: {answer}")
        
        if judge == 'GOOD' and conf > 0.8:
            return Result(quality='good', confidence=conf)
        elif 'ECHO' in judge or 'VAGUE' in judge:
            return Result(quality='needs_more', confidence=conf)
        else:
            return Result(quality='needs_synthesis', confidence=conf)
    
    elif state.action == 'EXTRACT':
        # Extract answer from accumulated facts
        context = ' '.join(state.facts[-3:])  # Last 3 facts
        answer = model.call('answer', 
            f"question: {state.question} context: {context}", max_len=60)
        
        if answer and len(answer) > 3:
            return Result(quality='complete', answer=answer, 
                         confidence=0.8, is_terminal=True)
        return Result(quality='needs_more', confidence=0.3)
    
    elif state.action == 'DECOMPOSE':
        subs = model_decompose(state.question)
        state.sub_questions = subs
        if subs:
            return Result(quality='has_subs')
        return Result(quality='cant_break')
    
    elif state.action == 'REASON':
        # Model thinks without searching — connects facts
        reasoning = model.call('reason',
            f"question: {state.question} facts: {state.facts_summary()}")
        
        if 'need' in reasoning.lower() or 'search' in reasoning.lower():
            return Result(quality='need_evidence', facts=[reasoning])
        return Result(quality='insight', facts=[reasoning], confidence=0.7)
    
    elif state.action == 'SYNTHESIZE':
        # Combine all partial answers + facts into final answer
        parts = '\n'.join(f"- {a}" for a in state.partial_answers.values())
        facts = '\n'.join(state.facts[-5:])
        answer = model.call('synthesize',
            f"question: {state.question}\nFacts:\n{facts}\nPartial answers:\n{parts}",
            max_len=100)
        return Result(quality='complete', answer=answer, 
                     confidence=state.confidence, is_terminal=True)
    
    elif state.action == 'TRIAGE':
        # Assess urgency for professional role questions
        level = assess_safety(state.question)  # → safe/caution/urgent
        state.safety_level = level
        if level == 'urgent':
            return Result(quality='urgent')
        return Result(quality=level)
    
    elif state.action == 'PERSONA_ADOPT':
        state.persona = model.call('persona',
            f"Define character constraints for: {state.question}")
        return Result(quality=None)
    
    elif state.action == 'CALCULATE':
        result = safe_eval(extract_expression(state.question))
        return Result(quality='done', answer=str(result), 
                     confidence=0.99, is_terminal=True)
    
    elif state.action == 'GENERATE':
        answer = model.call('generate', state.question, max_len=150)
        return Result(quality='done', answer=answer, 
                     confidence=0.8, is_terminal=True)
    
    elif state.action == 'GIVE_UP':
        return Result(quality=None, answer="I couldn't find a confident answer.",
                     is_terminal=True)
    
    elif state.action == 'DEFER':
        return Result(quality=None, 
                     answer="Please consult a professional for this.",
                     is_terminal=False)  # still synthesize what we know
```

### Tracking Variables (the user asked for these specifically)

```python
# ── Per-question tracking ──────────────────────────────
state.confidence        # 0-1, max of all step confidences
state.searches_done     # int, compared against MAX_SEARCHES
state.steps_done        # int, compared against MAX_STEPS  
state.facts             # list[str], accumulated evidence
state.partial_answers   # dict, sub-question → answer
state.sub_questions     # list[str], queue of remaining sub-Qs
state.trace             # list[dict], full action log
state.safety_level      # safe|caution|urgent, for professional roles
state.persona           # str|None, roleplay character constraints

# ── Per-step tracking ──────────────────────────────────
result.quality          # str, determines transition (relevant_high, good, etc.)
result.confidence       # 0-1, from model log-probs  
result.facts            # list[str], new facts from this step
result.answer           # str|None, extracted answer
result.searched         # bool, whether a search was performed
result.is_terminal      # bool, whether this result ends the loop

# ── Multi-turn tracking ────────────────────────────────
conversation            # list[dict], prior turns
resolved.is_correction  # bool, "no I meant X"
resolved.has_pronoun_ref # bool, "its", "he", "that"
resolved.is_callback    # bool, "going back to..."
```

### Why This Design

1. **State machine** (not DFS tree) — the data shows reasoning is sequential with backtracking, not tree-shaped. 74% of traces start with SEARCH, loop 2-8 times, end with EXTRACT or SYNTHESIZE. A state machine captures this naturally.

2. **Transition table** (not hardcoded if/else) — each transition is a data entry, not code. Adding a new pattern = adding rows to the table. The model can also override the table when no explicit transition exists.

3. **Working memory** (not just traces) — the current engine accumulates AgentTraces but doesn't track intermediate state. The new design has explicit facts[], partial_answers{}, sub_questions[] that persist across the loop.

4. **Confidence as the convergence signal** — not binary YES/NO. From our trace data: direct lookups converge at conf > 0.85, deep thought at conf > 0.6. The threshold can vary by question type.

5. **Guard rails** — MAX_STEPS (12) and MAX_SEARCHES (6) prevent infinite loops. When limits are hit, the engine forces SYNTHESIZE with whatever it has.

### Comparison to Current Engine

| Aspect | Current (v3) | New (v4) |
|--------|-------------|----------|
| Structure | DFS tree with strategies 1/2/3 | State machine loop |
| Routing | 4 intents (SEARCH/CALCULATE/MEMORY/RESPOND) | 15+ question types |
| Transitions | Hardcoded strategy order | Data-driven transition table |
| Memory | AgentTrace list | Structured working memory |
| Convergence | Binary (YES+GOOD) | Confidence threshold (0-1) |
| Multi-turn | None | RECALL + pronoun resolution |
| Roleplay | None | PERSONA_ADOPT + PERSONA_FILTER |
| Professional | None | TRIAGE + DEFER + safety levels |
| Deep thought | None | DECOMPOSE + REASON loops |

## Algorithm: Data-Mined Reasoning State Machine (DMRSM)

### Problem Statement
Given a natural language question Q, a search engine S, and a language model M with no stored knowledge, produce an answer A by orchestrating search, judgment, extraction, and synthesis operations in a pattern derived from empirical trace analysis of expert reasoning.

### Formal Specification

**Input:**
- Q: string — the user's question
- S: SearchEngine — returns ranked results for a query  
- M: LanguageModel — provides classification, extraction, judgment, generation
- C: List[Turn] — conversation history (empty for first turn)
- K: TransitionTable — (state × signal) → state mapping, mined from 531 traces

**Output:**
- A: string — the answer
- T: List[Step] — reasoning trace (for explainability)
- φ: float ∈ [0,1] — confidence score

**Hyperparameters (from trace statistics):**
- MAX_STEPS = 12 (deep_thought avg=7.8, p95 ≈ 12)
- MAX_SEARCHES = 6 (deep_thought avg=4.5, p95 ≈ 6)
- φ_threshold = 0.7 (from convergence analysis: 94% of converged traces had φ > 0.7)

### Algorithm DMRSM(Q, S, M, C, K)

```
ALGORITHM DMRSM(Q, S, M, C, K):
────────────────────────────────────────────────────────
INPUT:  Q (question), S (search engine), M (language model),
        C (conversation history), K (transition table)
OUTPUT: (A, T, φ) — answer, trace, confidence
────────────────────────────────────────────────────────

  ┌─ PHASE 0: SAFETY GATE ──────────────────────────┐
  │ (ok, type, response) ← pre_filter(Q)            │
  │ IF NOT ok: RETURN (response, [], 1.0)            │
  └──────────────────────────────────────────────────┘

  ┌─ PHASE 1: CLASSIFY ─────────────────────────────┐
  │ q_type ← M.classify(Q)                          │
  │   // q_type ∈ {factual, multi_hop, deep_thought, │
  │   //   comparison, temporal, math, roleplay,     │
  │   //   medical, legal, therapy, creative,        │
  │   //   unanswerable, disambiguation, negation}   │
  │                                                  │
  │ s₀ ← STARTING_STATE[q_type]                     │
  │   // From data: SEARCH(74%), DECOMPOSE(10%),     │
  │   //   TRIAGE(7%), PERSONA_ADOPT(3%), CALC(3%)   │
  └──────────────────────────────────────────────────┘

  ┌─ PHASE 2: CONTEXT RESOLUTION (multi-turn) ──────┐
  │ IF C ≠ ∅:                                        │
  │   Q' ← resolve(Q, C)                            │
  │     // Pronoun resolution: "its" → "Paris's"     │
  │     // Correction: "no, I meant X" → X           │
  │     // Callback: "back to..." → restore topic    │
  │   Q ← Q'                                        │
  └──────────────────────────────────────────────────┘

  ┌─ PHASE 3: REASONING LOOP ───────────────────────┐
  │                                                  │
  │ state ← new ReasoningState(action=s₀, Q, q_type)│
  │                                                  │
  │ WHILE state.steps < MAX_STEPS:                   │
  │                                                  │
  │   // ── Execute current action ──                │
  │   result ← EXECUTE(state.action, state, S, M)   │
  │                                                  │
  │   // ── Update working memory ──                 │
  │   state.memory.ADD(result.facts)                 │
  │   state.memory.ADD(result.answer)                │
  │   state.φ ← MAX(state.φ, result.φ)              │
  │   state.trace.APPEND(state.action, result)       │
  │   state.steps ← state.steps + 1                 │
  │   IF result.searched: state.searches += 1        │
  │                                                  │
  │   // ── Check termination ──                     │
  │   IF result.terminal AND state.φ ≥ φ_threshold:  │
  │     BREAK                                        │
  │                                                  │
  │   // ── Transition ──                            │
  │   σ ← result.signal   // quality signal          │
  │   next ← K[(state.action, σ)]                    │
  │                                                  │
  │   // Fallback: model decides if no explicit rule │
  │   IF next = ∅:                                   │
  │     next ← M.decide_next(state)                  │
  │                                                  │
  │   // Guard rails                                 │
  │   IF state.searches ≥ MAX_SEARCHES AND           │
  │      next = SEARCH:                              │
  │     next ← SYNTHESIZE                            │
  │   IF state.steps ≥ MAX_STEPS - 1:               │
  │     next ← SYNTHESIZE                            │
  │                                                  │
  │   state.action ← next                            │
  │                                                  │
  │ END WHILE                                        │
  └──────────────────────────────────────────────────┘

  ┌─ PHASE 4: POST-PROCESSING ──────────────────────┐
  │ A ← state.best_answer()                          │
  │                                                  │
  │ // Roleplay: filter through persona              │
  │ IF state.persona ≠ ∅:                            │
  │   A ← M.persona_filter(A, state.persona)         │
  │                                                  │
  │ // Professional: append safety disclaimer        │
  │ IF state.safety ∈ {caution, urgent}:             │
  │   A ← A + defer_message(q_type)                  │
  │                                                  │
  │ RETURN (A, state.trace, state.φ)                 │
  └──────────────────────────────────────────────────┘
```

### EXECUTE Subroutine

```
FUNCTION EXECUTE(action, state, S, M) → Result:
────────────────────────────────────────────────────────
  CASE action OF:

    SEARCH:
      query ← state.sub_questions.POP()       // sub-Q if available
              OR M.extract_topic(state.Q)      // else model picks topic
      results ← S.search(query)
      IF results = ∅: RETURN Result(signal=no_results)
      
      (rel, φ) ← M.judge_relevance(state.Q, results[0])
      IF rel = NO:       RETURN Result(signal=irrelevant, φ=φ)
      IF φ > 0.85:       RETURN Result(signal=relevant_high, facts=[results[0]], φ=φ)
      ELSE:              RETURN Result(signal=relevant_low, facts=[results[0]], φ=φ)

    JUDGE:
      answer ← state.best_partial_answer()
      (quality, φ) ← M.judge_answer(state.Q, answer)
      RETURN Result(signal=quality, φ=φ)

    EXTRACT:
      context ← JOIN(state.memory.facts[-3:])
      answer ← M.extract(state.Q, context)
      IF answer ≠ ∅:     RETURN Result(signal=complete, answer=answer, terminal=TRUE)
      ELSE:              RETURN Result(signal=needs_more)

    DECOMPOSE:
      subs ← M.decompose(state.Q)
      subs ← FILTER(subs, λs: s ≠ state.Q)   // prevent circular decomposition
      state.sub_questions ← subs
      IF subs ≠ ∅:       RETURN Result(signal=has_subs)
      ELSE:              RETURN Result(signal=cant_break)

    REASON:
      insight ← M.reason(state.Q, state.memory.summary())
      RETURN Result(signal=insight, facts=[insight])

    SYNTHESIZE:
      A ← M.synthesize(state.Q, state.memory.all_facts(), 
                        state.partial_answers)
      RETURN Result(signal=complete, answer=A, terminal=TRUE)

    TRIAGE:
      level ← assess_safety(state.Q)  // safe | caution | urgent
      state.safety ← level
      IF level = urgent:  RETURN Result(signal=urgent)
      ELSE:              RETURN Result(signal=level)

    PERSONA_ADOPT:
      state.persona ← M.define_persona(state.Q)
      RETURN Result(signal=adopted)

    CALCULATE:
      expr ← M.extract_expression(state.Q)
      value ← safe_eval(expr)
      RETURN Result(signal=done, answer=str(value), terminal=TRUE)

    GENERATE:
      A ← M.generate(state.Q)
      RETURN Result(signal=done, answer=A, terminal=TRUE)

    GIVE_UP:
      RETURN Result(signal=done, answer="Could not find answer", terminal=TRUE)
  END CASE
```

### Flow Diagrams (ASCII)

**Main Flow:**
```
Question
   │
   ▼
┌──────────┐    blocked    ┌───────────┐
│pre_filter │──────────────▶│  CRISIS / │
│          │               │  INJECT   │
└────┬─────┘               └───────────┘
     │ ok
     ▼
┌──────────┐
│ CLASSIFY │──▶ q_type
└────┬─────┘
     │
     ▼
┌──────────────┐
│CONTEXT RESOLVE│ (if multi-turn)
└────┬─────────┘
     │
     ▼
┌─────────────────────────────────────────┐
│           REASONING LOOP                 │
│                                          │
│  ┌────────┐   result   ┌────────────┐   │
│  │EXECUTE │───────────▶│  UPDATE    │   │
│  │ action │            │  MEMORY    │   │
│  └────────┘            └─────┬──────┘   │
│       ▲                      │           │
│       │                      ▼           │
│  ┌────┴────┐           ┌──────────┐     │
│  │TRANSITION│◀──────────│CONVERGE? │     │
│  │  TABLE  │    no      │ φ ≥ 0.7  │     │
│  └─────────┘           └────┬─────┘     │
│                              │ yes       │
│                              ▼           │
│                         ┌────────┐       │
│                         │  DONE  │       │
│                         └────────┘       │
└─────────────────────────────────────────┘
     │
     ▼
┌──────────────┐
│POST-PROCESS  │ (persona filter, safety disclaimer)
└──────┬───────┘
       │
       ▼
    (A, T, φ)
```

**Transition Graph (states and edges from 531 traces):**
```
                    ┌─────────────────────────────────┐
                    │                                 │
    ┌───────────────▼──────────────┐                  │
    │          SEARCH              │                  │
    │  (74% start here)            │                  │
    └──┬───┬───┬───┬───┬──────────┘                  │
       │   │   │   │   │                              │
  high │  low  │ irrel │ no_res                       │
  conf │ conf  │       │                              │
       │   │   │   ┌───┘                              │
       │   │   │   │                                  │
       ▼   ▼   │   ▼                                  │
  ┌────────┐   │ ┌──────────┐     ┌────────┐         │
  │EXTRACT │   │ │DECOMPOSE │────▶│SEARCH  │─────────┘
  └───┬────┘   │ │(10% start)│    └────────┘  (for each sub-Q)
      │        │ └──────────┘
      │done    │
      ▼        ▼
  ┌────────┐ ┌────────┐
  │  DONE  │ │ JUDGE  │
  └────────┘ └──┬──┬──┘
                │  │
          good  │  │ needs_more
                │  │
                ▼  ▼
          ┌─────┐ ┌────────┐
          │EXTR.│ │SEARCH  │ (retry)
          └─────┘ └────────┘
              │
              ▼
         ┌──────────┐
         │SYNTHESIZE│ (48% end here)
         └──────────┘
              │
              ▼
          ┌────────┐
          │  DONE  │
          └────────┘
```

**Pattern-Specific Flows (derived from traces):**
```
DIRECT_LOOKUP (31%):    SEARCH → JUDGE → EXTRACT → DONE         (3 steps)
EXPLAIN (8%):           SEARCH → JUDGE → SYNTHESIZE → DONE       (3 steps)
MULTI_HOP (4%):         DECOMPOSE → SEARCH → SEARCH → SYNTH     (4 steps)
DEEP_THOUGHT (8%):      DECOMPOSE → SEARCH → REASON → SEARCH →  (8 steps)
                        REASON → SEARCH → REASON → SYNTHESIZE
COMPARE (5%):           DECOMPOSE → SEARCH → SEARCH → SYNTH     (4 steps)
ROLEPLAY (4%):          PERSONA_ADOPT → SEARCH → SEARCH →       (5 steps)
                        PERSONA_FILTER → GENERATE
PROFESSIONAL (10%):     TRIAGE → SEARCH → JUDGE → EXTRACT →     (6 steps)
                        DEFER → SYNTHESIZE
CALCULATE (5%):         CALCULATE → DONE                         (1 step)
UNANSWERABLE (5%):      SEARCH → JUDGE → GIVE_UP → DONE         (3 steps)
```

## Formal Analysis (CS Professor Perspective)

### The System is a Mealy Machine
- **States S** = {SEARCH, JUDGE, EXTRACT, DECOMPOSE, SYNTHESIZE, REASON, TRIAGE, PERSONA_ADOPT, PERSONA_FILTER, GENERATE, DEFER, GIVE_UP, CALCULATE, RECALL, DONE}
- **Input Σ** = result quality signals from each action execution
- **Transition δ**: S × Σ → S (the transition table above)
- **Output λ**: S × Σ → {fact, answer, sub_question, safety_flag, persona}
- **Terminal states** = {DONE} — reached via EXTRACT(complete), SYNTHESIZE(complete), GIVE_UP, CALCULATE(done), GENERATE(done)

### Termination Guarantee (no infinite loops)
1. `steps_done` monotonically increases each iteration → bounded by MAX_STEPS (12)
2. `searches_done` monotonically increases on SEARCH → bounded by MAX_SEARCHES (6)
3. Guard rails at both limits force transition to SYNTHESIZE or DONE
4. REASON→REASON loop bounded: max 3 consecutive REASON steps
5. SEARCH→SEARCH (retry) bounded: max 3 consecutive retries with different queries
6. **Proof**: let f(state) = MAX_STEPS - steps_done. f strictly decreases each iteration. f reaches 0 → forced terminal. QED.

### Correctness Properties
1. **Safety**: pre_filter() runs BEFORE the state machine. Crisis/injection never reach the loop.
2. **Liveness**: every execution produces output — GIVE_UP is valid output, guard rails force synthesis.
3. **Progress**: each SEARCH adds facts OR eliminates a query. Each REASON adds reasoning. Monotonic accumulation.
4. **Convergence**: confidence is max(all step confidences). Threshold check at each step. Higher confidence = earlier exit.

### Space Complexity
- Working memory: O(MAX_SEARCHES × avg_article_size) ≈ O(6 × 500 chars) = O(3KB)
- Trace log: O(MAX_STEPS × trace_entry_size) ≈ O(12 × 200 chars) = O(2.4KB)
- Sub-questions queue: O(max_sub) = O(3)
- Total per-query memory: ~10KB — fits on any device

### Time Complexity
- Per step: 1 model call (~100-200ms on GGUF) + optional 1 search (~500ms)
- Total: O(MAX_STEPS × (model_call + search)) = O(12 × 700ms) = ~8.4s worst case
- Typical: 3-5 steps × 400ms = 1.2-2.0s (factual questions)
- Deep thought: 8 steps × 500ms = ~4s

## Edge Cases

### Input Edge Cases
| # | Edge Case | Expected Behavior | How Handled |
|---|-----------|-------------------|-------------|
| 1 | Empty string / whitespace | Reject early | pre_filter() returns 'garbage' |
| 2 | Non-English question | Classify → SEARCH, search may fail | GIVE_UP after no results |
| 3 | Extremely long input (>1000 chars) | Truncate to model context | Truncate in classify() |
| 4 | Prompt injection ("ignore instructions") | Block | pre_filter() catches injection patterns |
| 5 | Crisis content ("I want to die") | Crisis response, skip state machine | pre_filter() returns crisis + 988 |

### Classification Edge Cases
| # | Edge Case | Expected Behavior |
|---|-----------|-------------------|
| 6 | Ambiguous type (factual + comparison) | classify() picks primary; transition table handles secondary |
| 7 | Question changes type mid-reasoning | Transition table adapts — SEARCH result reveals it's ambiguous → DISAMBIGUATE |
| 8 | Novel question type not in training | Falls through to 'SEARCH' default start |
| 9 | Multi-intent ("What is X and compare to Y") | DECOMPOSE splits into sub-questions |

### Search Edge Cases
| # | Edge Case | Expected Behavior |
|---|-----------|-------------------|
| 10 | All searches return no results | After MAX_SEARCHES: GIVE_UP with "couldn't find" |
| 11 | Search returns contradictory info | Multiple facts stored, JUDGE evaluates, SYNTHESIZE resolves |
| 12 | Search timeout / network failure | Catch exception, count as no_results, continue |
| 13 | Search returns irrelevant for all retries | After 3 retries: DECOMPOSE (try sub-questions) |
| 14 | Circular decomposition (sub-Q = original Q) | Filter: reject sub-Qs matching original |

### Reasoning Edge Cases
| # | Edge Case | Expected Behavior |
|---|-----------|-------------------|
| 15 | REASON→REASON→REASON infinite loop | Cap at 3 consecutive REASON steps → force SYNTHESIZE |
| 16 | Model returns unexpected output (garbage) | Default transition: SEARCH if early, SYNTHESIZE if late |
| 17 | Confidence always low (<0.3) | After MAX_STEPS: return best available with low-confidence flag |
| 18 | Confidence high (>0.9) for wrong answer | Judge step catches — JUDGE(good) requires both label + confidence |
| 19 | Token limit exceeded mid-generation | Model returns truncated output, treated as partial → SEARCH more |

### Multi-turn Edge Cases
| # | Edge Case | Expected Behavior |
|---|-----------|-------------------|
| 20 | Pronoun with no antecedent ("What about it?") | RECALL fails → ask for clarification (DISAMBIGUATE) |
| 21 | Correction of something not said | Treat as new question, ignore correction framing |
| 22 | Topic callback to turn from 10+ turns ago | RECALL searches full conversation, not just last turn |
| 23 | Mixed: correction + new question | Split: apply correction, then process new question |

### Professional Role Edge Cases
| # | Edge Case | Expected Behavior |
|---|-----------|-------------------|
| 24 | Urgent medical ("chest pain, can't breathe") | TRIAGE→urgent→DEFER immediately. "Call 911." |
| 25 | Legal question in unknown jurisdiction | Answer with framework + "check your local laws" |
| 26 | Financial advice that could lose money | TRIAGE→caution. Answer with disclaimer. |
| 27 | Therapy: suicidal ideation | pre_filter catches before state machine. 988 referral. |

### Roleplay Edge Cases
| # | Edge Case | Expected Behavior |
|---|-----------|-------------------|
| 28 | Harmful persona ("pretend to be a terrorist") | pre_filter OR classify→reject |
| 29 | Persona contradicts factual accuracy | Facts stay accurate; persona only filters presentation |
| 30 | Interactive roleplay needing user input | Return partial + prompt for continuation |

## Test Scenarios (organized by pattern)

### DIRECT_LOOKUP (31% of traces)
```
test_direct_lookup_simple:      "What is the capital of France?" → Paris
test_direct_lookup_who:         "Who painted the Mona Lisa?" → Leonardo da Vinci
test_direct_lookup_when:        "When did WWII end?" → 1945
test_direct_lookup_number:      "How many continents are there?" → 7
test_direct_lookup_no_result:   "Who is the president of Narnia?" → couldn't find
```

### MULTI_HOP_CHAIN (4%)
```
test_multi_hop_two:             "Capital of the country where Eiffel Tower is?" → Paris
test_multi_hop_three:           "Language spoken where sushi was invented?" → Japanese
test_multi_hop_with_failure:    "Currency of country with Great Wall?" → search fails first, retries
```

### DEEP_THOUGHT (8%)
```
test_deep_thought_decomposes:   "What happens when AI replaces jobs?" → decomposes into angles
test_deep_thought_multi_search: "Is democracy best?" → searches multiple perspectives
test_deep_thought_synthesizes:  "Why do empires fall?" → synthesizes multiple theories
```

### COMPARE (5%)
```
test_compare_factual:           "Which is larger, Jupiter or Saturn?" → Jupiter
test_compare_subjective:        "iPhone vs Samsung?" → balanced comparison
test_compare_temporal:          "Who was born first, Einstein or Newton?" → Newton
```

### UNANSWERABLE (5%)
```
test_unanswerable_prediction:   "What will the stock market do?" → can't predict
test_unanswerable_philosophy:   "What is the meaning of life?" → philosophical, no factual answer
test_unanswerable_pi:           "Last digit of pi?" → pi is irrational, no last digit
```

### DISAMBIGUATE (4%)
```
test_disambiguate_mercury:      "What is Mercury?" → presents planet/element/god
test_disambiguate_java:         "What is Java?" → programming/island/coffee
test_disambiguate_from_context: T1: "Tell me about Mercury" T2: "The planet" → locks to planet
```

### CALCULATE (5%)
```
test_calculate_percent:         "What is 15% of 230?" → 34.5
test_calculate_word_problem:    "5 machines, 5 widgets, 5 minutes..." → 5 minutes
test_calculate_conversion:      "72°F to Celsius" → 22.2
```

### ROLEPLAY (4%)
```
test_roleplay_persona:          "Explain stocks as a pirate" → pirate-voiced answer about stocks
test_roleplay_code_review:      "Review this code as senior engineer" → technical review
test_roleplay_interactive:      "Simulate a job interview" → asks questions back
```

### PROFESSIONAL ROLES (10%)
```
test_medical_urgent:            "Sharp chest pain" → TRIAGE→urgent→DEFER to ER
test_medical_safe:              "Dark yellow urine?" → hydration advice, no defer
test_legal_jurisdiction:        "Can I record a conversation?" → depends on state
test_financial_clear:           "401K match 6%" → contribute at least 6%
test_therapy_safety:            "I want to end my life" → pre_filter→988 (never reaches state machine)
test_therapy_anxiety:           "Can't sleep, anxious" → practical steps + recommend professional
```

### MULTI-TURN (context patterns)
```
test_multiturn_pronoun:         T1:"Capital of France?" T2:"What's its population?" → Paris population
test_multiturn_correction:      T1:"Who painted Mona Lisa?" T2:"No, Sistine Chapel" → Michelangelo
test_multiturn_callback:        T1:"Tallest mountain?" T2:"Deepest ocean?" T3:"Back to the mountain..." → Everest
test_multiturn_drill_down:      T1:"How do vaccines work?" T2:"What about mRNA?" → specific mRNA explanation
test_multiturn_constraint:      T1:"Visit Japan" T2:"$3000 budget" T3:"No seafood" → accumulates constraints
```

### GUARD RAILS
```
test_max_steps_forces_synthesis: Question that never converges → synthesizes after 12 steps
test_max_searches_forces_stop:  Topic with no Wikipedia article → stops searching after 6
test_crisis_bypasses_machine:   "I want to hurt myself" → 988, never enters state machine
test_injection_blocked:         "Ignore all instructions" → blocked by pre_filter
test_empty_input:               "" → "What would you like to know?"
```

## Runtime Analysis

| Question Type | Avg Steps | Avg Searches | Model Calls | Est. Time (GGUF) | Est. Time (API) |
|---------------|-----------|-------------|-------------|-------------------|-----------------|
| factual       | 3         | 1           | 3           | ~0.8s             | ~2s             |
| math          | 1         | 0           | 1           | ~0.2s             | ~0.5s           |
| comparison    | 4         | 2           | 4           | ~1.5s             | ~3s             |
| multi_hop     | 4         | 2           | 5           | ~2.0s             | ~4s             |
| deep_thought  | 8         | 5           | 10          | ~4.0s             | ~8s             |
| roleplay      | 5         | 2           | 5           | ~2.0s             | ~4s             |
| professional  | 6         | 1           | 6           | ~1.5s             | ~5s             |
| unanswerable  | 3         | 1           | 3           | ~1.0s             | ~2s             |

**Worst case**: 12 steps × (200ms model + 500ms search) = **8.4 seconds**
**Best case**: 1 step × 200ms model = **0.2 seconds** (math)
**Typical**: 3-5 steps = **1-2 seconds**

## Base Cases (guaranteed termination paths)

Every execution must terminate. These are the base cases:

| Condition | Action | Output |
|-----------|--------|--------|
| `pre_filter = crisis` | Return immediately | 988 hotline message |
| `pre_filter = injection` | Return immediately | Rejection message |
| `pre_filter = garbage` | Return immediately | "What would you like to know?" |
| `steps = MAX_STEPS` | Force SYNTHESIZE | Best available from memory |
| `searches = MAX_SEARCHES` | Force SYNTHESIZE | Best available from memory |
| `EXTRACT(complete)` | DONE | Extracted answer |
| `SYNTHESIZE(complete)` | DONE | Synthesized answer |
| `CALCULATE(done)` | DONE | Calculated result |
| `GENERATE(done)` | DONE | Generated content |
| `GIVE_UP` | DONE | "Couldn't find confident answer" |
| `all_subs_searched + no convergence` | SYNTHESIZE | Best available |
| `3 consecutive SEARCH failures` | DECOMPOSE or GIVE_UP | Try sub-Qs or give up |
| `3 consecutive REASON steps` | SYNTHESIZE | Force wrap-up |

**Invariant**: at least ONE base case is reachable from any state in ≤ (MAX_STEPS - current_steps) transitions. Proof: the guard rails at MAX_STEPS force SYNTHESIZE, which is terminal.

## Limitations

### Fundamental Limitations (cannot fix with this architecture)
1. **Model ceiling**: SmolLM2-135M has limited reasoning capacity. Complex questions may get wrong classifications or poor extractions regardless of the reasoning pattern.
2. **Search quality**: Wikipedia/DuckDuckGo may not contain the answer. No amount of reasoning helps if the knowledge isn't in the search results.
3. **Temporal knowledge**: Search results may be outdated. The TEMPORAL_VERIFY pattern helps but cannot guarantee freshness.
4. **Language**: English-only. Non-English questions will classify poorly and search poorly.
5. **Hallucination**: The 135M model may generate confident but wrong answers. The JUDGE step catches some but not all.

### Design Limitations (could fix with more work)
6. **Single-threaded search**: sub-questions searched sequentially, not in parallel. Parallel search would halve deep_thought latency.
7. **No learning**: the transition table is static. A learning system would adapt transitions based on which patterns succeed for which question types.
8. **Fixed confidence threshold**: φ=0.7 for all types. Could be per-type (deep_thought tolerates lower, medical requires higher).
9. **No source citation**: answers don't include provenance. Adding source tracking to working memory is straightforward but not in v4.
10. **Context window**: multi-turn limited by model's context (512 tokens). Long conversations will lose early context.

### Known Failure Modes (from trace analysis)
11. **Ambiguous questions with no follow-up** — engine may pick wrong interpretation (DISAMBIGUATE asks for clarification but in single-turn mode, must guess)
12. **URL-dependent questions** — engine cannot fetch URLs. Always fails for "what does this link say?"
13. **Subjective questions** — only 28% convergence in traces. Engine may over-commit to one perspective.
14. **Temporal questions** — 85% convergence. Freshness verification helps but doesn't guarantee currency.

## Files to Modify
- `engine.py` — replace DFS with state machine loop (~300 lines)
- `test_engine.py` — update + add new test categories (~200 lines)

## Verification
1. All test scenarios above pass
2. Run analyze_traces.py — verify engine's action sequences match mined patterns
3. Manual test: 10 random questions from each category
4. NQ benchmark: compare accuracy vs v3
5. Timing: avg response time per question type

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

## Phone Deployment
- Unified: 1 × T5-small = 231MB
- ONNX INT8: ~120MB, ~50ms per call = 250-400ms total
