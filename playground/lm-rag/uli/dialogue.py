"""
DialogueEngine — 5-step pipeline for conversational responses.

    1. ATTENTION  — parse question, extract key signals (QW, subject, dialogue vs factual)
    2. RULES      — apply grammar transformations (person flip your→my, QW→slot mapping)
    3. KNOWLEDGE  — look up answer in personal KB or delegate to GraphReasoner
    4. TEMPLATE   — assemble answer sentence from slot value + grammatical structure
    5. PRAGMATICS — decide whether to add a follow-up question and which type

Usage:
    from uli.dialogue import DialogueEngine
    engine = DialogueEngine(reasoner, settings)

    response = engine.respond("What is your name?")
    print(response.answer)    # "My name is ULI Reasoning Engine."
    print(response.follow_up) # "What is yours?"

    response = engine.respond("Where was Einstein born?")
    print(response.answer)    # "Einstein was born in Ulm, Germany."
    print(response.follow_up) # ""
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

# ── Person flip rules (Step 2: RULES) ────────────────────────────────────────
# Applies when the question is about the engine itself ("you/your").

_PERSON_FLIP: Dict[str, str] = {
    'your':     'my',
    'yours':    'mine',
    'you':      'I',
    'yourself': 'myself',
}

# Tokens that mark a question as being about the engine itself
_DIALOGUE_SIGNALS = frozenset({'your', 'you', 'yours', 'yourself'})

# ── Slot detection (Step 2: RULES) ────────────────────────────────────────────
# Maps question keywords → which personal KB slot is being asked about.

# Imperative task verbs — sentence-initial VERB/NOUN that signals a TASK clause
_TASK_VERBS = frozenset({
    'find', 'search', 'download', 'get', 'fetch', 'summarize', 'translate',
    'write', 'build', 'create', 'calculate', 'check', 'read', 'analyze',
    'run', 'test', 'explain', 'compare', 'list', 'show', 'give', 'tell',
    'solve', 'define', 'describe', 'generate', 'print', 'sort', 'filter',
    'compile', 'execute', 'install', 'deploy', 'send', 'upload', 'convert',
})

_CONTEXT_VERBS = frozenset({
    'think', 'believe', 'feel', 'wonder', 'consider', 'plan',
    'hope', 'wish', 'want', 'imagine', 'suppose', 'guess',
})

_KEYWORD_TO_SLOT: Dict[str, str] = {
    'name':     'name',
    'called':   'name',
    'version':  'version',
    'do':       'capabilities',
    'can':      'capabilities',
    'capable':  'capabilities',
    'help':     'capabilities',
    'able':     'capabilities',
    'know':     'capabilities',
}

# ── Answer templates (Step 4: TEMPLATE) ──────────────────────────────────────
# One template per personal KB slot. {value} is filled with the KB lookup result.

_DIALOGUE_TEMPLATES: Dict[str, str] = {
    'name':         'My name is {value}.',
    'type':         'I am {value}.',
    'version':      'I am version {value}.',
    'capabilities': 'I can {value}.',
    'default':      'I am {value}.',
}

# ── Pragmatics rules (Step 5: PRAGMATICS) ────────────────────────────────────
# slot → (follow_up_when_known, follow_up_when_unknown)

_FOLLOW_UPS: Dict[str, Tuple[str, str]] = {
    'name':         ("What is yours?",                       ""),
    'type':         ("What can I help you with today?",      ""),
    'version':      ("",                                     ""),
    'capabilities': ("Would you like me to try something?",  ""),
}

_FACTUAL_UNKNOWN_FOLLOW_UP = "Could you rephrase that, or try asking differently?"


# ── Response dataclass ────────────────────────────────────────────────────────

@dataclass
class DialogueResponse:
    """Result of DialogueEngine.respond()."""
    answer:    str
    follow_up: str  = ''
    source:    str  = 'kg'         # 'personal_kb' | 'kg' | 'web' | 'unknown'
    slot:      str  = ''           # personal KB slot that was filled ('' = factual)


# ── DialogueEngine ────────────────────────────────────────────────────────────

class DialogueEngine:
    """
    5-step conversational pipeline that wraps GraphReasoner.

    Step 1  ATTENTION  — detect whether question is personal (about the engine)
                         or factual (about the world). Extract question word + keywords.
    Step 2  RULES      — apply person flip (your→my, you→I).
                         Map keywords to personal KB slot name.
    Step 3  KNOWLEDGE  — personal questions → personal KB lookup.
                         factual questions → GraphReasoner.try_answer().
    Step 4  TEMPLATE   — pick template for the slot and fill it.
                         factual answers are already complete sentences.
    Step 5  PRAGMATICS — append follow-up question when appropriate.
                         Personal/name → reciprocal. Factual unknown → rephrase hint.
    """

    def __init__(self, reasoner, settings=None, thinker=None):
        """
        Args:
            reasoner: GraphReasoner instance (already initialised).
            settings: SystemSettings from SystemConfig.load().
                      Used to populate the personal KB (name, capabilities, etc.).
                      Falls back to built-in defaults if None.
            thinker:  Thinker instance (optional). When GraphReasoner returns
                      nothing, falls through to thinker.reason() for DMRSM
                      state-machine reasoning.
        """
        self._reasoner   = reasoner
        self._thinker    = thinker
        self._personal_kb = self._build_personal_kb(settings)

    # ── Personal KB builder ───────────────────────────────────────────────────

    def _build_personal_kb(self, settings) -> Dict[str, str]:
        """Populate personal KB from SystemSettings or hardcoded defaults."""
        if settings is None:
            return {
                'name':         'ULI',
                'type':         ('a reasoning engine that answers questions from '
                                 'a knowledge graph and the web'),
                'version':      '1.0',
                'capabilities': ('answer factual questions, search the web, '
                                 'reason across facts, generate explanations, '
                                 'and assist with code'),
            }

        # Build capabilities string from the list in config
        cap_list = getattr(settings, 'capabilities', []) or []
        cap_str  = ', '.join(c.replace('_', ' ').replace('qa', 'QA') for c in cap_list)

        return {
            'name':         getattr(settings, 'name', 'ULI'),
            'type':         ('a reasoning engine that answers from a knowledge '
                             'graph and verified external sources'),
            'version':      getattr(settings, 'version', '1.0'),
            'capabilities': cap_str or 'answer questions and reason about facts',
        }

    # ── Step 1: ATTENTION ─────────────────────────────────────────────────────

    def _attention(self, question: str) -> dict:
        """
        Extract key signals from the question without running a full parse.
        Returns a signals dict consumed by the remaining steps.

        Handles:
        - Dialogue vs factual detection (your/you signals)
        - Imperative verbs at sentence start (tagged as NOUN by POS tagger)
        - Embedded-verb override: "Check if it is..." → verb='check', not 'is'
        """
        from uli import tag as uli_tag

        q_lower  = question.lower().strip().rstrip('?!.').strip()
        words    = q_lower.split()
        word_set = set(words)

        # Detect imperative: first token lemma is a task verb (even if tagged NOUN)
        tagged_first = uli_tag(words[0]) if words else []
        first_lemma  = tagged_first[0][2].lower() if tagged_first else ''
        is_imperative = first_lemma in _TASK_VERBS

        # Detect negation: explicit "do not" / "don't" / "never" at start
        is_negation = (
            q_lower.startswith('do not') or
            q_lower.startswith("don't") or
            words[0] == 'never'
        )

        # "Can you tell me X" / "Could you help me with X" = polite request,
        # NOT a capability question. Detect and reroute as imperative task.
        _POLITE_REQUEST = (
            'can you tell', 'could you tell', 'can you help',
            'could you help', 'can you show', 'could you show',
            'can you find', 'could you find', 'can you get',
            'can you check', 'could you check', 'can you explain',
            'would you tell', 'will you tell',
        )
        is_polite_request = any(q_lower.startswith(p) for p in _POLITE_REQUEST)

        if is_polite_request:
            # Extract the actual query after the polite prefix
            # "can you tell me what is the weather" → task verb = 'tell'
            is_imperative = True
            # Find the task verb (tell/help/show/find/check/explain)
            first_lemma = words[2] if len(words) > 2 else first_lemma

        # Only flag capability if "you" is present AND it's not a polite request
        has_cap = (bool({'do', 'can', 'help', 'capable', 'able', 'know'} & word_set)
                   and not is_polite_request)

        return {
            'original':       question,
            'words':          words,
            'word_set':       word_set,
            'is_dialogue':    bool(_DIALOGUE_SIGNALS & word_set) and not is_imperative,
            'is_imperative':  is_imperative,
            'is_negation':    is_negation,
            'first_lemma':    first_lemma,
            'question_word':  words[0] if words else '',
            'has_name':       bool({'name', 'called'} & word_set),
            'has_capability': has_cap,
            'has_version':    'version' in word_set,
        }

    # ── Step 2: RULES ─────────────────────────────────────────────────────────

    def _apply_rules(self, signals: dict) -> Tuple[str, str]:
        """
        Apply grammar rules to determine what to look up and how to phrase the answer.

        Returns:
            slot (str): personal KB key, or '' for factual questions.
            transformed (str): question with person flip applied (for tracing).

        Person flip rule:
            "What is your name?" → "What is my name?" → slot='name'
            The flip is: your→my, you→I, etc. (Farlex: deixis shift at dialogue boundary)
        """
        if not signals['is_dialogue']:
            return '', signals['original']

        # Apply person flip to each token
        original_words  = signals['original'].split()
        transformed_words = [
            _PERSON_FLIP.get(w.lower(), w) for w in original_words
        ]
        transformed = ' '.join(transformed_words)

        # Detect which slot is being asked about
        word_set = signals['word_set']
        slot = ''

        if signals['has_name'] or signals['question_word'] == 'who':
            slot = 'name'
        elif signals['has_version']:
            slot = 'version'
        elif signals['has_capability']:
            slot = 'capabilities'
        elif signals['question_word'] == 'what':
            slot = 'type'   # "What are you?" → type description
        else:
            slot = 'type'   # default for any "you" question not caught above

        return slot, transformed

    # ── Task helpers ──────────────────────────────────────────────────────────

    # Verbs that need attached content (can't execute without it)
    _CONTENT_NEEDED = frozenset({'summarize', 'translate', 'rewrite', 'paraphrase'})

    # Verbs that convert naturally to a "what is" question
    _EXPLAIN_VERBS  = frozenset({'explain', 'describe', 'define', 'clarify'})

    # Verbs that mean "find / retrieve"
    _SEARCH_VERBS   = frozenset({'find', 'search', 'get', 'fetch', 'lookup',
                                  'check', 'list', 'show', 'compare'})

    def _extract_task_query(self, question: str, task_verb: str) -> str:
        """
        Strip the imperative verb (and common filler) from the sentence
        to get the core query for KNOWLEDGE lookup.

        Examples:
            "Explain the difference between TCP and UDP" → "difference between TCP and UDP"
            "Search for recent news about climate change" → "recent news about climate change"
            "Find the capital of France" → "capital of France"
            "List 10 countries by GDP" → "countries by GDP"
        """
        words = question.split()
        if not words:
            return question

        # Strip polite prefix: "Can you tell me", "Could you show me", etc.
        _POLITE_STARTERS = {'can', 'could', 'would', 'will'}
        if len(words) >= 3 and words[0].lower() in _POLITE_STARTERS and words[1].lower() == 'you':
            words = words[2:]  # drop "can you" / "could you"

        # Drop first word if it's the task verb (case-insensitive)
        if words and words[0].lower() == task_verb:
            words = words[1:]

        # Drop leading filler: "for", "me", "out", "the", "all", "up", "a", "an"
        _FILLER = {'for', 'me', 'out', 'the', 'all', 'up', 'a', 'an', 'about',
                    'what', 'is'}
        while words and words[0].lower() in _FILLER:
            words = words[1:]

        result = ' '.join(words) if words else question
        return result.rstrip('?!.')

    def _handle_task(self, signals: dict, lang: str) -> Tuple[str, str]:
        """
        Route a TASK (imperative) to the right handler.
        Returns (answer_text, source).

        Routing:
          summarize/translate  → need content → prompt user
          explain/describe     → convert to "What is X?" → reasoner
          find/search/list/... → use object as query → reasoner
          write/build/create   → code or creative → reasoner (web fallback)
          calculate/solve      → math → try compute, then reasoner
        """
        verb  = signals['first_lemma']
        query = self._extract_task_query(signals['original'], verb)

        if not query:
            return "What would you like me to " + verb + "?", 'unknown'

        # Verbs needing attached content
        if verb in self._CONTENT_NEEDED:
            return (
                f"Sure, I can {verb} that. Please share the text you'd like me to work on.",
                'prompt'
            )

        # Explain/describe → reframe as a question
        if verb in self._EXPLAIN_VERBS:
            question_form = f"What is {query}?"
            result = self._reasoner.try_answer(question_form, lang=lang)
            if result:
                return result[0], 'kg'
            result = self._reasoner.try_answer(query, lang=lang)
            if result:
                return result[0], 'web'
            return f"I don't have enough information about {query} right now.", 'unknown'

        # Search/find/list/compare → pass object as query directly
        result = self._reasoner.try_answer(query, lang=lang)
        if result:
            src = 'web' if (
                result[0].startswith('[Wikipedia') or result[0].startswith('[1]')
            ) else 'kg'
            return result[0], src

        return f"I couldn't find information about {query}.", 'unknown'

    def _handle_constraint(self, signals: dict) -> Tuple[str, str]:
        """Acknowledge a constraint. Does not execute anything."""
        original = signals['original']
        # Strip the negation prefix for a clean echo
        for prefix in ("do not ", "don't ", "never "):
            if original.lower().startswith(prefix):
                constraint_body = original[len(prefix):]
                return f"Understood — I won't {constraint_body.lower().rstrip('.')}.", 'constraint'
        return "Understood, I'll keep that in mind.", 'constraint'

    # ── Step 3: KNOWLEDGE ─────────────────────────────────────────────────────

    def _get_answer(self, slot: str, question: str,
                    lang: str) -> Tuple[str, str]:
        """
        Look up the answer value.

        For dialogue questions (slot != ''): query personal KB.
        For factual questions (slot == ''):  delegate to GraphReasoner.

        Returns:
            (value, source) where source is 'personal_kb' | 'kg' | 'web' | 'unknown'.
        """
        if slot:
            value = self._personal_kb.get(slot, '')
            return (value, 'personal_kb') if value else ('', 'unknown')

        # Factual: delegate to the full reasoning pipeline
        result = self._reasoner.try_answer(question, lang=lang)
        if result:
            answer_text = result[0]
            # Detect whether the reasoner used web search or KG
            source = 'web' if (
                answer_text.startswith('[Wikipedia') or
                answer_text.startswith('[1]') or
                'http' in answer_text[:30]
            ) else 'kg'
            return answer_text, source

        # Fallback: DMRSM reasoning via Thinker
        if self._thinker:
            answer, trace, confidence = self._thinker.reason(question, lang=lang)
            if answer and confidence > 0.3:
                return answer, 'thinker'

        return '', 'unknown'

    # ── Step 4: TEMPLATE ──────────────────────────────────────────────────────

    def _fill_template(self, slot: str, value: str) -> str:
        """
        Assemble the answer sentence.

        Dialogue questions: fill the slot-specific template.
        Factual questions:  the value from reasoner is already a sentence.
        """
        if not value:
            return "I don't have that information right now."

        if slot:
            template = _DIALOGUE_TEMPLATES.get(slot, _DIALOGUE_TEMPLATES['default'])
            return template.format(value=value)

        # Factual: reasoner already returns a complete sentence
        return value

    # ── Step 5: PRAGMATICS ────────────────────────────────────────────────────

    def _pragmatics(self, slot: str, source: str, value: str) -> str:
        """
        Decide whether to add a follow-up question.

        Dialogue questions:
            name  → reciprocal "What is yours?"
            type  → open-ended "What can I help you with?"
            else  → no follow-up

        Factual questions:
            known   → no follow-up (factual answers are terminal)
            unknown → suggest rephrasing
        """
        if slot:
            known_fu, unknown_fu = _FOLLOW_UPS.get(slot, ('', ''))
            return known_fu if value else unknown_fu

        # Factual
        if source == 'unknown':
            return _FACTUAL_UNKNOWN_FOLLOW_UP
        return ''

    # ── Main entry point ──────────────────────────────────────────────────────

    def respond(self, question: str, lang: str = 'en') -> DialogueResponse:
        """
        Run the full 5-step pipeline for one turn.

        Routing (decided in ATTENTION + RULES):
            CONSTRAINT  → acknowledge, don't execute
            TASK        → extract object, delegate to reasoner/web
            DIALOGUE    → personal KB lookup + person-flip template
            FACTUAL     → GraphReasoner (KG → web fallback)

        Args:
            question: raw user question string
            lang:     ISO 639-1 language code (default: 'en')

        Returns:
            DialogueResponse with answer + optional follow_up
        """
        # ── 1. ATTENTION ──────────────────────────────────────────────────────
        signals = self._attention(question)

        # ── 1b. INTENT — let the classifier decide before rule-matching ───────
        try:
            from uli.router import classify
            intent = classify(question)
            intent_val = intent.value if intent else 'factual'
        except Exception:
            intent_val = 'factual'

        # If intent is non-conversational, skip personal KB entirely —
        # route as task/factual even if "can"/"you" appear in the sentence.
        if intent_val != 'conversation' and signals['is_dialogue']:
            # "Can you tell me the weather?" → intent=factual, has "you"
            # Override: treat as imperative task, not dialogue about self
            if signals['has_capability'] or signals['is_imperative']:
                signals['is_imperative'] = True
                signals['is_dialogue'] = False
                signals['has_capability'] = False

        # ── 2. RULES — route by clause type ───────────────────────────────────
        if signals['is_negation']:
            # CONSTRAINT: acknowledge and stop
            answer, source = self._handle_constraint(signals)
            return DialogueResponse(answer=answer, source=source, slot='constraint')

        if signals['is_imperative']:
            # TASK: extract object → reasoner/web
            answer, source = self._handle_task(signals, lang)
            follow_up = '' if source != 'unknown' else _FACTUAL_UNKNOWN_FOLLOW_UP
            return DialogueResponse(answer=answer, follow_up=follow_up,
                                    source=source, slot='task')

        # Dialogue or factual — original path
        slot, transformed = self._apply_rules(signals)

        # ── 3. KNOWLEDGE ─────────────────────────────────────────────────────
        value, source = self._get_answer(slot, question, lang)

        # ── 4. TEMPLATE ──────────────────────────────────────────────────────
        answer = self._fill_template(slot, value)

        # ── 5. PRAGMATICS ────────────────────────────────────────────────────
        follow_up = self._pragmatics(slot, source, value)

        return DialogueResponse(
            answer=answer,
            follow_up=follow_up,
            source=source,
            slot=slot,
        )
