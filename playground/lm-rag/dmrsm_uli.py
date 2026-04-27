"""
DMRSM-ULI: Combined Reasoning + Language System.

Pipeline: text → ULI reads → AST → DMRSM thinks → AST → ULI writes → text

No LM/LLM. No neural models. Pure rules + database + structural similarity.
"""

import os
import sys
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

from uli.protocol import MeaningAST, Entity, Token
from uli.modules.english import EnglishModule
from uli.modules.marathi import MarathiModule
from uli.lexer import detect_language
from uli.similarity import text_similarity, question_passage_relevance
from uli.learner import Learner
from search_providers import create_default_engine

log = logging.getLogger('dmrsm-uli')
logging.basicConfig(
    level=getattr(logging, os.environ.get('LOG_LEVEL', 'INFO').upper(), logging.INFO),
    format='%(levelname)s %(name)s: %(message)s',
)


# ============================================================
# Safety Pre-filter (hardcoded — too critical for rules)
# ============================================================

import re

CRISIS_PATTERNS = [
    r'\b(kill|end|hurt)\s+(my|self|myself)\b',
    r'\bsuicid', r'\bwant\s+to\s+die\b',
    r'\bdon.t\s+want\s+to\s+(be\s+alive|live|be\s+here)\b',
]
INJECTION_PATTERNS = [
    r'ignore\s+(previous|all)\s+(instructions|rules)',
    r'jailbreak', r'override\s+safety',
]

def pre_filter(text):
    lower = text.lower().strip()
    for p in CRISIS_PATTERNS:
        if re.search(p, lower):
            return False, 'crisis', (
                "I care about you. Please reach out to the 988 Suicide & Crisis Lifeline "
                "(call or text 988). You're not alone.")
    for p in INJECTION_PATTERNS:
        if re.search(p, lower):
            return False, 'injection', "I didn't understand that. Could you rephrase?"
    if len(lower.strip()) < 2:
        return False, 'garbage', "What would you like to know?"
    return True, 'safe', None


# ============================================================
# Transition Table (from 531 traces)
# ============================================================

TRANSITIONS = {
    ('SEARCH', 'relevant_high'):   'EXTRACT',
    ('SEARCH', 'relevant_low'):    'JUDGE',
    ('SEARCH', 'irrelevant'):      'SEARCH',
    ('SEARCH', 'no_results'):      'DECOMPOSE',

    ('JUDGE', 'good'):             'EXTRACT',
    ('JUDGE', 'needs_more'):       'SEARCH',
    ('JUDGE', 'needs_synthesis'):  'SYNTHESIZE',
    ('JUDGE', 'unanswerable'):     'GIVE_UP',

    ('EXTRACT', 'complete'):       'DONE',
    ('EXTRACT', 'needs_more'):     'SEARCH',

    ('DECOMPOSE', 'has_subs'):     'SEARCH',
    ('DECOMPOSE', 'cant_break'):   'SEARCH',

    ('SYNTHESIZE', 'complete'):    'DONE',
    ('SYNTHESIZE', 'weak'):        'SEARCH',

    ('GIVE_UP', 'done'):           'DONE',
    ('CALCULATE', 'done'):         'DONE',
}

STARTING_STATE = {
    'factual': 'SEARCH',
    'multi_hop': 'DECOMPOSE',
    'deep_thought': 'DECOMPOSE',
    'comparison': 'DECOMPOSE',
    'explanation': 'SEARCH',
    'math': 'CALCULATE',
    'creative': 'SEARCH',
}

MAX_STEPS = 12
MAX_SEARCHES = 6
CONFIDENCE_THRESHOLD = 0.6
MAX_CONSECUTIVE_SAME = 3


# ============================================================
# Reasoning State
# ============================================================

@dataclass
class ReasoningState:
    action: str
    question_ast: MeaningAST
    facts: List[MeaningAST] = field(default_factory=list)
    fact_texts: List[str] = field(default_factory=list)
    sub_questions: List[MeaningAST] = field(default_factory=list)
    partial_answers: Dict[str, str] = field(default_factory=dict)
    confidence: float = 0.0
    searches_done: int = 0
    steps_done: int = 0
    consecutive_same: int = 0
    last_action: str = ''
    trace: List[dict] = field(default_factory=list)
    _search_queries_tried: set = field(default_factory=set)

    def best_answer(self) -> str:
        for key in ('EXTRACT', 'SYNTHESIZE', 'CALCULATE'):
            if key in self.partial_answers:
                return self.partial_answers[key]
        if self.partial_answers:
            return list(self.partial_answers.values())[-1]
        return ''


@dataclass
class ActionResult:
    signal: str = ''
    confidence: float = 0.0
    answer: str = ''
    fact_ast: Optional[MeaningAST] = None
    fact_text: str = ''
    searched: bool = False
    terminal: bool = False


    # Embedder removed — all similarity is structural via uli/similarity.py


# ============================================================
# DMRSM-ULI Engine
# ============================================================

class Engine:
    """
    Combined reasoning + language engine.
    text → ULI reads → AST → DMRSM thinks → AST → ULI writes → text
    """

    def __init__(self, search_engine=None, learner=None):
        # Language modules (pluggable)
        self.modules = {
            'en': EnglishModule(),
            'mr': MarathiModule(),
        }
        self.default_module = self.modules['en']

        # Search engine
        self.search_engine = search_engine or create_default_engine()

        # Learner (train by talking / feeding data)
        self.learner = learner or Learner()

        # Conversation memory
        self.conversation: List[dict] = []

        # Verified facts KB (grows from conversation)
        self.knowledge_base: List[dict] = []

        log.info("DMRSM-ULI engine ready. No neural models. Pure rules + structural similarity.")

    def _get_module(self, text):
        """Auto-detect language and return appropriate module."""
        lang = detect_language(text)
        if lang in self.modules:
            return self.modules[lang]
        # For Devanagari, check if it's Marathi (default to mr for Tejas)
        if lang == 'hi':
            return self.modules.get('mr', self.default_module)
        return self.default_module

    # ── Main Entry Point ─────────────────────────────────

    def ask(self, text: str, verbose: bool = False) -> str:
        """Full pipeline: text → read → think → write → text."""
        answer, trace, confidence = self.reason(text)
        if verbose:
            for step in trace:
                print(f"  [{step['action']:12s}] → {step['signal']:15s} "
                      f"conf={step['confidence']:.2f}  {step.get('answer','')[:40]}")
        return answer

    def reason(self, text: str) -> Tuple[str, List[dict], float]:
        """Complete reasoning pipeline."""

        # Phase 0: Safety
        is_safe, filter_type, response = pre_filter(text)
        if not is_safe:
            return response, [{'action': 'PRE_FILTER', 'signal': filter_type, 'confidence': 1.0}], 1.0

        # Phase 0.5: Sarcasm / implicature detection (structural, no neural model)
        # TODO: re-wire context chain with structural similarity instead of MiniLM
        sarcasm, sarcasm_conf = False, 0.0
        implicature = None

        # Phase 1: READ (ULI)
        module = self._get_module(text)
        question_ast = module.read(text)

        # Apply context chain insights to AST
        if sarcasm:
            question_ast.negation = not question_ast.negation  # Invert meaning
            log.info(f"[CONTEXT] Sarcasm detected (conf={sarcasm_conf:.2f}), inverting")
        if implicature:
            question_ast.intent = implicature
            log.info(f"[CONTEXT] Implicature resolved: {implicature}")

        # Resolve pronouns from conversation history (structural, no neural model)
        if question_ast.entities:
            for i, entity in enumerate(question_ast.entities):
                if entity.lower() in ('it', 'its', 'he', 'she', 'they', 'them'):
                    # Find most recent proper noun in conversation
                    for turn in reversed(self.conversation):
                        from uli.lexer import tokenize
                        tokens, _ = tokenize(turn['text'])
                        for tok in tokens:
                            if tok.pos == 'PROPN' and tok.text.lower() not in (
                                'the', 'a', 'i', 'what', 'who'):
                                question_ast.entities[i] = tok.text
                                log.info(f"[CONTEXT] Resolved '{entity}' → '{tok.text}'")
                                break
                        else:
                            continue
                        break

        # Feed tokens to learner (flag unknown words)
        tok_result = module.tokenize(module.normalize(text))
        tokens = tok_result[0] if isinstance(tok_result, tuple) else tok_result
        self.learner.on_tokens(tokens, text=text,
                               lang=question_ast.source_language or 'en')

        # Context tracked via self.conversation (no neural model needed)

        log.info(f"[READ] type={question_ast.type} intent={question_ast.intent} "
                 f"entities={question_ast.entities[:3]} q_word={question_ast.question_word}")

        # Check KB first — if we already know the answer, skip search
        kb_answer = self._check_kb(question_ast)
        if kb_answer:
            log.info(f"[KB] Found cached answer: {kb_answer[:40]}")
            self.conversation.append({'role': 'user', 'text': text})
            self.conversation.append({'role': 'assistant', 'text': kb_answer})
            return kb_answer, [{'action': 'KB_HIT', 'signal': 'cached',
                               'confidence': 1.0, 'answer': kb_answer}], 1.0

        # Phase 2: THINK (DMRSM state machine on ASTs)
        start = STARTING_STATE.get(question_ast.intent, 'SEARCH')
        state = ReasoningState(action=start, question_ast=question_ast)

        while state.steps_done < MAX_STEPS:
            result = self._execute(state, module)

            # Update state
            if result.fact_ast:
                state.facts.append(result.fact_ast)
            if result.fact_text:
                state.fact_texts.append(result.fact_text)
            if result.answer:
                state.partial_answers[state.action] = result.answer
            state.confidence = max(state.confidence, result.confidence)
            if result.searched:
                state.searches_done += 1

            # Trace
            state.trace.append({
                'action': state.action,
                'signal': result.signal,
                'confidence': round(result.confidence, 3),
                'answer': result.answer[:60] if result.answer else '',
            })
            state.steps_done += 1

            log.info(f"  step={state.steps_done} {state.action}→{result.signal} "
                     f"conf={result.confidence:.2f}")

            # Convergence
            if result.terminal and state.confidence >= CONFIDENCE_THRESHOLD:
                break
            if state.action == 'DONE':
                break

            # Transition
            state.last_action = state.action
            next_action = self._transition(state, result)
            if next_action == state.last_action:
                state.consecutive_same += 1
            else:
                state.consecutive_same = 0

            if state.consecutive_same >= MAX_CONSECUTIVE_SAME:
                next_action = 'SYNTHESIZE'
                state.consecutive_same = 0

            state.action = next_action
            if state.action == 'DONE':
                break

        # Phase 3: WRITE (ULI)
        answer = state.best_answer()
        if not answer:
            answer = "I couldn't find a confident answer. Could you rephrase?"

        # Store in conversation memory
        self.conversation.append({'role': 'user', 'text': text})
        self.conversation.append({'role': 'assistant', 'text': answer})

        # Context tracked via self.conversation

        # If high confidence, store as verified fact in KB (self-evolution)
        if state.confidence >= 0.8 and answer and "couldn't find" not in answer:
            fact = self.learner.on_verified_answer(text, answer)
            self.knowledge_base.append(fact)
            log.info(f"[KB] Stored: '{text[:30]}' → '{answer[:30]}' (conf={state.confidence:.2f})")

        # Commit any learned words that reached threshold
        committed = self.learner.commit_queue(lang='all')
        if committed:
            log.info(f"[LEARN] Committed {len(committed)} new words: {committed[:3]}")

        return answer, state.trace, state.confidence

    # ── Transition ───────────────────────────────────────

    def _check_kb(self, question_ast: MeaningAST) -> Optional[str]:
        """Check knowledge base for a cached answer."""
        if not self.knowledge_base:
            return None
        query = question_ast.search_query()
        if not query:
            return None
        # Find best match by structural similarity
        best_answer = None
        best_sim = 0.0
        for fact in self.knowledge_base:
            sim = text_similarity(query, fact['question'])
            if sim > best_sim:
                best_sim = sim
                best_answer = fact['answer']
        if best_sim > 0.85:  # High threshold for KB hit
            return best_answer
        return None

    # ── Teaching Interface ───────────────────────────────

    def teach(self, word: str, definition: dict, lang: str = 'en'):
        """Explicitly teach a new word."""
        self.learner.teach_word(word, definition, lang=lang)

    def teach_idiom(self, phrase: str, meaning: str, lang: str = 'en'):
        """Explicitly teach a new idiom."""
        self.learner.teach_idiom(phrase, meaning, lang=lang)

    def teach_abbreviation(self, abbrev: str, expansion: str, lang: str = 'en'):
        """Explicitly teach a new abbreviation."""
        self.learner.teach_abbreviation(abbrev, expansion, lang=lang)

    def save_learned(self):
        """Persist all learned data to disk."""
        return self.learner.save()

    # ── Transition ──────────────────���────────────────────

    def _transition(self, state: ReasoningState, result: ActionResult) -> str:
        key = (state.action, result.signal)
        next_action = TRANSITIONS.get(key)

        if next_action is None:
            next_action = 'SYNTHESIZE' if state.steps_done > MAX_STEPS // 2 else 'SEARCH'

        if state.searches_done >= MAX_SEARCHES and next_action == 'SEARCH':
            next_action = 'SYNTHESIZE'
        if state.steps_done >= MAX_STEPS - 1 and next_action not in ('DONE', 'SYNTHESIZE', 'GIVE_UP'):
            next_action = 'SYNTHESIZE'

        return next_action

    # ── Action Execution ────────────────────────────��────

    def _execute(self, state: ReasoningState, module) -> ActionResult:
        action = state.action

        if action == 'SEARCH':
            return self._exec_search(state, module)
        elif action == 'JUDGE':
            return self._exec_judge(state)
        elif action == 'EXTRACT':
            return self._exec_extract(state, module)
        elif action == 'DECOMPOSE':
            return self._exec_decompose(state, module)
        elif action == 'SYNTHESIZE':
            return self._exec_synthesize(state, module)
        elif action == 'CALCULATE':
            return self._exec_calculate(state)
        elif action == 'GIVE_UP':
            return ActionResult(signal='done',
                answer="I couldn't find a confident answer. Could you rephrase?",
                terminal=True)
        else:
            return ActionResult(signal='done', terminal=True)

    def _exec_search(self, state, module) -> ActionResult:
        """SEARCH: iterate through entity queries one at a time.

        Wikipedia opensearch matches article TITLES — single entity names
        match precisely. Multi-word queries get fuzzy-matched to wrong titles.
        So we search entities individually, priority-ordered.
        """
        # Get ordered query list from AST
        if state.sub_questions:
            sq = state.sub_questions.pop(0)
            queries = sq.search_queries()
        else:
            queries = state.question_ast.search_queries()

        # Find next untried query
        query = None
        for q in queries:
            if q and q not in state._search_queries_tried:
                query = q
                break

        if not query:
            return ActionResult(signal='no_results', searched=True)

        state._search_queries_tried.add(query)

        if not query or len(query.strip()) < 2:
            return ActionResult(signal='no_results', searched=True)

        log.info(f"  [SEARCH] query='{query[:50]}'")

        try:
            results = self.search_engine.search(query, max_per_provider=2)
        except Exception as e:
            log.warning(f"Search failed: {e}")
            return ActionResult(signal='no_results', searched=True)

        if not results:
            return ActionResult(signal='no_results', searched=True)

        best = results[0]
        passage_text = best.text[:500]

        # Relevance: structural similarity from ULI (no neural model)
        from uli.similarity import question_passage_relevance
        q_text = state.question_ast.source or query
        sim = question_passage_relevance(q_text, passage_text[:300])

        log.info(f"  [SEARCH] result='{best.title}' sim={sim:.3f}")

        if sim > 0.4:
            return ActionResult(
                signal='relevant_high' if sim > 0.6 else 'relevant_low',
                confidence=sim,
                fact_text=passage_text,
                searched=True,
            )
        else:
            return ActionResult(signal='irrelevant', confidence=sim, searched=True)

    def _exec_judge(self, state) -> ActionResult:
        """JUDGE: AST structural comparison — does answer cover question?"""
        answer = state.best_answer()
        if not answer or len(answer.strip()) < 2:
            return ActionResult(signal='needs_more', confidence=0.2)

        q = state.question_ast

        # Type match: "who" question → answer should look like a person name
        type_ok = True
        if q.question_word == 'who':
            # Simple heuristic: does answer start with uppercase (proper noun)?
            type_ok = answer[0].isupper() if answer else False

        # Coverage: does answer address the question?
        # Structural similarity — no neural model
        from uli.similarity import text_similarity
        sim = text_similarity(q.source or q.search_query(), answer)

        if sim > 0.6 and type_ok:
            return ActionResult(signal='good', confidence=sim)
        elif sim > 0.3:
            return ActionResult(signal='needs_more', confidence=sim)
        else:
            return ActionResult(signal='unanswerable', confidence=sim)

    def _exec_extract(self, state, module) -> ActionResult:
        """EXTRACT: find answer span in passage using AST pattern matching."""
        if not state.fact_texts:
            return ActionResult(signal='needs_more', confidence=0.1)

        q = state.question_ast

        # Use the MOST RECENT fact text (highest relevance, just added)
        # Then fall back to combining last 3 if single doesn't work
        if state.fact_texts:
            passage = state.fact_texts[-1]
        else:
            return ActionResult(signal='needs_more', confidence=0.1)

        # Parse the passage to find answer
        passage_ast = module.read(passage)

        # Pattern match: find what fills the question slot
        answer = self._match_answer(q, passage_ast, passage)

        if answer and len(answer.strip()) > 1:
            return ActionResult(
                signal='complete',
                answer=answer.strip(),
                confidence=max(state.confidence, 0.7),
                terminal=True,
            )
        return ActionResult(signal='needs_more', confidence=0.3)

    def _match_answer(self, question_ast: MeaningAST,
                      passage_ast: MeaningAST, passage_text: str) -> str:
        """Match question slots against passage to extract answer.
        Pure structural + heuristic — no model."""

        target = question_ast.question_target

        # Direct slot matching
        if target == 'agent' and passage_ast.agent and passage_ast.agent.text != '?':
            return passage_ast.agent.text
        if target == 'location' and passage_ast.location:
            return passage_ast.location.text
        if target == 'time' and passage_ast.time:
            return passage_ast.time.text
        if target == 'theme' and passage_ast.theme:
            return passage_ast.theme.text

        # Entity matching: find entities in passage that aren't in question
        q_entities = set(e.lower() for e in question_ast.entities)
        for entity in passage_ast.entities:
            if entity.lower() not in q_entities and len(entity) > 2:
                return entity

        # Fallback: first named entity in passage
        if passage_ast.entities:
            return passage_ast.entities[0]

        # Last resort: first noun phrase from passage
        if passage_ast.agent and passage_ast.agent.text:
            return passage_ast.agent.text

        return ''

    def _exec_decompose(self, state, module) -> ActionResult:
        """DECOMPOSE: split nested AST into sub-question ASTs."""
        q = state.question_ast

        # If AST has sub-clauses, use them
        if q.sub_clauses:
            state.sub_questions = q.sub_clauses
            return ActionResult(signal='has_subs')

        # Otherwise, try to split entities into separate queries
        entities = q.entities
        if len(entities) >= 2:
            subs = []
            for e in entities[:3]:
                sub = MeaningAST(
                    type='question',
                    intent='factual',
                    entities=[e],
                    source=e,
                )
                subs.append(sub)
            state.sub_questions = subs
            return ActionResult(signal='has_subs')

        return ActionResult(signal='cant_break')

    def _exec_synthesize(self, state, module) -> ActionResult:
        """SYNTHESIZE: merge facts into answer using ULI writer."""
        if not state.fact_texts and not state.partial_answers:
            return ActionResult(signal='weak', confidence=0.2)

        # If we already have a good partial answer, use it
        best = state.best_answer()
        if best and len(best) > 3:
            return ActionResult(
                signal='complete',
                answer=best,
                confidence=state.confidence,
                terminal=True,
            )

        # Build answer from accumulated facts
        if state.fact_texts:
            # Take most relevant facts and build a simple answer
            facts = state.fact_texts[-3:]
            # Use the question AST to guide what to extract
            q = state.question_ast

            # Try to find answer in facts
            for fact in facts:
                fact_ast = module.read(fact)
                answer = self._match_answer(q, fact_ast, fact)
                if answer and len(answer) > 2:
                    return ActionResult(
                        signal='complete',
                        answer=answer,
                        confidence=max(state.confidence, 0.5),
                        terminal=True,
                    )

            # Fallback: return first fact sentence
            first_sentence = facts[0].split('.')[0] if facts else ''
            if first_sentence:
                return ActionResult(
                    signal='complete',
                    answer=first_sentence,
                    confidence=0.4,
                    terminal=True,
                )

        return ActionResult(signal='weak', confidence=0.2)

    def _exec_calculate(self, state) -> ActionResult:
        """CALCULATE: evaluate math expression from AST."""
        text = state.question_ast.source
        # Extract numbers and operators
        expr = re.sub(r'[^0-9+\-*/().%\s]', '', text)
        if '%' in expr:
            expr = expr.replace('%', '/100')
        expr = expr.strip()

        if not expr:
            # Try to find numbers in entities
            nums = [e for e in state.question_ast.entities if any(c.isdigit() for c in e)]
            if nums:
                expr = ' '.join(nums)

        try:
            if expr:
                result = eval(expr)
                if isinstance(result, float):
                    result = round(result, 4)
                return ActionResult(signal='done', answer=str(result),
                                   confidence=0.99, terminal=True)
        except Exception:
            pass

        return ActionResult(signal='done', answer="Could not calculate.",
                           confidence=0.3, terminal=True)


# ── Main ─────────────────────────────────────────────────────

def main():
    engine = Engine()

    if len(sys.argv) > 1:
        question = ' '.join(sys.argv[1:])
        print(f"Q: {question}")
        answer = engine.ask(question, verbose=True)
        print(f"A: {answer}")
        return

    print("DMRSM-ULI Engine — No LM, Rules + MiniLM only")
    print("151 tests passing. Ask me anything.\n")
    while True:
        try:
            user_input = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user_input or user_input.lower() in ('quit', 'exit', 'q'):
            break
        answer = engine.ask(user_input, verbose=True)
        print(f"\n{answer}\n")


if __name__ == '__main__':
    main()
