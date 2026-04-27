"""
LM-RAG Engine v4 — Data-Mined Reasoning State Machine (DMRSM)

Derived from 531 reasoning traces across 44 question categories.
Replaces DFS with a state machine driven by empirically mined patterns.

Algorithm: CLASSIFY → [ACTION → EVALUATE → TRANSITION]* → TERMINAL
"""

import os
import re
import sys
import json
import math
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

# Import shared components from v3
from engine import (
    ModelPool, WorkingMemory, pre_filter, SYSTEM_PROMPT,
    ORCHESTRATOR_PATH, GGUF_BASE, GGUF_INSTRUCT, USE_GGUF
)
from search_providers import create_default_engine, SearchResult

log = logging.getLogger('lm-rag-v4')


# ============================================================
# Data Structures
# ============================================================

@dataclass
class ActionResult:
    """Result of executing one action in the state machine."""
    signal: str = ''           # Quality signal for transition lookup
    confidence: float = 0.0
    facts: List[str] = field(default_factory=list)
    answer: str = ''
    searched: bool = False
    terminal: bool = False


@dataclass
class ReasoningState:
    """Full state of the reasoning loop."""
    action: str                          # Current action
    question: str                        # Original (or resolved) question
    question_type: str = 'factual'       # Classified type

    # Working memory
    facts: List[str] = field(default_factory=list)
    sub_questions: List[str] = field(default_factory=list)
    partial_answers: Dict[str, str] = field(default_factory=dict)
    confidence: float = 0.0
    searches_done: int = 0
    steps_done: int = 0
    consecutive_same: int = 0            # Track consecutive same-action (loop guard)
    last_action: str = ''

    # Context
    persona: Optional[str] = None
    safety_level: str = 'safe'
    conversation: List[dict] = field(default_factory=list)

    # Trace
    trace: List[dict] = field(default_factory=list)

    def best_answer(self):
        """Return the best available answer from partial answers."""
        if self.partial_answers:
            # Prefer EXTRACT/SYNTHESIZE answers over others
            for key in ('EXTRACT', 'SYNTHESIZE', 'CALCULATE', 'GENERATE'):
                if key in self.partial_answers:
                    return self.partial_answers[key]
            # Fall back to any answer
            return list(self.partial_answers.values())[-1]
        return ''

    def facts_summary(self, max_chars=600):
        """Summarize accumulated facts."""
        text = ' '.join(self.facts[-5:])
        return text[:max_chars] if text else ''


# ============================================================
# Transition Table (mined from 531 traces)
# ============================================================

TRANSITIONS = {
    # After SEARCH (74% of starts)
    ('SEARCH', 'relevant_high'):   'EXTRACT',
    ('SEARCH', 'relevant_low'):    'JUDGE',
    ('SEARCH', 'irrelevant'):      'SEARCH',       # retry with different query
    ('SEARCH', 'ambiguous'):       'DISAMBIGUATE',
    ('SEARCH', 'no_results'):      'DECOMPOSE',
    ('SEARCH', 'partial'):         'JUDGE',

    # After JUDGE
    ('JUDGE', 'good'):             'EXTRACT',
    ('JUDGE', 'needs_more'):       'SEARCH',
    ('JUDGE', 'needs_synthesis'):  'SYNTHESIZE',
    ('JUDGE', 'unanswerable'):     'GIVE_UP',
    ('JUDGE', 'needs_expert'):     'DEFER',

    # After EXTRACT (30% of endings)
    ('EXTRACT', 'complete'):       'DONE',
    ('EXTRACT', 'needs_more'):     'SEARCH',
    ('EXTRACT', 'multiple'):       'SYNTHESIZE',

    # After DECOMPOSE (10% of starts)
    ('DECOMPOSE', 'has_subs'):     'SEARCH',
    ('DECOMPOSE', 'cant_break'):   'SEARCH',

    # After REASON
    ('REASON', 'need_evidence'):   'SEARCH',
    ('REASON', 'insight'):         'SYNTHESIZE',
    ('REASON', 'more_angles'):     'REASON',

    # After TRIAGE (7% of starts)
    ('TRIAGE', 'safe'):            'SEARCH',
    ('TRIAGE', 'caution'):         'SEARCH',
    ('TRIAGE', 'urgent'):          'DEFER',

    # After PERSONA_ADOPT (3% of starts)
    ('PERSONA_ADOPT', 'adopted'):  'SEARCH',

    # After SYNTHESIZE (48% of endings)
    ('SYNTHESIZE', 'complete'):    'DONE',
    ('SYNTHESIZE', 'weak'):        'SEARCH',

    # Terminal transitions
    ('GIVE_UP', 'done'):           'DONE',
    ('DEFER', 'done'):             'SYNTHESIZE',
    ('CALCULATE', 'done'):         'DONE',
    ('GENERATE', 'done'):          'DONE',
    ('DISAMBIGUATE', 'done'):      'DONE',
}

# Starting state by question type (from trace analysis)
STARTING_STATE = {
    'factual':       'SEARCH',
    'multi_hop':     'DECOMPOSE',
    'deep_thought':  'DECOMPOSE',
    'comparison':    'DECOMPOSE',
    'temporal':      'SEARCH',
    'math':          'CALCULATE',
    'roleplay':      'PERSONA_ADOPT',
    'medical':       'TRIAGE',
    'legal':         'TRIAGE',
    'financial':     'SEARCH',
    'therapy':       'TRIAGE',
    'creative':      'GENERATE',
    'unanswerable':  'SEARCH',
    'disambiguation':'SEARCH',
    'negation':      'SEARCH',
    'advice':        'SEARCH',
    'how_to':        'SEARCH',
    'opinion':       'SEARCH',
    'code':          'SEARCH',
}

# Hyperparameters (from trace statistics)
MAX_STEPS = 12
MAX_SEARCHES = 6
CONFIDENCE_THRESHOLD = 0.7
MAX_CONSECUTIVE_SAME = 3   # Max times same action can repeat


# ============================================================
# Engine v4 — State Machine
# ============================================================

class ReasoningEngine:
    """
    Pattern-driven reasoning engine.
    Replaces DFS with a state machine derived from 531 reasoning traces.
    """

    def __init__(self, pool=None, search_engine=None):
        self.pool = pool if pool else ModelPool()
        self.search_engine = search_engine if search_engine else create_default_engine()
        self.memory = WorkingMemory()
        self._search_retry_queries = set()  # Track retried queries within a run

    # ── Model interface ──────────────────────────────────────
    def call(self, prefix, text, max_len=64):
        return self.pool.call(prefix, text, max_len)

    def call_with_confidence(self, prefix, text, max_len=64):
        return self.pool.call_with_confidence(prefix, text, max_len)

    # ── Classification ───────────────────────────────────────
    def classify(self, question):
        """Classify question type. Model decides, not hardcoded rules."""
        result = self.call('route', question)
        upper = result.strip().upper()

        # Parse TOOL(params) format from v3 model
        match = re.match(r'(\w+)\((.*)\)', result, re.DOTALL)
        if match:
            tool = match.group(1).upper()
            if tool == 'CALCULATE':
                return 'math', match.group(2).strip()
            if tool == 'SEARCH':
                return 'factual', match.group(2).strip()
            if tool == 'MEMORY':
                return 'recall', match.group(2).strip()

        return 'factual', question

    # ── Action Execution ─────────────────────────────────────

    def execute(self, state: ReasoningState) -> ActionResult:
        """Execute the current action and return a result with signal."""

        if state.action == 'SEARCH':
            return self._exec_search(state)
        elif state.action == 'JUDGE':
            return self._exec_judge(state)
        elif state.action == 'EXTRACT':
            return self._exec_extract(state)
        elif state.action == 'DECOMPOSE':
            return self._exec_decompose(state)
        elif state.action == 'REASON':
            return self._exec_reason(state)
        elif state.action == 'SYNTHESIZE':
            return self._exec_synthesize(state)
        elif state.action == 'TRIAGE':
            return self._exec_triage(state)
        elif state.action == 'PERSONA_ADOPT':
            return self._exec_persona_adopt(state)
        elif state.action == 'CALCULATE':
            return self._exec_calculate(state)
        elif state.action == 'GENERATE':
            return self._exec_generate(state)
        elif state.action == 'GIVE_UP':
            return ActionResult(signal='done',
                answer="I couldn't find a confident answer. Could you rephrase?",
                terminal=True)
        elif state.action == 'DEFER':
            return self._exec_defer(state)
        elif state.action == 'DISAMBIGUATE':
            return self._exec_disambiguate(state)
        else:
            # Unknown action — treat as terminal
            return ActionResult(signal='done', terminal=True)

    def _exec_search(self, state):
        # Pick query: sub-question queue > model topic extraction > original question
        if state.sub_questions:
            query = state.sub_questions.pop(0)
        else:
            # Use model to extract search topic
            route_result = self.call('route', state.question)
            match = re.match(r'\w+\((.*)\)', route_result, re.DOTALL)
            query = match.group(1).strip() if match else state.question

        # Avoid re-searching the same query
        if query in self._search_retry_queries:
            query = state.question  # fallback to full question
        self._search_retry_queries.add(query)

        try:
            results = self.search_engine.search(query, max_per_provider=2)
        except Exception as e:
            log.warning(f"Search failed: {e}")
            results = []
        if not results:
            return ActionResult(signal='no_results', searched=True)

        best = results[0]
        # Judge relevance with confidence
        rel, conf = self.call_with_confidence('relevant',
            f"question: {state.question} context: {best.text[:300]}")
        rel_label = rel.strip().upper()

        if rel_label == 'NO':
            return ActionResult(signal='irrelevant', confidence=conf, searched=True)

        # Store fact
        fact = best.text[:500]
        if conf > 0.85:
            return ActionResult(signal='relevant_high', facts=[fact],
                               confidence=conf, searched=True)
        else:
            return ActionResult(signal='relevant_low', facts=[fact],
                               confidence=conf, searched=True)

    def _exec_judge(self, state):
        answer = state.best_answer()
        if not answer:
            # No answer to judge — need more search
            return ActionResult(signal='needs_more', confidence=0.2)

        judge, conf = self.call_with_confidence('judge',
            f"question: {state.question} answer: {answer}")
        judge_label = judge.strip().upper()

        if 'GOOD' in judge_label and conf > 0.7:
            return ActionResult(signal='good', confidence=conf)
        elif 'ECHO' in judge_label or 'VAGUE' in judge_label:
            return ActionResult(signal='needs_more', confidence=conf)
        elif conf < 0.3:
            return ActionResult(signal='unanswerable', confidence=conf)
        else:
            return ActionResult(signal='needs_synthesis', confidence=conf)

    def _exec_extract(self, state):
        context = ' '.join(state.facts[-3:]) if state.facts else ''
        if not context:
            return ActionResult(signal='needs_more', confidence=0.1)

        answer = self.call('answer',
            f"question: {state.question} context: {context}", max_len=60)

        if answer and len(answer.strip()) > 2:
            return ActionResult(signal='complete', answer=answer.strip(),
                               confidence=max(state.confidence, 0.75), terminal=True)
        return ActionResult(signal='needs_more', confidence=0.3)

    def _exec_decompose(self, state):
        prompt = (f"Break this into 3 simpler search queries, one per line. "
                  f"Only output the queries, nothing else.\n{state.question}")
        result = self.call('decompose', prompt, max_len=80)
        lines = [re.sub(r'^[\d\.\-\*\)\:]+\s*', '', l.strip())
                 for l in result.split('\n') if l.strip()]
        # Filter: no duplicates of original, min length
        subs = [l for l in lines if len(l) > 3 and l.lower() != state.question.lower()][:3]
        state.sub_questions = subs
        log.info(f"Decomposed: {subs}")
        if subs:
            return ActionResult(signal='has_subs')
        return ActionResult(signal='cant_break')

    def _exec_reason(self, state):
        facts_text = state.facts_summary()
        if not facts_text:
            return ActionResult(signal='need_evidence')

        reasoning = self.call('reason',
            f"question: {state.question} facts: {facts_text}", max_len=100)

        if reasoning:
            return ActionResult(signal='insight', facts=[reasoning],
                               confidence=0.6)
        return ActionResult(signal='need_evidence')

    def _exec_synthesize(self, state):
        parts = '\n'.join(f"- {a}" for a in state.partial_answers.values()) if state.partial_answers else ''
        facts_text = '\n'.join(state.facts[-5:]) if state.facts else ''

        prompt = f"question: {state.question}"
        if facts_text:
            prompt += f"\nFacts:\n{facts_text}"
        if parts:
            prompt += f"\nPartial answers:\n{parts}"

        answer = self.call('synthesize', prompt, max_len=100)
        if answer and len(answer.strip()) > 2:
            return ActionResult(signal='complete', answer=answer.strip(),
                               confidence=max(state.confidence, 0.6), terminal=True)
        # Synthesis failed — try to use best partial answer
        best = state.best_answer()
        if best:
            return ActionResult(signal='complete', answer=best,
                               confidence=state.confidence, terminal=True)
        return ActionResult(signal='weak', confidence=0.3)

    def _exec_triage(self, state):
        """Assess safety for professional role questions."""
        q = state.question.lower()
        # Urgent: chest pain, breathing difficulty, severe symptoms
        urgent_patterns = [
            r'chest\s+pain', r'can.t\s+breathe', r'breathing\s+difficult',
            r'severe\s+(pain|bleeding)', r'emergency',
        ]
        for p in urgent_patterns:
            if re.search(p, q):
                state.safety_level = 'urgent'
                return ActionResult(signal='urgent')

        # Caution: medication, diagnosis, legal advice
        caution_patterns = [
            r'should\s+i\s+take', r'prescri', r'diagnos',
            r'is\s+it\s+(safe|legal|normal)',
        ]
        for p in caution_patterns:
            if re.search(p, q):
                state.safety_level = 'caution'
                return ActionResult(signal='caution')

        state.safety_level = 'safe'
        return ActionResult(signal='safe')

    def _exec_persona_adopt(self, state):
        # Extract persona from question
        state.persona = self.call('persona',
            f"Define the character for: {state.question}", max_len=60)
        return ActionResult(signal='adopted')

    def _exec_calculate(self, state):
        # Extract expression
        expr = state.question
        # Try to get just the math part
        route_result = self.call('route', state.question)
        match = re.match(r'CALCULATE\((.*)\)', route_result, re.IGNORECASE | re.DOTALL)
        if match:
            expr = match.group(1).strip()

        try:
            clean = expr.replace('^', '**').replace('×', '*').replace('÷', '/')
            clean = re.sub(r'[^0-9+\-*/().%\s]', '', clean)
            if '%' in clean:
                clean = clean.replace('%', '/100')
            if clean.strip():
                result = eval(clean)
                if isinstance(result, float):
                    result = round(result, 4)
                return ActionResult(signal='done', answer=str(result),
                                   confidence=0.99, terminal=True)
        except Exception:
            pass
        # Fallback: ask model
        answer = self.call('answer', f"calculate: {expr}", max_len=30)
        return ActionResult(signal='done', answer=answer,
                           confidence=0.7, terminal=True)

    def _exec_generate(self, state):
        answer = self.call('generate', state.question, max_len=150)
        return ActionResult(signal='done', answer=answer,
                           confidence=0.8, terminal=True)

    def _exec_defer(self, state):
        msg = "Please consult a qualified professional for personalized advice."
        if state.safety_level == 'urgent':
            msg = "This sounds urgent. Please seek immediate medical attention or call 911."
        state.partial_answers['DEFER'] = msg
        return ActionResult(signal='done', answer=msg, confidence=0.9)

    def _exec_disambiguate(self, state):
        # Present what we found
        if state.facts:
            answer = self.call('synthesize',
                f"The term is ambiguous. Explain all meanings based on: {state.facts_summary()}",
                max_len=100)
        else:
            answer = self.call('answer',
                f"This is ambiguous. What could '{state.question}' mean?", max_len=80)
        return ActionResult(signal='done', answer=answer,
                           confidence=0.6, terminal=True)

    # ── Transition ───────────────────────────────────────────

    def transition(self, state: ReasoningState, result: ActionResult) -> str:
        """Determine next action from transition table + guard rails."""

        key = (state.action, result.signal)
        next_action = TRANSITIONS.get(key)

        # Fallback: model decides
        if next_action is None:
            if state.steps_done > MAX_STEPS // 2:
                next_action = 'SYNTHESIZE'
            else:
                next_action = 'SEARCH'

        # Guard rail: max searches
        if state.searches_done >= MAX_SEARCHES and next_action == 'SEARCH':
            next_action = 'SYNTHESIZE'

        # Guard rail: max steps
        if state.steps_done >= MAX_STEPS - 1 and next_action not in ('DONE', 'SYNTHESIZE', 'GIVE_UP'):
            next_action = 'SYNTHESIZE'

        # Guard rail: consecutive same action (prevent infinite loops)
        if next_action == state.last_action:
            state.consecutive_same += 1
        else:
            state.consecutive_same = 0

        if state.consecutive_same >= MAX_CONSECUTIVE_SAME:
            if next_action == 'SEARCH':
                next_action = 'DECOMPOSE' if not state.sub_questions else 'SYNTHESIZE'
            elif next_action == 'REASON':
                next_action = 'SYNTHESIZE'
            else:
                next_action = 'SYNTHESIZE'
            state.consecutive_same = 0

        return next_action

    # ── Main Reasoning Loop ──────────────────────────────────

    def reason(self, question, conversation=None) -> Tuple[str, List[dict], float]:
        """
        Main entry point. Implements DMRSM algorithm.

        Returns: (answer, trace, confidence)
        """
        self._search_retry_queries = set()

        # Phase 0: Safety gate
        is_safe, filter_type, response = pre_filter(question)
        if not is_safe:
            return response, [{'action': 'PRE_FILTER', 'signal': filter_type}], 1.0

        # Phase 1: Classify
        q_type, params = self.classify(question)
        start = STARTING_STATE.get(q_type, 'SEARCH')

        # Override: if classifier returned CALCULATE with expression
        if q_type == 'math' and params and params != question:
            # Store the expression for the CALCULATE action
            question_for_calc = question
        else:
            question_for_calc = question

        state = ReasoningState(
            action=start,
            question=question,
            question_type=q_type,
            conversation=conversation or [],
        )

        log.info(f"[v4] Q='{question[:60]}' type={q_type} start={start}")

        # Phase 3: Reasoning loop
        while state.steps_done < MAX_STEPS:
            # Execute
            result = self.execute(state)

            # Update memory
            if result.facts:
                state.facts.extend(result.facts)
            if result.answer:
                state.partial_answers[state.action] = result.answer
            state.confidence = max(state.confidence, result.confidence)
            if result.searched:
                state.searches_done += 1

            # Trace
            state.trace.append({
                'step': state.steps_done,
                'action': state.action,
                'signal': result.signal,
                'confidence': result.confidence,
                'answer': result.answer[:80] if result.answer else '',
            })
            state.steps_done += 1

            log.info(f"  step={state.steps_done} {state.action}→{result.signal} "
                     f"conf={result.confidence:.2f}")

            # Convergence check
            if result.terminal and state.confidence >= CONFIDENCE_THRESHOLD:
                break
            if state.action == 'DONE':
                break

            # Transition
            state.last_action = state.action
            state.action = self.transition(state, result)

            if state.action == 'DONE':
                break

        # Phase 4: Post-processing
        answer = state.best_answer()
        if not answer:
            answer = "I couldn't find a confident answer. Could you rephrase?"

        # Persona filter
        if state.persona:
            filtered = self.call('persona_filter',
                f"Rewrite in character ({state.persona}): {answer}", max_len=150)
            if filtered and len(filtered) > 5:
                answer = filtered

        # Safety disclaimer
        if state.safety_level == 'urgent':
            if 'emergency' not in answer.lower() and '911' not in answer:
                answer = "⚠ This sounds urgent — please seek immediate medical attention.\n\n" + answer
        elif state.safety_level == 'caution':
            if 'professional' not in answer.lower() and 'doctor' not in answer.lower():
                answer += "\n\nPlease consult a qualified professional for personalized advice."

        # Store in conversation memory
        self.memory.next_turn()
        self.memory.add_turn('user', question)
        self.memory.add_turn('assistant', answer)
        for fact in state.facts:
            self.memory.add_fact(fact, source='search')

        return answer, state.trace, state.confidence

    # ── Convenience method (v3-compatible) ───────────────────

    def ask(self, question, verbose=False):
        """v3-compatible interface."""
        answer, trace, confidence = self.reason(question)
        if verbose:
            for step in trace:
                print(f"  [{step['action']}] → {step['signal']} "
                      f"(conf={step['confidence']:.2f})")
        return answer


# ── Main ─────────────────────────────────────────────────────
def main():
    engine = ReasoningEngine()

    if len(sys.argv) > 1:
        question = ' '.join(sys.argv[1:])
        print(f"Q: {question}")
        answer = engine.ask(question, verbose=True)
        print(f"A: {answer}")
        return

    print("LM-RAG v4 — State Machine Reasoning Engine")
    print("Ask me anything. Type 'quit' to exit.\n")
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
