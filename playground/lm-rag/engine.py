"""
LM-RAG Engine v3 — Dynamic Intent + Orchestrator + DFS

No hardcoded intents. Intents come from DB.
Orchestrator decides what tools to use dynamically.
DFS for multi-hop reasoning.

Tools available to orchestrator:
  - SEARCH: get external facts from Wikipedia
  - CALCULATE: math evaluation
  - MEMORY: recall from conversation
  - RESPOND: generate response (with or without context)

The model UNDERSTANDS intent. The engine provides TOOLS.
"""

import os
import re
import sys
import json
import math
import logging
import torch
import urllib.request
import urllib.parse
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from search_providers import create_default_engine, SearchResult

# Logging
log = logging.getLogger('lm-rag')
logging.basicConfig(
    level=getattr(logging, os.environ.get('LOG_LEVEL', 'INFO').upper(), logging.INFO),
    format='%(levelname)s %(name)s: %(message)s',
)

# Model paths
ORCHESTRATOR_PATH = os.path.join(os.path.dirname(__file__), 'smollm_orchestrator_v7')
GGUF_BASE = os.path.join(os.path.dirname(__file__), 'smollm_v7_q4.gguf')
GGUF_INSTRUCT = os.path.join(os.path.dirname(__file__), 'SmolLM2-135M-Instruct-Q4_K_M.gguf')
USE_GGUF = os.path.exists(GGUF_BASE) and os.path.exists(GGUF_INSTRUCT)

SYSTEM_PROMPT = """You are a language processing engine. You have these tools:
- SEARCH(topic): search for facts. Use when you need external knowledge.
- CALCULATE(expr): evaluate math. Use for arithmetic.
- RESPOND: reply directly. Use for greetings, opinions, creative tasks, explanations from context.

For each user message, output ONLY one:
- SEARCH(topic) if you need facts
- CALCULATE(expression) if it's math
- RESPOND if you can answer directly

After receiving context from search, answer the question based on that context.
For relevance/grounding checks, output YES, PARTIAL, NO, UNSURE, GOOD, ECHO, or VAGUE."""


# ============================================================
# Intent DB — teachable, no hardcoding
# ============================================================

class IntentDB:
    """Intents stored as data, not code. Extensible without code changes."""

    def __init__(self):
        self.intents = [
            # Each intent maps to a tool the engine provides
            {"pattern": "needs_facts", "tool": "SEARCH",
             "description": "question requires external knowledge (who/what/when/where/why/how)"},
            {"pattern": "math", "tool": "CALCULATE",
             "description": "mathematical computation (numbers, percentages, arithmetic)"},
            {"pattern": "recall", "tool": "MEMORY",
             "description": "recall from conversation (repeat, what did you say, sources)"},
            {"pattern": "direct", "tool": "RESPOND",
             "description": "can answer directly (greetings, opinions, creative, explanations with context)"},
        ]

    def get_tool_descriptions(self):
        """Return tool descriptions for the model to understand."""
        return '\n'.join([f"- {i['tool']}: {i['description']}" for i in self.intents])

    def add_intent(self, pattern, tool, description):
        """Teach new intent — no code change needed."""
        self.intents.append({"pattern": pattern, "tool": tool, "description": description})


# ============================================================
# Model Pool
# ============================================================

class ModelPool:
    """Dual model: Base (structured) + Instruct (generation).
    Supports both HF (PyTorch) and GGUF (llama.cpp) backends."""

    INSTRUCT_MODEL = 'HuggingFaceTB/SmolLM2-135M-Instruct'

    def __init__(self, model_path=ORCHESTRATOR_PATH):
        if USE_GGUF:
            self._init_gguf()
        else:
            self._init_hf(model_path)

    def _init_gguf(self):
        from llama_cpp import Llama
        self.backend = 'gguf'
        log.info(f"Loading GGUF base from {GGUF_BASE}")
        self.base = Llama(model_path=GGUF_BASE, n_ctx=512, verbose=False)
        log.info(f"Loading GGUF instruct from {GGUF_INSTRUCT}")
        self.instruct = Llama(model_path=GGUF_INSTRUCT, n_ctx=512, verbose=False)
        log.info("Model pool: dual GGUF loaded")

    def _init_hf(self, model_path):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.backend = 'hf'
        log.info(f"Loading HF base from {model_path}")
        self.tok = AutoTokenizer.from_pretrained(model_path)
        self.im_end_id = self.tok.convert_tokens_to_ids('<|im_end|>')
        self.base = AutoModelForCausalLM.from_pretrained(model_path)
        self.base.eval()
        params = sum(p.numel() for p in self.base.parameters())
        log.info(f"Base: {params/1e6:.0f}M params")
        log.info(f"Loading HF instruct from {self.INSTRUCT_MODEL}")
        self.instruct = AutoModelForCausalLM.from_pretrained(self.INSTRUCT_MODEL)
        self.instruct.eval()
        log.info(f"Instruct: {params/1e6:.0f}M params")

    def _call_gguf(self, llm, prompt, max_len):
        out = llm(prompt, max_tokens=max_len, stop=['<|im_end|>'], echo=False)
        return out['choices'][0]['text'].split('\n')[0].strip()

    def _call_hf(self, model, prompt, max_len):
        ids = self.tok(prompt, return_tensors='pt').input_ids
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=max_len, do_sample=False,
                pad_token_id=self.tok.eos_token_id, eos_token_id=self.im_end_id)
        return self.tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).split('\n')[0].strip()

    def _call_gguf_with_conf(self, llm, prompt, max_len):
        """GGUF inference returning (text, confidence) via token log-probs."""
        out = llm(prompt, max_tokens=max_len, stop=['<|im_end|>'], echo=False, logprobs=1)
        text = out['choices'][0]['text'].split('\n')[0].strip()
        logprobs = out['choices'][0].get('logprobs', {})
        token_logprobs = logprobs.get('token_logprobs', [])
        if token_logprobs and token_logprobs[0] is not None:
            conf = math.exp(token_logprobs[0])
        else:
            conf = 0.5
        return text, conf

    def _call_hf_with_conf(self, model, prompt, max_len):
        """HF inference returning (text, confidence) via output scores."""
        ids = self.tok(prompt, return_tensors='pt').input_ids
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=max_len, do_sample=False,
                pad_token_id=self.tok.eos_token_id, eos_token_id=self.im_end_id,
                output_scores=True, return_dict_in_generate=True)
        text = self.tok.decode(out.sequences[0][ids.shape[1]:],
                               skip_special_tokens=True).split('\n')[0].strip()
        if out.scores:
            probs = torch.softmax(out.scores[0][0], dim=-1)
            first_token_id = out.sequences[0][ids.shape[1]]
            conf = probs[first_token_id].item()
        else:
            conf = 0.5
        return text, conf

    def call_with_confidence(self, prefix, text, max_len=64):
        """Like call() but returns (result, confidence) where confidence is 0-1.
        Uses first-token log-probability as the confidence signal."""
        prompt = (f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
                  f"<|im_start|>user\n{prefix}: {text}<|im_end|>\n"
                  f"<|im_start|>assistant\n")
        if self.backend == 'gguf':
            result, conf = self._call_gguf_with_conf(self.base, prompt, max_len)
        else:
            result, conf = self._call_hf_with_conf(self.base, prompt, max_len)
        log.debug(f"[{prefix}] ...→ {result[:60]} (conf={conf:.3f})")
        return result, conf

    def call(self, prefix, text, max_len=64):
        """Route to base (structured + extraction) or instruct (conversation)."""
        if prefix in ('route', 'relevant', 'judge', 'answer'):
            # Base model: structured output + concise extraction
            prompt = (f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
                      f"<|im_start|>user\n{prefix}: {text}<|im_end|>\n"
                      f"<|im_start|>assistant\n")
            if self.backend == 'gguf':
                result = self._call_gguf(self.base, prompt, max_len)
            else:
                result = self._call_hf(self.base, prompt, max_len)
        else:
            prompt = (f"<|im_start|>user\n{text}<|im_end|>\n"
                      f"<|im_start|>assistant\n")
            if self.backend == 'gguf':
                result = self._call_gguf(self.instruct, prompt, max_len)
            else:
                result = self._call_hf(self.instruct, prompt, max_len)

        log.debug(f"[{prefix}] ...→ {result[:60]}")
        return result


# ============================================================
# Agent Trace
# ============================================================

@dataclass
class AgentTrace:
    query: str
    depth: int
    article_title: str = ''
    relevant: str = ''
    relevant_conf: float = 0.0
    answer: str = ''
    grounded: str = ''
    judge: str = ''
    judge_conf: float = 0.0
    confidence: float = 0.0     # P(relevant=YES) × P(judge=GOOD)
    context_snippet: str = ''
    converged: bool = False


# ============================================================
# Working Memory
# ============================================================

@dataclass
class WorkingMemory:
    facts: list = field(default_factory=list)
    entities: list = field(default_factory=list)
    conversation: list = field(default_factory=list)
    preferences: dict = field(default_factory=dict)
    personal: dict = field(default_factory=dict)
    turn: int = 0

    def add_fact(self, text, source='unknown'):
        self.facts.append({'text': text[:500], 'source': source, 'turn': self.turn})
        entities = re.findall(r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+', text[:300])
        for e in entities[:5]:
            self.entities.append({'name': e, 'turn': self.turn})

    def add_turn(self, role, text):
        self.conversation.append({'role': role, 'text': text[:500], 'turn': self.turn})

    def last_response(self):
        for turn in reversed(self.conversation):
            if turn['role'] == 'assistant':
                return turn['text']
        return None

    def last_entities(self, n=3):
        return [e['name'] for e in self.entities[-n:]]

    def context_string(self, max_chars=400):
        texts = [f['text'] for f in self.facts[-5:]]
        ctx = ' '.join(texts)
        return ctx[:max_chars] if ctx else ''

    def next_turn(self):
        self.turn += 1


# ============================================================
# Pre-filter (safety — the ONLY hardcoded thing)
# ============================================================

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
# Wikidata Type Checker
# ============================================================

_wikidata_cache = {}

def wikidata_entity_type(entity):
    if entity in _wikidata_cache:
        return _wikidata_cache[entity]
    try:
        url = 'https://www.wikidata.org/w/api.php?' + urllib.parse.urlencode({
            'action': 'wbsearchentities', 'search': entity,
            'language': 'en', 'format': 'json', 'limit': 1,
        })
        req = urllib.request.Request(url, headers={'User-Agent': 'LM-RAG/1.0'})
        with urllib.request.urlopen(req, timeout=5) as r:
            data = json.loads(r.read())
        if not data.get('search'):
            _wikidata_cache[entity] = set()
            return set()
        qid = data['search'][0]['id']
        url2 = 'https://www.wikidata.org/w/api.php?' + urllib.parse.urlencode({
            'action': 'wbgetclaims', 'entity': qid, 'property': 'P31', 'format': 'json',
        })
        req2 = urllib.request.Request(url2, headers={'User-Agent': 'LM-RAG/1.0'})
        with urllib.request.urlopen(req2, timeout=5) as r2:
            claims = json.loads(r2.read())
        types = set()
        for claim in claims.get('claims', {}).get('P31', []):
            try:
                types.add(claim['mainsnak']['datavalue']['value']['id'])
            except (KeyError, TypeError):
                pass
        _wikidata_cache[entity] = types
        return types
    except Exception:
        _wikidata_cache[entity] = set()
        return set()


def type_check(question, answer):
    if question.lower().strip().startswith(('who ', 'whom ')):
        if len(answer.strip()) < 3:
            return True
        types = wikidata_entity_type(answer.strip())
        if types:
            return 'Q5' in types
    return True


# ============================================================
# Engine v3 — Dynamic Intent + DFS
# ============================================================

class LMRAGEngine:

    def __init__(self, pool=None):
        if pool:
            self.pool = pool
        else:
            self.pool = ModelPool()

        self.intent_db = IntentDB()

        log.info("Loading search providers...")
        self.search_engine = create_default_engine()

        self.memory = WorkingMemory()
        log.info(f"Search providers: {[p.name for p in self.search_engine.providers]}")
        log.info("Ready.")

    # ── Model interface ──────────────────────────────────────
    def call(self, prefix, text, max_len=64):
        return self.pool.call(prefix, text, max_len)

    def call_with_confidence(self, prefix, text, max_len=64):
        return self.pool.call_with_confidence(prefix, text, max_len)

    def generate(self, prompt, max_len=128):
        return self.call('answer', prompt, max_len)

    # ── Dynamic intent detection ─────────────────────────────
    def detect_intent(self, user_input):
        """Model decides what tool to use. Returns (tool, params)."""
        result = self.call('route', user_input)
        log.info(f"Intent: {result}")

        # Parse tool call
        match = re.match(r'(\w+)\((.*)\)', result, re.DOTALL)
        if match:
            tool = match.group(1).upper()
            params = match.group(2).strip()
            return tool, params

        # No tool detected — model wants to respond directly
        return 'RESPOND', ''

    # ── Sub-question generation ────────────────────────────────
    def decompose(self, question, max_sub=3):
        """Break a question into simpler sub-questions for multi-hop search.
        Uses instruct model — it generates, base model classifies."""
        prompt = (f"Break this into {max_sub} simpler search queries, one per line. "
                  f"Only output the queries, nothing else.\n{question}")
        result = self.call('decompose', prompt, max_len=80)
        lines = [re.sub(r'^[\d\.\-\*\)\:]+\s*', '', l.strip())
                 for l in result.split('\n') if l.strip()]
        sub_qs = [l for l in lines if len(l) > 3 and l.lower() != question.lower()]
        log.info(f"Sub-questions: {sub_qs[:max_sub]}")
        return sub_qs[:max_sub]

    # ── DFS Search ───────────────────────────────────────────
    def search(self, query, user_input):
        """DFS tree search — agents explore, orchestrator synthesizes."""
        log.info(f"DFS: query='{query}' question='{user_input[:50]}'")

        visited = set()
        context_seen = set()
        traces: List[AgentTrace] = []
        skip_entities = {'The', 'This', 'That', 'And', 'For', 'With', 'During',
                         'Between', 'After', 'Before', 'Under', 'Over'}

        def agent_node(search_query, depth=0, max_depth=3):
            """Agent: search → relevant → extract → judge. Converge or branch."""
            if depth >= max_depth or search_query in visited or len(visited) > 10:
                return
            visited.add(search_query)
            log.debug(f"Agent: depth={depth} query='{search_query}'")

            results = self.search_engine.search(search_query, max_per_provider=2)
            for result in results:
                text_key = result.text[:100]
                if text_key in context_seen:
                    continue
                context_seen.add(text_key)

                # 1. Relevant? (with confidence)
                rel, rel_conf = self.call_with_confidence('relevant',
                    f"question: {user_input} context: {result.text[:300]}")
                rel_label = rel.strip().upper()

                if rel_label == 'NO':
                    traces.append(AgentTrace(query=search_query, depth=depth,
                        article_title=result.title, relevant='NO',
                        relevant_conf=rel_conf))
                    # Branch: extract entities, try one
                    entities = re.findall(r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+', result.text[:300])
                    for e in entities[:1]:
                        if e not in skip_entities and len(e) > 4:
                            agent_node(e, depth + 1, max_depth)
                    continue

                self.memory.add_fact(result.text[:500], source=result.source)

                # 2. Extract answer (base model — concise extraction)
                context = result.text[:1200]
                answer = self.call('answer',
                    f"question: {user_input} context: {context}", max_len=40)
                if not answer:
                    continue

                if not type_check(user_input, answer):
                    continue

                # 3. Judge (with confidence)
                judge, judge_conf = self.call_with_confidence('judge',
                    f"question: {user_input} answer: {answer}")
                judge_label = judge.strip().upper()

                # Convergence VALUE: P(relevant=YES) × P(judge=GOOD)
                confidence = rel_conf * judge_conf
                is_converged = (judge_label == 'GOOD' and rel_label == 'YES')

                log.info(f"  [{result.title}] rel={rel_label}({rel_conf:.2f}) "
                         f"judge={judge_label}({judge_conf:.2f}) → conf={confidence:.3f}"
                         f"{' ✓' if is_converged else ''}")

                traces.append(AgentTrace(
                    query=search_query, depth=depth,
                    article_title=result.title, relevant=rel_label,
                    relevant_conf=rel_conf, answer=answer,
                    judge=judge_label, judge_conf=judge_conf,
                    confidence=confidence,
                    context_snippet=context[:200],
                    converged=is_converged))

                # CONVERGE: fully relevant + good quality
                if is_converged:
                    return

                # NOT GOOD: branch on entities from article text
                entities = re.findall(r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+', result.text[:300])
                for e in entities[:1]:
                    if e not in skip_entities and len(e) > 4:
                        agent_node(e, depth + 1, max_depth)

        # Convergence loop: try multiple search strategies
        # Strategy 1: direct question
        agent_node(query, depth=0)

        # Check: did we converge?
        if any(t.converged for t in traces):
            return self.orchestrate(user_input, traces)

        # Strategy 2: try the full user question (different search angle)
        if user_input not in visited:
            agent_node(user_input, depth=0)
            if any(t.converged for t in traces):
                return self.orchestrate(user_input, traces)

        # Strategy 3: decompose into sub-questions (multi-hop reasoning)
        # Run ALL sub-questions — don't stop at first convergence.
        # Multiple partial answers may need synthesis.
        sub_questions = self.decompose(user_input)
        for sq in sub_questions:
            if sq not in visited:
                agent_node(sq, depth=0)

        return self.orchestrate(user_input, traces)

    # ── Orchestrator ─────────────────────────────────────────
    def orchestrate(self, user_input, traces: List[AgentTrace]):
        log.info(f"Orchestrator: {len(traces)} traces")
        self._last_traces = traces  # Store for debugging

        # Converged traces: relevant=YES + judge=GOOD
        converged = [t for t in traces if t.converged]

        if converged:
            # Deduplicate answers (different sub-questions may find the same thing)
            unique_answers = {}
            for t in converged:
                ans = t.answer.strip()
                if ans not in unique_answers or t.confidence > unique_answers[ans].confidence:
                    unique_answers[ans] = t

            if len(unique_answers) == 1:
                # Single converged answer — return directly
                best = max(unique_answers.values(), key=lambda t: t.confidence)
                log.info(f"Converged: '{best.answer}' (conf={best.confidence:.3f})")
                return best.answer

            # Multiple different converged answers → synthesize
            # Feed all partial answers + original question to instruct model
            parts = [f"- {t.query}: {t.answer}" for t in
                     sorted(unique_answers.values(), key=lambda t: t.confidence, reverse=True)]
            synthesis_prompt = (f"Answer this question using the facts below.\n"
                                f"Question: {user_input}\nFacts:\n" + '\n'.join(parts))
            log.info(f"Synthesizing {len(parts)} partial answers")
            synthesized = self.call('synthesize', synthesis_prompt, max_len=60)
            if synthesized and len(synthesized.strip()) > 2:
                return synthesized.strip()

            # Synthesis failed — fall back to highest confidence single answer
            best = max(unique_answers.values(), key=lambda t: (t.confidence, -t.depth))
            return best.answer

        # Any answer that passed judge (not ECHO/VAGUE/etc)
        with_answer = [t for t in traces if t.answer and t.judge not in ('ECHO', 'VAGUE', 'TOO_SHORT', 'TYPE_MISMATCH')]
        if with_answer:
            best = max(with_answer, key=lambda t: (t.confidence, -t.depth))
            return best.answer

        # Any answer at all (fallback)
        any_answer = [t for t in traces if t.answer]
        if any_answer:
            best = max(any_answer, key=lambda t: t.confidence)
            return best.answer

        return "I couldn't find a confident answer. Could you rephrase?"

    # ── Calculate ────────────────────────────────────────────
    def calculate(self, expr):
        try:
            clean = expr.replace('^', '**').replace('×', '*').replace('÷', '/')
            clean = re.sub(r'[^0-9+\-*/().%\s]', '', clean)
            if '%' in clean:
                clean = clean.replace('%', '/100')
            result = eval(clean)
            if isinstance(result, float):
                result = round(result, 4)
            return str(result)
        except Exception:
            return self.generate(f"calculate: {expr}")

    # ── Memory recall ────────────────────────────────────────
    def recall(self, what):
        if 'last' in what:
            last = self.memory.last_response()
            return last if last else "I haven't said anything yet."
        elif 'source' in what:
            sources = [f['source'] for f in self.memory.facts[-3:]]
            return f"Recent sources: {', '.join(sources)}" if sources else "No sources yet."
        ctx = self.memory.context_string()
        if ctx:
            return self.generate(f"summarize this conversation: {ctx}")
        return "We haven't discussed much yet."

    # ── Main entry point ─────────────────────────────────────
    def ask(self, user_input, verbose=False):
        log.info(f"Question: '{user_input[:80]}'")
        self.memory.next_turn()
        self.memory.add_turn('user', user_input)

        # Pre-filter (safety — only hardcoded thing)
        is_safe, filter_type, response = pre_filter(user_input)
        if not is_safe:
            log.warning(f"Pre-filter: {filter_type}")
            self.memory.add_turn('assistant', response)
            return response

        # Dynamic intent detection
        tool, params = self.detect_intent(user_input)

        if verbose:
            print(f"  [intent: {tool}({params})]")

        # Execute tool
        if tool == 'SEARCH':
            # Model decides what to search — no hardcoded word lists
            search_query = params if params else user_input
            log.info(f"Search query: '{search_query}'")
            response = self.search(search_query, user_input)
        elif tool == 'CALCULATE':
            response = self.calculate(params)
        elif tool == 'MEMORY':
            response = self.recall(params)
        else:
            # RESPOND — use Instruct model for natural conversation
            ctx = self.memory.context_string()
            prompt = f"{user_input}"
            if ctx:
                prompt = f"{user_input} Context: {ctx}"
            # Call Instruct directly for natural conversation
            if hasattr(self.pool, '_call_gguf') and self.pool.backend == 'gguf':
                instruct_prompt = (f"<|im_start|>user\n{prompt}<|im_end|>\n"
                                  f"<|im_start|>assistant\n")
                response = self.pool._call_gguf(self.pool.instruct, instruct_prompt, 100)
            elif hasattr(self.pool, '_call_hf') and self.pool.backend == 'hf':
                instruct_prompt = (f"<|im_start|>user\n{prompt}<|im_end|>\n"
                                  f"<|im_start|>assistant\n")
                response = self.pool._call_hf(self.pool.instruct, instruct_prompt, 100)
            else:
                # Mock or fallback — use standard call
                response = self.call('answer', prompt)

        self.memory.add_turn('assistant', response)
        return response


# ── Main ─────────────────────────────────────────────────────
def main():
    engine = LMRAGEngine()

    if len(sys.argv) > 1:
        question = ' '.join(sys.argv[1:])
        print(f"Q: {question}")
        answer = engine.ask(question, verbose=True)
        print(f"A: {answer}")
        return

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
