"""
ULI Graph Reasoner — DMRSM-style state machine for Knowledge Graph question answering.

States (mirrors the DMRSM transition table in dmrsm_uli.py):

  PARSE     ULI tags and parses the question text.
  RESOLVE   EntityResolver maps surface mentions → KG node IDs.
  DECOMPOSE Detect which hop-pattern the question maps to.
  HOP       Traverse/reverse graph edges (loops until answer or dead end).
  COMPILE   Build answer text + MeaningAST from the traversal path.
  DONE      Return answer — caller skips DMRSM web-search loop.
  FAILED    Graph has no path — caller falls through to DMRSM SEARCH state.

Features:
  - Multilingual: language-specific QW maps, agent descriptor maps.
    Language is auto-detected if not specified (uses uli/lexer.py detect_language()).
  - Code switching: entity mentions in any language resolve via multilingual aliases.
  - Normalization: BK-tree spelling correction + slang expansion before parsing.
  - Resilience: all exceptions caught, returns None instead of propagating.
  - Working memory: pronoun resolution across conversation turns.
  - MAX_HOPS guard: prevents infinite loops.
"""

import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from .protocol import MeaningAST, Entity
from .graph import GraphDB
from .resolver import EntityResolver
from .memory import WorkingMemory, _ALL_PRONOUNS

log = logging.getLogger('uli.reasoner')

MAX_HOPS = 10   # Safety guard — raise RuntimeError if exceeded


# ── State machine ─────────────────────────────────────────────────────────────

class State(Enum):
    PARSE     = 'PARSE'
    RESOLVE   = 'RESOLVE'
    DECOMPOSE = 'DECOMPOSE'
    HOP       = 'HOP'
    COMPILE   = 'COMPILE'
    DONE      = 'DONE'
    FAILED    = 'FAILED'


@dataclass
class TraceStep:
    """One step in the DMRSM reasoning trace (matches dmrsm_uli.py trace format)."""
    action: str
    node_id: str = ''
    relation: str = ''
    result_id: str = ''
    confidence: float = 1.0
    note: str = ''
    answer: str = ''

    def as_dict(self) -> dict:
        return {
            'action':     self.action,
            'node_id':    self.node_id,
            'relation':   self.relation,
            'result_id':  self.result_id,
            'confidence': self.confidence,
            'note':       self.note,
            'answer':     self.answer,
        }


# ── Hop step descriptor ───────────────────────────────────────────────────────

@dataclass
class HopStep:
    """A single graph traversal step in a multi-hop chain."""
    direction: str    # 'forward' (A→B) or 'reverse' (B←A)
    source: str       # slot name: 'agent'|'patient'|'theme'|'intermediate'|'_of_entity'
    relation: str     # KG edge relation
    target: str       # result slot: 'agent'|'patient'|'location'|'time'|'intermediate'


# ── Multilingual question word maps ───────────────────────────────────────────

_QW_MAP: Dict[str, frozenset] = {
    'en': frozenset({'what', 'who', 'where', 'when', 'why', 'how', 'which'}),
    'hi': frozenset({'कहाँ', 'क्या', 'कौन', 'कब', 'क्यों', 'कैसे'}),
    'mr': frozenset({'कुठे', 'काय', 'कोण', 'केव्हा', 'का', 'कसे'}),
    'fr': frozenset({'où', 'que', 'qui', 'quand', 'pourquoi', 'comment', 'quel', 'quelle'}),
    'de': frozenset({'wo', 'was', 'wer', 'wann', 'warum', 'wie', 'welche', 'welcher'}),
    'es': frozenset({'dónde', 'qué', 'quién', 'cuándo', 'cómo', 'cuál', 'dónde'}),
    'ar': frozenset({'أين', 'ما', 'من', 'متى', 'لماذا', 'كيف'}),
    'zh': frozenset({'哪里', '什么', '谁', '什么时候', '为什么', '怎么'}),
    'ja': frozenset({'どこ', '何', '誰', 'いつ', 'なぜ', 'どのように'}),
    'ru': frozenset({'где', 'что', 'кто', 'когда', 'почему', 'как'}),
    'ko': frozenset({'어디', '무엇', '누가', '언제', '왜', '어떻게'}),
}

# Fallback: all known QW in any language
_ALL_QW = frozenset().union(*_QW_MAP.values())

_QW_TO_TARGET: Dict[str, Dict[str, str]] = {
    'en': {'where': 'location', 'when': 'time', 'who': 'agent', 'what': 'patient', 'which': 'patient'},
    'hi': {'कहाँ': 'location', 'कब': 'time', 'कौन': 'agent', 'क्या': 'patient'},
    'mr': {'कुठे': 'location', 'केव्हा': 'time', 'कोण': 'agent', 'काय': 'patient'},
    'fr': {'où': 'location', 'quand': 'time', 'qui': 'agent', 'que': 'patient'},
    'de': {'wo': 'location', 'wann': 'time', 'wer': 'agent', 'was': 'patient'},
    'es': {'dónde': 'location', 'cuándo': 'time', 'quién': 'agent', 'qué': 'patient'},
}

# ── Agent descriptor maps per language ────────────────────────────────────────

_AGENT_DESCRIPTORS_MAP: Dict[str, frozenset] = {
    'en': frozenset({
        'creator', 'inventor', 'author', 'founder', 'discoverer',
        'writer', 'architect', 'designer', 'developer', 'maker',
        'pioneer', 'father', 'mother', 'originator',
    }),
    'hi': frozenset({'निर्माता', 'आविष्कारक', 'लेखक', 'संस्थापक', 'खोजकर्ता'}),
    'mr': frozenset({'निर्माता', 'शोधक', 'लेखक', 'संस्थापक'}),
    'fr': frozenset({'créateur', 'inventeur', 'auteur', 'fondateur', 'découvreur'}),
    'de': frozenset({'Erfinder', 'Autor', 'Gründer', 'Entdecker', 'Schöpfer'}),
    'es': frozenset({'creador', 'inventor', 'autor', 'fundador', 'descubridor'}),
}

# Verb lemma → KG relation name (English; other languages extend via QW mapping)
_PREDICATE_TO_RELATION: Dict[str, str] = {
    'born':       'born_in',
    'birth':      'born_in',
    'die':        'died_in',
    'death':      'died_in',
    'died':       'died_in',
    'create':     'created',
    'make':       'created',
    'invent':     'invented',
    'discover':   'discovered',
    'write':      'wrote',
    'author':     'wrote',
    'found':      'founded',
    'establish':  'founded',
    'capital':    'capital_of',
    'locate':     'located_in',
    'work':       'worked_at',
    'study':      'studied_at',
    'marry':      'married_to',
}

# Agent descriptor → KG relation for first hop in two-hop
_DESCRIPTOR_TO_REL: Dict[str, str] = {
    'creator':    'created',
    'inventor':   'invented',
    'author':     'wrote',
    'founder':    'founded',
    'discoverer': 'discovered',
    'writer':     'wrote',
    'architect':  'designed',
    'designer':   'designed',
}

# Answer sentence templates
_TEMPLATES: Dict[str, str] = {
    'born_in':       '{agent} was born in {object}.',
    'died_in':       '{agent} died in {object}.',
    'created':       '{agent} created {object}.',
    'created_by':    '{object} was created by {agent}.',
    'invented':      '{agent} invented {object}.',
    'invented_by':   '{object} was invented by {agent}.',
    'discovered':    '{agent} discovered {object}.',
    'discovered_by': '{object} was discovered by {agent}.',
    'wrote':         '{agent} wrote {object}.',
    'written_by':    '{object} was written by {agent}.',
    'founded':       '{agent} founded {object}.',
    'founded_by':    '{object} was founded by {agent}.',
    'capital_of':    '{agent} is the capital of {object}.',
    'located_in':    '{agent} is located in {object}.',
    'nationality':   '{agent} is {object}.',
    'worked_at':     '{agent} worked at {object}.',
    'studied_at':    '{agent} studied at {object}.',
    'married_to':    '{agent} is married to {object}.',
    'part_of':       '{agent} is part of {object}.',
}

_PAST_TENSE_RELATIONS = frozenset({
    'born_in', 'died_in', 'created', 'invented', 'discovered',
    'wrote', 'founded', 'worked_at', 'studied_at',
})


# ── Pattern decomposition helpers ─────────────────────────────────────────────

def _rel(predicate: str) -> str:
    return _PREDICATE_TO_RELATION.get(predicate.lower(), predicate.lower())


def _detect_two_hop(tagged: list, parsed, agent_descriptors: frozenset) -> Optional[Tuple[str, str]]:
    """Detect 'creator/inventor/... of X' pattern. Returns (descriptor, entity_text) or None."""
    _TERMINATORS = frozenset({
        'born', 'died', 'found', 'invented', 'discovered', 'written', 'created',
        'made', 'built', 'established', 'founded',
    })
    for i, (w, pos, lemma) in enumerate(tagged):
        if w.lower() in agent_descriptors:
            for j in range(i + 1, min(i + 4, len(tagged))):
                if tagged[j][0].lower() == 'of' and j + 1 < len(tagged):
                    entity_tokens = []
                    for k in range(j + 1, len(tagged)):
                        wk, pk, lk = tagged[k]
                        if pk == 'DET':
                            continue
                        if wk.lower() in _TERMINATORS:
                            break
                        if pk in ('VERB', 'AUX', 'ADV') and entity_tokens:
                            break
                        if wk in ('?', '.', '!'):
                            break
                        entity_tokens.append(wk)
                    if entity_tokens:
                        return (w.lower(), ' '.join(entity_tokens))
    return None


def _decompose(tagged: list, parsed, ast: MeaningAST, db,
               resolver: EntityResolver, lang: str = 'en',
               agent_descriptors: frozenset = None) -> List[HopStep]:
    """Map question to a sequence of HopSteps."""
    if agent_descriptors is None:
        agent_descriptors = _AGENT_DESCRIPTORS_MAP.get(lang, _AGENT_DESCRIPTORS_MAP['en'])

    qw   = (ast.question_word or '').lower()
    pred = (ast.predicate or '').lower()
    steps: List[HopStep] = []

    # ── Two-hop: "creator/inventor/author of X" ────────────────────────────────
    two_hop = _detect_two_hop(tagged, parsed, agent_descriptors)
    if two_hop:
        descriptor, of_entity_text = two_hop
        intermediate_rel = _DESCRIPTOR_TO_REL.get(descriptor, 'created')
        of_node_id = resolver.resolve_text(of_entity_text, lang=lang)
        if of_node_id:
            hop0 = HopStep(direction='reverse', source='_of_entity',
                           relation=intermediate_rel, target='intermediate')
            hop0.__dict__['_of_node_id'] = of_node_id
            steps.append(hop0)

            tagged_lower = {w.lower() for w, _, _ in tagged}
            if 'born' in tagged_lower:
                main_rel = 'born_in'
            elif 'died' in tagged_lower:
                main_rel = 'died_in'
            elif pred and pred not in ('be', 'be born'):
                main_rel = _rel(pred)
            else:
                main_rel = 'born_in'

            steps.append(HopStep(direction='forward', source='intermediate',
                                 relation=main_rel,
                                 target=ast.question_target or 'location'))
            return steps

    # ── Single-hop ─────────────────────────────────────────────────────────────
    main_rel = _rel(pred)

    _PARTICIPLE_TO_REL = {
        'born': 'born_in', 'died': 'died_in',
        'invented': 'invented', 'discovered': 'discovered',
        'created': 'created', 'wrote': 'wrote', 'founded': 'founded',
    }
    tagged_lower = {w.lower() for w, _, _ in tagged}
    for pp, rel in _PARTICIPLE_TO_REL.items():
        if pp in tagged_lower:
            main_rel = rel
            pred = pp
            break

    if qw in ('where', 'कहाँ', 'कुठे', 'où', 'wo', 'dónde', 'أين', '哪里', 'どこ', 'где', '어디'):
        if pred in ('born', 'birth') or main_rel == 'born_in':
            steps.append(HopStep('forward', 'agent', 'born_in', 'location'))
        elif pred in ('die', 'died', 'death') or main_rel == 'died_in':
            steps.append(HopStep('forward', 'agent', 'died_in', 'location'))
        else:
            steps.append(HopStep('forward', 'agent', main_rel, 'location'))

    elif qw in ('who', 'कौन', 'कोण', 'qui', 'wer', 'quién', 'من', '谁', '誰', 'кто', '누가'):
        steps.append(HopStep('reverse', 'patient', main_rel, 'agent'))

    elif qw in ('what', 'क्या', 'काय', 'que', 'was', 'qué', 'ما', '什么', '何', 'что', '무엇'):
        if 'capital' in (pred + ' '.join(w.lower() for w, _, _ in tagged)):
            steps.append(HopStep('reverse', 'patient', 'capital_of', 'patient'))
        else:
            steps.append(HopStep('forward', 'agent', main_rel, 'patient'))

    elif qw in ('when', 'कब', 'केव्हा', 'quand', 'wann', 'cuándo', 'متى', '什么时候', 'いつ', 'когда', '언제'):
        steps.append(HopStep('forward', 'agent', main_rel, 'time'))

    elif qw in ('which', 'how', 'कैसे', 'comment', 'wie', 'cómo', 'كيف', '怎么', 'どのように', 'как', '어떻게'):
        steps.append(HopStep('forward', 'agent', main_rel, 'patient'))

    return steps


# ── Compile ───────────────────────────────────────────────────────────────────

def _compile(path: List[dict], question_ast: MeaningAST, db, lang: str = 'en') -> Tuple[str, MeaningAST]:
    """Build answer text + MeaningAST from the traversal path."""
    if not path:
        return '', MeaningAST(intent='unanswerable')

    last = path[-1]
    from_id  = last['from']
    to_id    = last['to']
    relation = last['relation']

    # Use lang-aware label if db supports it
    from_label = db.label(from_id, lang) if _db_supports_lang_label(db) else db.label(from_id)
    to_label   = db.label(to_id, lang)   if _db_supports_lang_label(db) else db.label(to_id)

    # Try relation template from DB first (localized), then local dict
    template = None
    if hasattr(db, 'relation_template'):
        template = db.relation_template(relation, lang)
    if not template:
        template = _TEMPLATES.get(relation, '{agent} — {relation} — {object}.')

    answer_text = (template
                   .replace('{agent}',    from_label)
                   .replace('{object}',   to_label)
                   .replace('{relation}', relation.replace('_', ' ')))

    from_node = db.node(from_id) or {}
    to_node   = db.node(to_id)   or {}

    answer_ast = MeaningAST(
        type='statement',
        intent='factual',
        predicate=relation.replace('_', ' '),
        agent=Entity(text=from_label, type=from_node.get('type', 'unknown'), id=from_id),
        patient=Entity(text=to_label, type=to_node.get('type', 'unknown'),   id=to_id),
        tense='past' if relation in _PAST_TENSE_RELATIONS else 'present',
        source=answer_text,
        confidence=1.0,
    )
    if to_node.get('type') == 'location':
        answer_ast.location = Entity(text=to_label, type='location', id=to_id)
    if to_node.get('type') == 'time':
        answer_ast.time = Entity(text=to_label, type='time', id=to_id)

    return answer_text, answer_ast


def _db_supports_lang_label(db) -> bool:
    """Check if db.label() accepts a lang= keyword argument."""
    import inspect
    try:
        sig = inspect.signature(db.label)
        return len(sig.parameters) >= 2
    except (TypeError, ValueError):
        return False


# ── SentenceStructure → MeaningAST bridge ────────────────────────────────────

def _parse_to_ast(tagged: list, parsed, lang: str = 'en') -> MeaningAST:
    """Build a minimal MeaningAST from ULI SentenceStructure + tagged tokens."""
    ast = MeaningAST()

    _PASSIVE_PARTICIPLES: Dict[str, str] = {
        'born': 'born', 'died': 'died', 'found': 'found',
        'invented': 'invented', 'discovered': 'discovered',
        'written': 'wrote', 'created': 'created', 'made': 'made',
        'built': 'built', 'founded': 'founded',
    }

    # Past-tense auxiliary forms
    _PAST_AUX = frozenset({'was', 'were', 'did', 'had'})

    # Question word (language-aware)
    _QW = _QW_MAP.get(lang, _QW_MAP['en'])
    if tagged:
        first = tagged[0][0].lower()
        if first in _QW or first in _ALL_QW:
            ast.type = 'question'
            ast.question_word = first

    if any(w == '?' for w, _, _ in tagged):
        ast.type = 'question'

    # Tense — detect from AUX tokens
    for w, pos, lemma in tagged:
        if pos == 'AUX' and w.lower() in _PAST_AUX:
            ast.tense = 'past'
            break

    # Predicate
    ast.predicate = parsed.verb_lemma or parsed.verb
    for w, pos, lemma in tagged:
        if w.lower() in _PASSIVE_PARTICIPLES:
            ast.predicate = _PASSIVE_PARTICIPLES[w.lower()]
            break

    # Adjectives → manner (predicate adjectives for "be" constructions)
    if hasattr(parsed, 'adjectives') and parsed.adjectives:
        ast.manner = ' '.join(parsed.adjectives)

    # Agent (skip QW + agent descriptors + passive participles)
    agent_descs = _AGENT_DESCRIPTORS_MAP.get(lang, _AGENT_DESCRIPTORS_MAP['en'])
    _SKIP = _QW | agent_descs | set(_PASSIVE_PARTICIPLES)

    # Try parsed subjects first, but also handle pronouns
    if parsed.subjects:
        for s in parsed.subjects:
            if s.word.lower() not in _QW and s.word.lower() not in _ALL_QW:
                ast.agent = Entity(text=s.word, type='unknown')
                break

    if not ast.agent:
        from uli.memory import _PERSON_PRONOUNS
        for w, pos, lemma in tagged:
            if pos == 'PRON' and (w.lower() in _PERSON_PRONOUNS or w.lower() in ('they', 'it', 'this', 'that')):
                ast.agent = Entity(text=w, type='unknown')
                break
            if pos in ('PROPN', 'NOUN') and w.lower() not in _SKIP and w.lower() not in _ALL_QW:
                ast.agent = Entity(text=w, type='unknown')
                break

    # Patient
    if parsed.direct_objects:
        ast.patient = Entity(text=parsed.direct_objects[0].word, type='unknown')

    # Location from prep phrases
    for po in parsed.prep_objects:
        if po.prep in ('in', 'at', 'on'):
            ast.location = Entity(text=po.word, type='location')
            break

    # Entity list
    ast.entities = []
    for nf in (parsed.subjects + parsed.direct_objects + parsed.prep_objects):
        if nf.word and nf.word not in ast.entities:
            ast.entities.append(nf.word)

    # Question target
    qt_map = _QW_TO_TARGET.get(lang, _QW_TO_TARGET.get('en', {}))
    ast.question_target = qt_map.get((ast.question_word or '').lower(), 'patient')

    return ast


# ── Main Reasoner ─────────────────────────────────────────────────────────────

class GraphReasoner:
    """
    DMRSM-style state machine for KG-based multi-hop question answering.

    - Multilingual: detects language, uses lang-specific QW maps + agent descriptors.
    - Code switching: entities in any language resolve via cross-lang aliases.
    - Normalization: spelling correction + slang expansion.
    - Resilient: all exceptions caught, returns None instead of propagating.
    - Working memory: pronoun resolution across turns.
    """

    _QUESTION_WORDS = _ALL_QW  # never resolve these as pronoun references

    def __init__(self, db=None, memory: WorkingMemory = None,
                 mcp_client=None, system_config=None):
        self._db         = db or GraphDB()
        self._resolver   = EntityResolver(self._db)
        self.memory      = memory or WorkingMemory()
        self._normalizer = None   # lazy-loaded
        self._mcp        = mcp_client  # MCPClient instance or None
        self._config     = system_config  # SystemConfig instance or None

    @property
    def db(self):
        return self._db

    def _get_normalizer(self):
        if self._normalizer is None:
            from .normalizer import Normalizer
            self._normalizer = Normalizer()
        return self._normalizer

    # ── Reference resolution ──────────────────────────────────────────────────

    def _resolve_references(self, ast: MeaningAST) -> MeaningAST:
        """Substitute pronouns/demonstratives in AST slots using WorkingMemory."""
        def _sub(entity: Optional[Entity]) -> Optional[Entity]:
            if entity is None or not entity.text:
                return entity
            w = entity.text.lower()
            if w in self._QUESTION_WORDS:
                return entity
            if w not in _ALL_PRONOUNS:
                return entity
            resolved = self.memory.resolve_pronoun(entity.text)
            if resolved:
                surface, node_id = resolved
                node = self._db.node(node_id) or {}
                return Entity(text=surface, type=node.get('type', 'unknown'), id=node_id)
            return entity

        ast.agent   = _sub(ast.agent)
        ast.patient = _sub(ast.patient)
        ast.theme   = _sub(ast.theme)
        return ast

    # ── Public API ────────────────────────────────────────────────────────────

    def try_answer(self, question_text: str,
                   ast: Optional[MeaningAST] = None,
                   lang: Optional[str] = None,
                   ) -> Optional[Tuple[str, MeaningAST, List[dict]]]:
        """
        Run the DMRSM graph-reasoning loop. Returns (answer, ast, trace) or None.

        All exceptions are caught — returns None on any internal error.
        """
        try:
            return self._try_answer_impl(question_text, ast=ast, lang=lang)
        except Exception as e:
            log.warning("GraphReasoner error: %s: %s", type(e).__name__, e)
            return None

    def _try_answer_impl(self, question_text: str,
                         ast: Optional[MeaningAST] = None,
                         lang: Optional[str] = None,
                         ) -> Optional[Tuple[str, MeaningAST, List[dict]]]:
        trace: List[TraceStep] = []

        # ── Language detection ─────────────────────────────────────────────────
        if lang is None:
            try:
                from uli.lexer import detect_language
                lang = detect_language(question_text)
            except Exception:
                lang = 'en'
        if not lang:
            lang = 'en'

        # ── PARSE ──────────────────────────────────────────────────────────────
        trace.append(TraceStep('PARSE', note=f'lang={lang}: {question_text}'))

        # Normalize before parsing (spelling, slang)
        try:
            normalizer = self._get_normalizer()
            normalized = normalizer.normalize(question_text, lang=lang)
        except Exception:
            normalized = question_text

        from uli import tag, parse as uli_parse
        tagged = tag(normalized)
        parsed = uli_parse(normalized)

        if ast is None:
            ast = _parse_to_ast(tagged, parsed, lang=lang)

        # ── RESOLVE ────────────────────────────────────────────────────────────
        trace.append(TraceStep('RESOLVE', note='pronouns + entity mentions'))
        ast = self._resolve_references(ast)
        self._resolver.resolve_ast(ast, lang=lang)

        # ── DECOMPOSE ──────────────────────────────────────────────────────────
        trace.append(TraceStep('DECOMPOSE', note='detecting hop pattern'))
        agent_descs = _AGENT_DESCRIPTORS_MAP.get(lang, _AGENT_DESCRIPTORS_MAP['en'])
        steps = _decompose(tagged, parsed, ast, self._db, self._resolver,
                           lang=lang, agent_descriptors=agent_descs)
        if not steps:
            trace.append(TraceStep('FAILED', note='no matching hop pattern'))
            # MCP fallback — try external sources when KG has no answer
            mcp_result = self._mcp_fallback(question_text, lang=lang)
            if mcp_result:
                return mcp_result, ast, [t.as_dict() for t in trace]
            return None

        trace.append(TraceStep('DECOMPOSE',
                               note=f'{len(steps)} hop(s): ' +
                                    ', '.join(f'{s.direction}({s.source},{s.relation})' for s in steps)))

        # ── HOP ────────────────────────────────────────────────────────────────
        path: List[dict] = []
        intermediate_node: Optional[str] = None
        hop_count = 0

        for step_idx, step in enumerate(steps):
            hop_count += 1
            if hop_count > MAX_HOPS:
                raise RuntimeError(f"MAX_HOPS ({MAX_HOPS}) exceeded")

            # Determine source node ID
            source_id: Optional[str] = None
            if step.source == 'intermediate':
                source_id = intermediate_node
            elif step.source == 'agent' and ast.agent and ast.agent.id:
                source_id = ast.agent.id
            elif step.source == 'patient' and ast.patient and ast.patient.id:
                source_id = ast.patient.id
            elif step.source == 'theme' and ast.theme and ast.theme.id:
                source_id = ast.theme.id
            elif step.source == '_of_entity':
                source_id = step.__dict__.get('_of_node_id')
            else:
                # Last resort: scan all entities from AST
                for ent_text in ast.entities:
                    nid = self._resolver.resolve_text(ent_text, lang=lang)
                    if nid:
                        source_id = nid
                        break

            if not source_id:
                trace.append(TraceStep('HOP', note=f'no source for step {step_idx} ({step.source})'))
                trace.append(TraceStep('FAILED', note='dead end: missing source — trying MCP'))
                mcp_result = self._mcp_fallback(question_text, lang=lang)
                if mcp_result:
                    return mcp_result, ast, [t.as_dict() for t in trace]
                return None

            # Traverse
            if step.direction == 'forward':
                targets = self._db.traverse(source_id, step.relation)
            else:
                targets = self._db.reverse(source_id, step.relation)

            source_label = db_label(self._db, source_id, lang)
            trace.append(TraceStep(
                'HOP',
                node_id=source_id,
                relation=step.relation,
                result_id=targets[0] if targets else '',
                note=(f'{step.direction}({source_label}, {step.relation})'
                      f' → {db_label(self._db, targets[0], lang) if targets else "∅"}'),
            ))

            if not targets:
                trace.append(TraceStep('FAILED', note=f'dead end at ({source_id}, {step.relation}) — trying MCP'))
                mcp_result = self._mcp_fallback(question_text, lang=lang)
                if mcp_result:
                    return mcp_result, ast, [t.as_dict() for t in trace]
                return None

            result_id = targets[0]
            if step.direction == 'reverse':
                path.append({'from': result_id, 'relation': step.relation, 'to': source_id})
            else:
                path.append({'from': source_id, 'relation': step.relation, 'to': result_id})

            if step.target == 'intermediate':
                intermediate_node = result_id
            else:
                result_entity = Entity(
                    text=db_label(self._db, result_id, lang),
                    type=(self._db.node(result_id) or {}).get('type', 'unknown'),
                    id=result_id,
                )
                if step.target == 'agent':
                    ast.agent = result_entity
                elif step.target == 'patient':
                    ast.patient = result_entity
                elif step.target == 'location':
                    ast.location = result_entity
                elif step.target == 'time':
                    ast.time = result_entity

        # ── COMPILE ────────────────────────────────────────────────────────────
        answer_text, answer_ast = _compile(path, ast, self._db, lang=lang)
        trace.append(TraceStep('COMPILE', answer=answer_text, confidence=1.0))
        trace.append(TraceStep('DONE', answer=answer_text))

        # ── RECORD in working memory ───────────────────────────────────────────
        path_entities: List[Tuple] = []
        seen_nodes: set = set()
        for hop in path:
            for nid in (hop['from'], hop['to']):
                if nid not in seen_nodes:
                    seen_nodes.add(nid)
                    node = self._db.node(nid)
                    if node:
                        path_entities.append((
                            db_label(self._db, nid, lang),
                            nid,
                            node.get('type', 'unknown'),
                            lang,
                        ))
        self.memory.record(question_text, answer_text, path_entities)

        return answer_text, answer_ast, [t.as_dict() for t in trace]

    # ── MCP fallback (when KG has no answer) ─────────────────────────────────

    def _mcp_fallback(self, question_text: str,
                      lang: str = 'en') -> Optional[str]:
        """
        After KG miss, try external MCP sources in capability order.
        Returns answer text or None. Never raises.
        """
        if not self._mcp:
            return None
        try:
            from .router import classify, Intent
            intent = classify(question_text)
        except Exception:
            intent = None

        # Capability order per intent
        _CAP_MAP = {
            'coding':    ['documentation', 'web_search'],
            'research':  ['research', 'encyclopedia', 'web_search'],
            # Factual: encyclopedia (Wikipedia) first, then general web.
            # arXiv ('research') is intentionally excluded here — academic preprints
            # are irrelevant for general factual questions and return noisy results.
            'factual':   ['encyclopedia', 'web_search'],
            'math':      ['web_search'],
            'creative':  ['web_search'],
            'conversation': [],
        }
        intent_key = intent.value if intent else 'factual'
        capabilities = _CAP_MAP.get(intent_key, ['web_search'])

        for cap in capabilities:
            try:
                result = self._mcp.call_capability(cap, {'query': question_text})
                if result and result.strip():
                    log.info("MCP fallback: %s answered via capability=%s", intent_key, cap)
                    return result
            except Exception as e:
                log.debug("MCP cap=%s failed: %s", cap, e)
        return None

    # ── Generative mode (delegated to generator.py) ───────────────────────────

    def generate(self, topic: str, intent: str = 'explain',
                 lang: str = 'en', max_facts: int = 8) -> Optional[str]:
        """
        Generate text (essay/explain/brainstorm/timeline) about topic.

        Returns None if topic not in KG or generator unavailable.
        """
        try:
            node_id = self._resolver.resolve_text(topic, lang=lang)
            if not node_id:
                return None
            from .generator import ContentGenerator
            gen = ContentGenerator(self._db)
            return gen.generate(node_id, intent=intent, lang=lang, max_facts=max_facts)
        except Exception as e:
            log.warning("GraphReasoner.generate error: %s", e)
            return None


# ── DB label helper (works with both GraphDB and AdaptiveGraphDB) ─────────────

def db_label(db, node_id: str, lang: str = 'en') -> str:
    """Return display label for node_id, with lang support if available."""
    try:
        if _db_supports_lang_label(db):
            return db.label(node_id, lang)
        return db.label(node_id)
    except Exception:
        return node_id.replace('_', ' ')
