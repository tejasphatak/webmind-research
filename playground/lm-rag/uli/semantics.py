"""
ULI Semantics — Layer 3: Convert dependency parse → MeaningAST.
Extracts semantic roles, detects questions, identifies entities.
70% rules, 30% MiniLM cosine (for WSD only).
"""

import json
import os
from typing import List, Optional
from .protocol import Token, Entity, MeaningAST


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')


def _load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


# Question word → semantic role mapping
QUESTION_TARGET = {
    'who': 'agent',
    'whom': 'patient',
    'what': 'theme',
    'where': 'location',
    'when': 'time',
    'why': 'reason',
    'how': 'manner',
    'which': 'theme',
    'how many': 'quantity',
    'how much': 'quantity',
}


def tokens_to_ast(tokens: List[Token], text: str = '') -> MeaningAST:
    """Convert tokenized + parsed text into MeaningAST.
    This is the core semantic interpretation — all rule-based."""

    ast = MeaningAST(source=text)

    if not tokens:
        return ast

    # Detect language (majority vote on tokens)
    langs = {}
    for t in tokens:
        langs[t.lang] = langs.get(t.lang, 0) + 1
    ast.source_language = max(langs, key=langs.get) if langs else 'en'

    # Find root verb (the main predicate)
    root = _find_root(tokens)

    # Detect question
    q_word = _find_question_word(tokens)
    if q_word:
        ast.type = 'question'
        ast.question_word = q_word.lower()
        ast.question_target = QUESTION_TARGET.get(q_word.lower(), 'theme')
    elif text.rstrip().endswith('?'):
        ast.type = 'question'

    # Detect command/imperative
    if root and root.pos == 'VERB' and _is_imperative(tokens, root):
        ast.type = 'command'

    # Extract predicate
    if root:
        ast.predicate = root.lemma or root.text.lower()
        ast.tense = _detect_tense(tokens, root)

    # Extract semantic roles from dependencies
    for tok in tokens:
        if tok.dep in ('nsubj', 'nsubjpass'):
            if ast.type == 'question' and tok.text.lower() in ('who', 'what', 'which'):
                ast.agent = Entity(text='?', type='unknown')
            else:
                ast.agent = _make_entity(tok, tokens)
        elif tok.dep in ('dobj', 'obj'):
            ast.patient = _make_entity(tok, tokens)
        elif tok.dep in ('iobj',):
            ast.theme = _make_entity(tok, tokens)
        elif tok.dep in ('prep', 'obl') and _is_location_prep(tok, tokens):
            ast.location = _make_entity(tok, tokens)
        elif tok.dep in ('advmod', 'npadvmod') and tok.entity_type in ('DATE', 'TIME'):
            ast.time = Entity(text=tok.text, type='time')

    # Extract named entities
    ast.entities = _extract_entities(tokens)

    # Detect negation
    ast.negation = any(t.dep == 'neg' or t.text.lower() in ('not', "n't", 'no', 'never')
                       for t in tokens)

    # Detect nested clauses (for multi-hop)
    ast.sub_clauses = _extract_subclauses(tokens)

    # Classify intent
    ast.intent = _classify_intent(ast, tokens, text)

    return ast


def _find_root(tokens: List[Token]) -> Optional[Token]:
    """Find the root token (main verb)."""
    for tok in tokens:
        if tok.dep == 'ROOT':
            return tok
    # Fallback: first verb
    for tok in tokens:
        if tok.pos in ('VERB', 'AUX'):
            return tok
    return None


def _find_question_word(tokens: List[Token]) -> Optional[str]:
    """Find question word (who, what, where, when, why, how)."""
    for tok in tokens:
        if tok.text.lower() in QUESTION_TARGET:
            return tok.text
    return None


def _is_imperative(tokens: List[Token], root: Token) -> bool:
    """Check if sentence is imperative (command)."""
    # No subject before verb = imperative
    for tok in tokens:
        if tok.dep == 'nsubj' and tokens.index(tok) < tokens.index(root):
            return False
    return root.pos == 'VERB'


def _detect_tense(tokens: List[Token], root: Token) -> str:
    """Detect tense from auxiliary verbs."""
    for tok in tokens:
        if tok.dep == 'aux' and tok.head_idx == tokens.index(root):
            if tok.text.lower() in ('will', 'shall', "'ll"):
                return 'future'
            if tok.text.lower() in ('was', 'were', 'did', 'had'):
                return 'past'
    if root.text != root.lemma and root.text.endswith('ed'):
        return 'past'
    return 'present'


def _make_entity(tok: Token, tokens: List[Token]) -> Entity:
    """Create Entity from token, including its modifiers."""
    # Collect the full noun phrase
    phrase_parts = []
    idx = tokens.index(tok)

    # Collect dependents (modifiers, determiners, compounds)
    for t in tokens:
        if t.head_idx == idx and t.dep in ('amod', 'compound', 'det', 'nummod', 'poss'):
            phrase_parts.append((tokens.index(t), t.text))
    phrase_parts.append((idx, tok.text))
    phrase_parts.sort(key=lambda x: x[0])
    full_text = ' '.join(p[1] for p in phrase_parts)

    # Determine entity type
    etype = 'unknown'
    if tok.entity_type == 'PERSON':
        etype = 'person'
    elif tok.entity_type in ('GPE', 'LOC'):
        etype = 'place'
    elif tok.entity_type == 'ORG':
        etype = 'organization'
    elif tok.entity_type in ('DATE', 'TIME'):
        etype = 'time'
    elif tok.entity_type in ('CARDINAL', 'QUANTITY', 'MONEY'):
        etype = 'number'

    return Entity(text=full_text, type=etype)


def _is_location_prep(tok: Token, tokens: List[Token]) -> bool:
    """Check if a prepositional phrase indicates location."""
    # Look for preposition head
    if tok.head_idx >= 0 and tok.head_idx < len(tokens):
        head = tokens[tok.head_idx]
        if head.text.lower() in ('in', 'at', 'on', 'near', 'from', 'to', 'where', 'through'):
            return True
    return False


def _extract_entities(tokens: List[Token]) -> List[str]:
    """Extract named entities and content words as search terms."""
    entities = []
    seen = set()

    # Named entities first (highest priority)
    for tok in tokens:
        if tok.is_entity and tok.text.lower() not in seen:
            entities.append(tok.text)
            seen.add(tok.text.lower())

    # Content words (nouns, proper nouns, verbs — not stop words)
    stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'of', 'in',
                  'to', 'for', 'on', 'at', 'by', 'with', 'what', 'who',
                  'where', 'when', 'why', 'how', 'which', 'do', 'does', 'did'}
    for tok in tokens:
        if (tok.pos in ('NOUN', 'PROPN', 'VERB') and
            tok.text.lower() not in seen and
            tok.text.lower() not in stop_words and
            len(tok.text) > 2):
            entities.append(tok.lemma or tok.text)
            seen.add(tok.text.lower())

    return entities


def _extract_subclauses(tokens: List[Token]) -> List[MeaningAST]:
    """Extract relative clauses and subordinate clauses as sub-ASTs."""
    subclauses = []
    # Find relative clause markers (where, which, that, who as relative)
    for i, tok in enumerate(tokens):
        if tok.dep in ('relcl', 'advcl', 'acl'):
            # This token heads a subclause — extract its subtree
            subtree_tokens = _get_subtree(tokens, i)
            if len(subtree_tokens) > 2:
                sub_ast = tokens_to_ast(subtree_tokens, '')
                subclauses.append(sub_ast)
    return subclauses


def _get_subtree(tokens: List[Token], head_idx: int) -> List[Token]:
    """Get all tokens in the subtree rooted at head_idx."""
    result = [tokens[head_idx]]
    for i, tok in enumerate(tokens):
        if tok.head_idx == head_idx and i != head_idx:
            result.extend(_get_subtree(tokens, i))
    result.sort(key=lambda t: tokens.index(t))
    return result


def _classify_intent(ast: MeaningAST, tokens: List[Token], text: str) -> str:
    """Classify the intent/question type from AST structure."""
    text_lower = text.lower()

    # Math
    if any(c in text for c in '+-*/=%') or any(t.pos == 'NUM' for t in tokens):
        if any(w in text_lower for w in ('calculate', 'compute', 'percent', 'plus', 'minus', 'times', 'divided')):
            return 'math'

    # Multi-hop: nested clauses
    if ast.has_nested():
        return 'multi_hop'

    # Comparison
    if any(w in text_lower for w in ('compare', 'vs', 'versus', 'better', 'larger', 'smaller',
                                      'faster', 'which is', 'or')):
        return 'comparison'

    # Deep thought
    if any(w in text_lower for w in ('think about', 'analyze', 'implications', 'what would happen',
                                      'deep thought', 'what if')):
        return 'deep_thought'

    # Explanation
    if ast.question_word in ('why', 'how') and ast.type == 'question':
        return 'explanation'

    # Factual (default for questions)
    if ast.type == 'question':
        return 'factual'

    # Creative
    if any(w in text_lower for w in ('write', 'create', 'design', 'suggest', 'come up with')):
        return 'creative'

    return 'factual'
