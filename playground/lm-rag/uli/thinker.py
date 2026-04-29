"""
ULI Thinker — Inner monologue layer.

The missing piece: a thinking chain that reads text sentence by sentence,
builds context, detects knowledge gaps, generates questions, learns answers,
and grows the database.

Architecture:
    Text → Parser (encoder) → AST (hidden state) → Writer (decoder) → Text

The thinker orchestrates the loop:
    1. PARSE     — tokenize, POS tag, group compounds
    2. CONTEXT   — extract entities, build running context
    3. GAPS      — detect unknown compounds (things we don't understand)
    4. QUESTION  — generate context-aware questions from gaps
    5. LEARN     — retrieve answers, add to DB
    6. RESPOND   — compose response from accumulated understanding

Usage:
    from uli.thinker import Thinker
    thinker = Thinker(reasoner, wordnet_db_path)
    thought = thinker.think("Tell me about black holes")
    print(thought.response)
"""

import json
import re
import sqlite3
import logging
import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

log = logging.getLogger('uli.thinker')

# Default path for wordnet DB
_DEFAULT_DB = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    'data', 'vocab', 'wordnet.db'
)

# Data directory for language data
_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')


def _load_language_data():
    """Load thinker_data and writer_data from wordnet.db."""
    db_path = os.path.join(_DATA_DIR, 'vocab', 'wordnet.db')
    try:
        conn = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)
        cur = conn.cursor()
        cur.execute(
            "SELECT key, value FROM grammar_rules WHERE category = 'en' "
            "AND key IN ('thinker_data', 'writer_data')"
        )
        result = {}
        for key, value in cur.fetchall():
            try:
                result[key] = json.loads(value)
            except (json.JSONDecodeError, TypeError):
                result[key] = value
        conn.close()
        return result
    except (sqlite3.Error, OSError):
        return {}


_LANG_DATA = _load_language_data()

# ── Conversation patterns (loaded from wordnet.db) ──

def _load_conversation_data():
    """Load conversation patterns from wordnet.db."""
    db_path = os.path.join(_DATA_DIR, 'vocab', 'wordnet.db')
    try:
        conn = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)
        cur = conn.cursor()
        cur.execute(
            "SELECT value FROM grammar_rules WHERE category='en' AND key='conversation'"
        )
        row = cur.fetchone()
        conn.close()
        if row:
            return json.loads(row[0])
    except (sqlite3.Error, OSError, json.JSONDecodeError):
        pass
    return {}


_CONV = _load_conversation_data()
_TD = _LANG_DATA.get('thinker_data', {})

# All word lists loaded from DB (grammar_rules table).
# conversation data
_GREETINGS = frozenset(_CONV.get('greetings', []))
_GREETING_PHRASES = tuple(_CONV.get('greeting_phrases', []))
_FAREWELLS = frozenset(_CONV.get('farewells', []))
_FAREWELL_PHRASES = tuple(_CONV.get('farewell_phrases', []))
_ACKNOWLEDGMENTS = frozenset(_CONV.get('acknowledgments', []))
_ACK_PHRASES = tuple(_CONV.get('acknowledgment_phrases', []))
_INTRODUCTIONS = frozenset(_CONV.get('introductions', []))
_INTRO_PHRASES = tuple(_CONV.get('introduction_phrases', []))
_INTRO_RESPONSE = _CONV.get('introduction_response', 'Nice to meet you, {name}!')

# thinker data
_PRONOUNS = frozenset(_TD.get('pronouns', []))
_GENERIC_ADJ = frozenset(_TD.get('generic_adjectives', []))
_NOT_ENTITIES = frozenset(_TD.get('not_entities', []))
_NUMBERS = frozenset(_TD.get('number_words', []))
_REF_PRONOUNS = frozenset(_TD.get('ref_pronouns', []))
_NON_REF_PRONOUNS = frozenset(_TD.get('non_ref_pronouns', []))
_SKIP_PROPERTIES = frozenset(_TD.get('skip_properties', []))
_WHO_VERBS = frozenset(_TD.get('who_verbs', []))
_STOPWORDS = frozenset(_TD.get('stopwords', []))


# ── Data structures ──────────────────────────────────────────

@dataclass
class ConversationTurn:
    """One turn in conversation history."""
    user_input: str
    response: str
    topic: str
    entities: List[str]


@dataclass
class KnowledgeGap:
    """Something the system doesn't understand."""
    compound: str          # The unknown compound ("event horizon")
    context_topic: str     # What we're currently reading about ("black holes")
    source_sentence: str   # The sentence that triggered this gap
    question: str          # Generated question ("What is an event horizon?")
    answered: bool = False
    answer: str = ''


@dataclass
class Thought:
    """Accumulated context from the thinking chain."""
    raw_input: str

    # Parse
    tagged: list = field(default_factory=list)
    grouped: list = field(default_factory=list)   # After compound grouping
    structure: object = None  # SentenceStructure from grammar engine

    # Context
    topic: str = ''
    entities: List[str] = field(default_factory=list)
    definitions: Dict[str, str] = field(default_factory=dict)
    facts: List[Tuple[str, str, str]] = field(default_factory=list)

    # Gaps
    gaps: List[KnowledgeGap] = field(default_factory=list)

    # Resolution (for conversation)
    references: Dict[str, str] = field(default_factory=dict)
    resolved_input: str = ''
    response_type: str = ''  # greeting, farewell, acknowledgment, factual

    # Response
    response: str = ''


# ── Thinker ──────────────────────────────────────────────────

class Thinker:
    """
    Inner monologue — reads text, builds understanding, detects gaps, learns.

    Maintains:
    - Conversation history (for pronoun resolution)
    - Running context (topic, entities, definitions)
    - WordNet compound DB (grows as it learns)
    """

    def __init__(self, reasoner=None, db_path=None, dictionary=None,
                 auto_search=True):
        self._db_path = db_path or _DEFAULT_DB
        self._conn = None
        self._dictionary = dictionary  # DictionaryProvider instance
        # Auto-wire web search if no reasoner provided
        if reasoner is not None:
            self._reasoner = reasoner
        elif auto_search:
            self._reasoner = self._build_default_reasoner()
        else:
            self._reasoner = None
        self._history: List[ConversationTurn] = []
        self._context = {
            'topic': '',
            'entities': {},         # name → {mentions, first_seen}
            'definitions': {},      # name → definition string
            'facts': [],            # (subject, predicate, object) triples
        }

    def _build_default_reasoner(self):
        """Auto-construct a reasoner with web search from config.yaml."""
        try:
            from .reasoner import GraphReasoner
            from .mcp_client import MCPClient
            from .system_prompt import SystemConfig
            cfg = SystemConfig.load()
            mcp = MCPClient(cfg.settings.mcp_servers)
            return GraphReasoner(mcp_client=mcp, system_config=cfg)
        except Exception as e:
            log.warning("Could not auto-wire reasoner: %s", e)
            return None

    @property
    def context(self):
        return self._context

    @property
    def history(self):
        return self._history

    def _get_conn(self):
        if self._conn is None and os.path.exists(self._db_path):
            self._conn = sqlite3.connect(self._db_path)
        return self._conn

    # ── Concept rules (learned behaviors) ──────────────

    def _check_rules(self, text: str) -> Optional[str]:
        """Check if any learned rules apply to this input.

        Returns transformed input if a rule fired, None otherwise.
        Rules are patterns the thinker learned from concepts —
        e.g., "if input looks like a URL, fetch its content."

        Rules are sorted by priority (highest first). More specific
        rules override more general ones.
        """
        conn = self._get_conn()
        if not conn:
            return None

        cur = conn.cursor()
        cur.execute(
            'SELECT rule_id, condition, action, priority '
            'FROM concept_rules ORDER BY priority DESC, uses DESC'
        )
        for rule_id, condition, action, priority in cur.fetchall():
            if self._condition_matches(condition, text):
                result = self._execute_rule(action, text)
                if result:
                    # Rule fired — update usage count + bump confidence
                    cur.execute(
                        'UPDATE concept_rules SET uses = uses + 1, '
                        'confidence = MIN(1.0, confidence + 0.05) '
                        'WHERE rule_id = ?',
                        (rule_id,)
                    )
                    conn.commit()
                    log.info("Rule fired: %s (gen=%d) on '%s'",
                             rule_id, priority, text[:40])
                    return result
        return None

    def _condition_matches(self, condition: str, text: str) -> bool:
        """Check if a rule condition matches the input text.

        Conditions are simple patterns stored in the DB:
        - 'starts_with:http' — text starts with 'http'
        - 'contains:@' — text contains '@'
        - 'pattern:...' — regex pattern
        - 'pos_sequence:NOUN VERB' — POS tag sequence
        """
        if ':' not in condition:
            return condition.lower() in text.lower()

        ctype, cvalue = condition.split(':', 1)
        if ctype == 'starts_with':
            return text.lower().startswith(cvalue.lower())
        if ctype == 'contains':
            return cvalue in text
        if ctype == 'pattern':
            return bool(re.search(cvalue, text))
        if ctype == 'ends_with':
            return text.lower().endswith(cvalue.lower())
        return False

    def _execute_rule(self, action: str, text: str) -> Optional[str]:
        """Execute a rule action on the input text.

        Actions can be:
        1. Legacy labels: 'fetch_content', 'read_content' (backward compat)
        2. Python expressions: executed in a sandboxed namespace with access to:
           - thinker: this Thinker instance
           - text: the input text
           - re: the re module
           - result: dict to put output in (set result['output'] = ...)

        The action IS the code. When the system learns a rule, it writes
        the action as a Python snippet. This way the system's behavior
        grows from data, not from hardcoded if/else branches.
        """
        # Legacy string labels (for backward compatibility with existing rules)
        if action == 'fetch_content':
            return self._fetch_url_content(text.strip())
        if action == 'read_content':
            content = self._fetch_url_content(text.strip())
            if content:
                self.read_text(content, resolve_gaps=True)
                return content
            return None

        # Python expression — execute in sandboxed namespace
        if action.startswith('py:'):
            code = action[3:]
            return self._run_action_code(code, text)

        # If it looks like Python (has assignments, function calls, etc.)
        if any(marker in action for marker in ('=', '(', 'thinker.', 'result')):
            return self._run_action_code(action, text)

        return None

    def _run_action_code(self, code: str, text: str) -> Optional[str]:
        """Execute a rule's action code in a sandboxed namespace.

        The namespace gives the action access to:
        - thinker: this Thinker instance (can call _fetch_url_content, read_text, etc.)
        - text: the input that triggered the rule
        - re: regex module
        - result: dict — set result['output'] to return a value

        Example action code:
          content = thinker._fetch_url_content(text.strip())
          if content:
              thinker.read_text(content, resolve_gaps=True)
              result['output'] = content
        """
        namespace = {
            'thinker': self,
            'text': text,
            're': re,
            'result': {'output': None},
            'log': log,
        }
        try:
            exec(code, {'__builtins__': {}}, namespace)
            return namespace['result'].get('output')
        except Exception as e:
            log.warning("Rule action failed: %s — %s", code[:60], e)
            return None

    def _fetch_url_content(self, url: str) -> Optional[str]:
        """Fetch content from a URL. Used by rules that handle URLs."""
        import urllib.request
        import json

        # Wikipedia URL → use API for clean text
        wiki_match = re.match(
            r'https?://(\w+)\.wikipedia\.org/wiki/(.+)', url
        )
        if wiki_match:
            lang = wiki_match.group(1)
            title = urllib.parse.unquote(wiki_match.group(2))
            try:
                params = urllib.parse.urlencode({
                    'action': 'query', 'titles': title, 'prop': 'extracts',
                    'explaintext': 1, 'exlimit': 1, 'format': 'json',
                })
                req = urllib.request.Request(
                    f'https://{lang}.wikipedia.org/w/api.php?{params}',
                    headers={'User-Agent': 'LM-RAG/1.0'}
                )
                with urllib.request.urlopen(req, timeout=15) as r:
                    data = json.loads(r.read())
                    pages = data.get('query', {}).get('pages', {})
                    for pid, page in pages.items():
                        extract = page.get('extract', '')
                        if extract:
                            return extract[:5000]  # Cap at 5K chars
            except Exception as e:
                log.warning("Failed to fetch %s: %s", url, e)
            return None

        # Generic URL — try to fetch
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'LM-RAG/1.0'})
            with urllib.request.urlopen(req, timeout=15) as r:
                content = r.read().decode('utf-8', errors='ignore')
                # Strip HTML tags (basic)
                content = re.sub(r'<[^>]+>', ' ', content)
                content = re.sub(r'\s+', ' ', content).strip()
                return content[:5000]
        except Exception as e:
            log.warning("Failed to fetch %s: %s", url, e)
        return None

    def _learn_rule(self, concept: str, definition: str):
        """After learning a concept, check if it implies a behavioral rule.

        A concept becomes a rule when the definition describes
        HOW to process something, not just WHAT something is.

        The action is generated as Python code — the system writes its
        own behavior. Rules can mutate: if a new concept refines an
        existing rule's domain, the old rule gets a child that overrides it
        for the more specific case.

        Examples:
        - "URL = web address" → rule: if starts_with:http → fetch + read
        - "Wikipedia = online encyclopedia" → specializes URL rule:
          if contains:wikipedia.org → fetch via API (cleaner text)
        - "email = electronic message" → rule: if contains:@ → parse as address
        """
        conn = self._get_conn()
        if not conn:
            return

        defn_lower = definition.lower()
        cur = conn.cursor()

        # ── Detect what kind of rule this concept implies ──

        # Reference/pointer → fetch content
        ref_patterns = [
            'reference to', 'pointer to', 'web link', 'web address',
            'network address', 'internet address', 'uniform resource',
            'hyperlink', 'hypertext',
        ]
        is_reference = any(p in defn_lower for p in ref_patterns)

        # Notation/format → parse specially
        # Use multi-word patterns to avoid false positives
        # ("Hindi-language film" should NOT trigger a format rule)
        format_patterns = [
            'notation', 'file format', 'encoding', 'data format',
            'markup language', 'programming language', 'syntax',
            'serialization', 'data representation',
        ]
        is_format = any(p in defn_lower for p in format_patterns)

        # Protocol → communication/transport rule
        protocol_patterns = ['protocol', 'standard', 'specification']
        is_protocol = any(p in defn_lower for p in protocol_patterns)

        if not (is_reference or is_format or is_protocol):
            return  # This concept is just a fact, not a rule

        # ── Derive condition from the concept ──
        # The condition should match INSTANCES of the concept, not
        # the concept word itself. "URL" → match things that look like
        # URLs (start with http), not things containing the word "url".

        concept_lower = concept.lower()

        if concept_lower.startswith('http'):
            condition = 'starts_with:http'
        elif '@' in concept:
            condition = 'contains:@'
        elif '.' in concept_lower and len(concept_lower.split('.')) >= 2:
            # Domain-like: wikipedia.org, github.com
            condition = f'contains:{concept_lower}'
        elif is_reference and any(w in defn_lower for w in
                                   ('web', 'internet', 'http', 'online',
                                    'resource', 'locator', 'uri', 'url')):
            # The concept describes a web reference (URL, URI, link, etc.)
            # → match things that start with http
            condition = 'starts_with:http'
        elif is_reference and 'email' in defn_lower:
            condition = 'contains:@'
        elif is_reference and 'file' in defn_lower:
            condition = 'starts_with:/'
        else:
            condition = f'contains:{concept_lower}'

        # ── Generate action as Python code ──

        if is_reference:
            action_code = (
                "content = thinker._fetch_url_content(text.strip())\n"
                "if content:\n"
                "    thinker.read_text(content, resolve_gaps=True)\n"
                "    result['output'] = content\n"
            )
        elif is_format:
            # Format rules: read the text through the thinker as-is
            # (future: could add format-specific parsers)
            action_code = (
                "thinker.read_text(text, resolve_gaps=True)\n"
                "result['output'] = text\n"
            )
        elif is_protocol:
            # Protocol — similar to reference but may need different handling
            action_code = (
                "content = thinker._fetch_url_content(text.strip())\n"
                "if content:\n"
                "    result['output'] = content\n"
            )
        else:
            return

        # ── Check for existing rules to mutate/override ──

        rule_id = f'rule_{self._node_id(concept)}'
        parent_rule = None
        generation = 0
        priority = 0

        # Find existing rules with overlapping conditions
        cur.execute('SELECT rule_id, condition, generation, priority FROM concept_rules')
        existing_rules = cur.fetchall()

        for existing_id, existing_cond, existing_gen, existing_pri in existing_rules:
            overlap = self._conditions_overlap(condition, existing_cond)
            if overlap == 'specializes':
                # New rule is more specific → becomes child, higher priority
                parent_rule = existing_id
                generation = (existing_gen or 0) + 1
                priority = (existing_pri or 0) + 1
                log.info("Rule %s specializes %s (gen %d → %d)",
                         rule_id, existing_id, existing_gen or 0, generation)
                break
            elif overlap == 'same':
                # Same condition → update the existing rule (mutate in place)
                cur.execute(
                    'UPDATE concept_rules SET action=?, learned_from=?, '
                    'generation=generation+1, confidence=0.5 WHERE rule_id=?',
                    (action_code,
                     f'{concept}: {definition[:100]}',
                     existing_id)
                )
                conn.commit()
                log.info("Mutated rule %s with new action from %s",
                         existing_id, concept)
                self._add_to_graph(concept, 'mutated_rule', existing_id)
                return
            elif overlap == 'generalizes':
                # New rule is MORE general → existing rule already handles it
                # Don't add a less specific rule that would override
                log.info("Skipping rule %s — existing %s is more specific",
                         rule_id, existing_id)
                return

        # ── Insert new rule ──

        cur.execute(
            'INSERT OR REPLACE INTO concept_rules '
            '(rule_id, condition, action, learned_from, priority, '
            'parent_rule, generation) VALUES (?,?,?,?,?,?,?)',
            (rule_id, condition, action_code,
             f'{concept}: {definition[:100]}',
             priority, parent_rule, generation)
        )
        conn.commit()
        log.info("Learned rule: %s → [python code] (from %s, gen=%d, pri=%d)",
                 condition, concept, generation, priority)

        # Store in graph
        self._add_to_graph(concept, 'implies_rule', rule_id)
        if parent_rule:
            self._add_to_graph(rule_id, 'specializes', parent_rule)

    def _conditions_overlap(self, cond_a: str, cond_b: str) -> str:
        """Check how two rule conditions relate.

        Returns:
        - 'same' — identical conditions
        - 'specializes' — cond_a is more specific than cond_b
        - 'generalizes' — cond_a is more general than cond_b
        - 'disjoint' — no overlap
        """
        if cond_a == cond_b:
            return 'same'

        # Parse conditions
        def parse_cond(c):
            if ':' in c:
                return c.split(':', 1)
            return ('contains', c)

        type_a, val_a = parse_cond(cond_a)
        type_b, val_b = parse_cond(cond_b)

        # starts_with:http specializes starts_with:http (same)
        # contains:wikipedia.org specializes starts_with:http
        # starts_with:https specializes starts_with:http
        if type_a == type_b:
            if val_a == val_b:
                return 'same'
            if val_a.startswith(val_b):
                return 'specializes'
            if val_b.startswith(val_a):
                return 'generalizes'
        elif type_a == 'contains' and type_b == 'starts_with':
            # contains:wikipedia.org is more specific for URLs
            # (only matches wikipedia URLs, not all http URLs)
            if val_b in ('http', 'https'):
                return 'specializes'
        elif type_a == 'starts_with' and type_b == 'contains':
            if val_a in ('http', 'https'):
                return 'generalizes'

        return 'disjoint'

    # ── Graph operations (facts as edges in DB) ────────

    def _node_id(self, text: str) -> str:
        """Normalize text to a graph node ID.

        Resolution order:
        1. Try synset lookup (WordNet concept hash) — language-independent
        2. Fall back to normalized text ID (lowercase, singularized)

        Synset IDs like 'dog.n.01' are the same across all languages,
        so edges learned in English are queryable from Marathi.
        """
        # Try synset resolution first
        synset = self._to_synset(text)
        if synset:
            return synset

        # Fallback: normalized text
        nid = text.lower().strip().replace(' ', '_')
        parts = nid.split('_')
        last = parts[-1]
        if not any(last.endswith(suf) for suf in ('ous', 'ss', 'us', 'is', 'ness')):
            if last.endswith('ies') and len(last) > 4:
                parts[-1] = last[:-3] + 'y'
            elif last.endswith('ses') or last.endswith('xes') or last.endswith('zes'):
                parts[-1] = last[:-2]
            elif last.endswith('s') and len(last) > 3:
                parts[-1] = last[:-1]
        return '_'.join(parts)

    @lru_cache(maxsize=4096)
    def _to_synset(self, text: str) -> str:
        """Try to resolve text to a WordNet synset ID.

        'dog' → 'dog.n.01', 'weather' → 'weather.n.01'
        Multi-word: 'black hole' → tries compound, then head noun.
        Returns '' if no synset found (proper nouns, unknown words).

        Cached: synset mappings don't change during a session.
        """
        conn = self._get_conn()
        if not conn:
            return ''
        cur = conn.cursor()
        clean = text.lower().strip()

        # Skip intent/meta nodes (they're not WordNet concepts)
        if clean.endswith('_intent') or clean in ('intent_disambiguation',):
            return ''

        # Try exact match first
        cur.execute(
            'SELECT synset_id FROM senses WHERE word=? AND sense_num=(SELECT MAX(sense_num) FROM senses WHERE word=?) LIMIT 1',
            (clean, clean)
        )
        row = cur.fetchone()
        if row:
            return row[0]

        # Multi-word: try head noun (last word)
        words = clean.split()
        if len(words) > 1:
            head = words[-1]
            cur.execute(
                'SELECT synset_id FROM senses WHERE word=? AND sense_num=(SELECT MAX(sense_num) FROM senses WHERE word=?) LIMIT 1',
                (head, head)
            )
            row = cur.fetchone()
            if row:
                return row[0]

        # No synset found — generate a concept ID for novel concepts.
        # Format: word.n.00 (00 = learned, not from WordNet)
        # This gets written to senses so future lookups find it.
        normalized = clean.replace(' ', '_')
        concept_id = f'{normalized}.n.00'
        cur.execute(
            'INSERT OR IGNORE INTO senses (word, pos, synset_id, sense_num, definition) '
            'VALUES (?, ?, ?, 0, ?)',
            (clean, 'NOUN', concept_id, f'learned concept: {clean}')
        )
        conn.commit()
        return concept_id

    def _add_to_graph(self, subject: str, relation: str, obj: str,
                      source_type: str = 'thinker', source_url: str = None,
                      source_text: str = None):
        """Add a fact as an edge in the knowledge graph (DB-backed).

        Provenance is stored IN the graph itself:
        - The edge gets an edge_id (from__rel__to)
        - A source node is created (type='source')
        - A provenance edge links: edge_id --sourced_from--> source_node

        Confidence is NOT a stored number — it's the degree of the
        edge node. More source edges pointing to a fact = stronger fact.
        Call _edge_confidence(edge_id) to compute it from topology.

        If a CONTRADICTING fact exists (same subject+relation, different object),
        both are kept but the contradiction is recorded as an edge.
        """
        conn = self._get_conn()
        if not conn:
            return

        # Minimal filters — only skip empty and pronouns
        if not subject or not subject.strip():
            return
        if subject.lower().strip() in _PRONOUNS or subject.lower().strip() in _REF_PRONOUNS:
            return

        # Cap subject/object length
        if len(subject.split()) > 6:
            subject = ' '.join(subject.split()[-4:])
        if obj and len(obj.split()) > 8:
            obj = ' '.join(obj.split()[:8])

        cur = conn.cursor()
        s_id = self._node_id(subject)
        o_id = self._node_id(obj) if obj else ''

        # Ensure nodes exist
        cur.execute(
            'INSERT OR IGNORE INTO graph_nodes (node_id, label, node_type) VALUES (?,?,?)',
            (s_id, subject, 'entity')
        )
        if not o_id:
            conn.commit()
            return

        cur.execute(
            'INSERT OR IGNORE INTO graph_nodes (node_id, label, node_type) VALUES (?,?,?)',
            (o_id, obj, 'entity')
        )

        edge_id = f'{s_id}__{relation}__{o_id}'

        # Insert edge if new (IGNORE if exists — we'll add provenance either way)
        cur.execute(
            'INSERT OR IGNORE INTO graph_edges '
            '(from_id, relation, to_id, edge_id, source) VALUES (?,?,?,?,?)',
            (s_id, relation, o_id, edge_id, source_type)
        )

        # Check for contradictions — but only if this relation is
        # FUNCTIONAL (single-valued). We check the graph for a relation
        # node that declares multiplicity. If no declaration exists yet,
        # we learn it: compute MAX fanout across all nodes using this relation.
        # High max fanout = multi-valued = no contradiction.
        cur.execute(
            'SELECT COUNT(*) FROM graph_edges '
            'WHERE from_id=? AND relation=? AND to_id!=?',
            (s_id, relation, o_id)
        )
        other_count = cur.fetchone()[0]
        if other_count > 0 and self._is_functional_relation(relation, cur):
            cur.execute(
                'SELECT to_id, edge_id FROM graph_edges '
                'WHERE from_id=? AND relation=? AND to_id!=?',
                (s_id, relation, o_id)
            )
            for contra_to, contra_eid in cur.fetchall():
                cur.execute(
                    'INSERT OR IGNORE INTO graph_edges '
                    '(from_id, relation, to_id, edge_id, source) VALUES (?,?,?,?,?)',
                    (edge_id, 'contradicts',
                     contra_eid or f'{s_id}__{relation}__{contra_to}',
                     f'{edge_id}__contradicts__{contra_eid}', 'system')
                )
                log.warning("Contradiction: %s --%s--> %s vs %s",
                            subject, relation, obj, contra_to)

        # Record provenance — source node + sourced_from edge
        # This is what gives the fact its strength:
        # more source edges → higher confidence (computed, not stored)
        self._record_provenance(edge_id, source_type, source_url,
                                source_text, cur)

        conn.commit()

    def _record_provenance(self, edge_id: str, source_type: str,
                           source_url: str, source_text: str, cur):
        """Record where a fact came from — as nodes and edges in the graph.

        Source node: type='source', label describes the source
        Provenance edge: edge_id --sourced_from--> source_node

        Every provenance edge is a vote of confidence. The fact's
        strength = number of sourced_from edges it has.
        """
        # Create a source node ID from the URL or type
        if source_url:
            # Stable ID from URL so same URL = same source node
            src_id = f'src_{self._node_id(source_url[:80])}'
            label = f'{source_type}: {source_url[:100]}'
        else:
            # Stable from content — same extraction = same source
            import hashlib
            h = hashlib.md5(f'{source_type}:{source_text or ""}'.encode()).hexdigest()[:8]
            src_id = f'src_{source_type}_{h}'
            label = f'{source_type}: {(source_text or "")[:80]}'

        cur.execute(
            'INSERT OR IGNORE INTO graph_nodes (node_id, label, node_type) VALUES (?,?,?)',
            (src_id, label, 'source')
        )

        # Store raw text on the source node
        if source_text:
            import json
            cur.execute(
                'UPDATE graph_nodes SET aliases=? WHERE node_id=?',
                (json.dumps([source_text[:500]]), src_id)
            )

        # Link: edge --sourced_from--> source
        prov_edge_id = f'{edge_id}__sourced_from__{src_id}'
        cur.execute(
            'INSERT OR IGNORE INTO graph_edges '
            '(from_id, relation, to_id, edge_id, source) VALUES (?,?,?,?,?)',
            (edge_id, 'sourced_from', src_id, prov_edge_id, 'system')
        )

    def _edge_confidence(self, edge_id: str, epsilon: float = 1e-7) -> float:
        """Confidence of a fact = activation over source edge count.

        confidence = n / (n + ε)

        n = number of sourced_from edges. ε is small (1e-7) — just
        enough to avoid division by zero. Even 1 source gives near-1.0
        confidence because a sourced fact IS a real fact. The number of
        sources is the signal, not a probability curve.

        With ε=1e-7:
          0 sources → 0.0
          1 source  → ~1.0
          n sources → ~1.0

        The REAL differentiation between facts comes from edge count
        itself (available via the raw query), not from squashing it
        into [0,1]. ε could eventually be learned per-relation.
        """
        conn = self._get_conn()
        if not conn:
            return 0.0

        cur = conn.cursor()
        cur.execute(
            'SELECT COUNT(*) FROM graph_edges '
            'WHERE from_id=? AND relation=?',
            (edge_id, 'sourced_from')
        )
        n = cur.fetchone()[0]
        return n / (n + epsilon)

    def _is_functional_relation(self, relation: str, cur=None) -> bool:
        """Is this relation single-valued (functional)?

        Learned from graph topology: compute the MAX fanout of this
        relation across all nodes. If any node has 3+ outgoing edges
        with this relation, it's clearly multi-valued.

        "is" → max fanout 20+ → multi-valued → no contradictions
        "born_in" → max fanout 1 → functional → contradictions matter
        """
        conn = self._get_conn()
        if not conn:
            return False
        if cur is None:
            cur = conn.cursor()

        cur.execute(
            'SELECT MAX(cnt) FROM ('
            '  SELECT COUNT(*) as cnt FROM graph_edges '
            '  WHERE relation=? AND source != ? '
            '  GROUP BY from_id'
            ')', (relation, 'system')
        )
        row = cur.fetchone()
        max_fanout = row[0] if row and row[0] else 0

        # If the max fanout across the whole graph is ≤ 2, it's likely functional
        return max_fanout <= 2

    def _edge_contradictions(self, edge_id: str) -> List[str]:
        """Find all edges that contradict this one."""
        conn = self._get_conn()
        if not conn:
            return []

        cur = conn.cursor()
        cur.execute(
            'SELECT to_id FROM graph_edges '
            'WHERE from_id=? AND relation=?',
            (edge_id, 'contradicts')
        )
        return [r[0] for r in cur.fetchall()]

    def _add_inference(self, subject: str, relation: str, obj: str,
                       premise_edges: List[str]):
        """Add an inferred fact with its reasoning chain as provenance.

        The inference itself becomes a source node (type='inference'),
        and the premise edges are linked to it via 'premise' edges.
        The inferred fact gets a 'sourced_from' edge to the inference node.

        So the full chain is:
          fact --sourced_from--> inference_node --premise--> premise_fact_1
                                               --premise--> premise_fact_2
        """
        conn = self._get_conn()
        if not conn:
            return

        cur = conn.cursor()
        s_id = self._node_id(subject)
        o_id = self._node_id(obj) if obj else ''
        edge_id = f'{s_id}__{relation}__{o_id}'

        # Create inference node
        import hashlib
        h = hashlib.md5('|'.join(premise_edges).encode()).hexdigest()[:8]
        inf_id = f'inf_{h}'

        cur.execute(
            'INSERT OR IGNORE INTO graph_nodes (node_id, label, node_type) VALUES (?,?,?)',
            (inf_id, f'inferred: {subject} {relation} {obj}', 'inference')
        )

        # Add the inferred fact (provenance will be the inference node)
        self._add_to_graph(subject, relation, obj, source_type='inference')

        # Link premises to inference node
        for premise_eid in premise_edges:
            premise_link_id = f'{inf_id}__premise__{premise_eid}'
            cur.execute(
                'INSERT OR IGNORE INTO graph_edges '
                '(from_id, relation, to_id, edge_id, source) '
                'VALUES (?,?,?,?,?)',
                (inf_id, 'premise', premise_eid, premise_link_id, 'system')
            )

        # Link the inferred edge to its inference source
        cur.execute(
            'INSERT OR IGNORE INTO graph_edges '
            '(from_id, relation, to_id, edge_id, source) VALUES (?,?,?,?,?)',
            (edge_id, 'sourced_from', inf_id,
             f'{edge_id}__sourced_from__{inf_id}', 'system')
        )

        conn.commit()

    def _query_graph(self, node: str, relation: str = None) -> List[Tuple[str, str, str]]:
        """Query the graph for facts about a node.

        Returns [(relation, target_label, direction), ...]
        If relation is specified, only returns edges with that relation.
        """
        conn = self._get_conn()
        if not conn:
            return []

        cur = conn.cursor()
        n_id = self._node_id(node)
        results = []

        # Outgoing edges
        if relation:
            cur.execute(
                'SELECT e.relation, n.label FROM graph_edges e '
                'JOIN graph_nodes n ON e.to_id = n.node_id '
                'WHERE e.from_id = ? AND e.relation = ?',
                (n_id, relation)
            )
        else:
            cur.execute(
                'SELECT e.relation, n.label FROM graph_edges e '
                'JOIN graph_nodes n ON e.to_id = n.node_id '
                'WHERE e.from_id = ?',
                (n_id,)
            )
        for rel, label in cur.fetchall():
            results.append((rel, label, 'out'))

        # Incoming edges
        if relation:
            cur.execute(
                'SELECT e.relation, n.label FROM graph_edges e '
                'JOIN graph_nodes n ON e.from_id = n.node_id '
                'WHERE e.to_id = ? AND e.relation = ?',
                (n_id, relation)
            )
        else:
            cur.execute(
                'SELECT e.relation, n.label FROM graph_edges e '
                'JOIN graph_nodes n ON e.from_id = n.node_id '
                'WHERE e.to_id = ?',
                (n_id,)
            )
        for rel, label in cur.fetchall():
            results.append((rel, label, 'in'))

        return results

    def _query_graph_with_inference(self, node: str, relation: str = None,
                                     max_depth: int = 3) -> List[Tuple[str, str, str]]:
        """Query graph with is_a chain inference.

        If 'dog --is--> animal' and 'animal --can--> breathe',
        then querying 'dog' for 'can' returns 'breathe' (inherited).

        Walks is_a edges up to max_depth to collect inherited properties.
        Direct edges are returned first (higher priority).
        """
        # Direct results first
        direct = self._query_graph(node, relation)
        if direct:
            return direct

        # Walk is_a chain upward: node → parent → grandparent → ...
        conn = self._get_conn()
        if not conn:
            return []

        cur = conn.cursor()
        visited = set()
        current = self._node_id(node)
        inherited = []

        for depth in range(max_depth):
            if current in visited:
                break
            visited.add(current)

            # Find is_a parent: current --is--> parent
            cur.execute(
                'SELECT n.label, n.node_id FROM graph_edges e '
                'JOIN graph_nodes n ON e.to_id = n.node_id '
                'WHERE e.from_id = ? AND e.relation = ?',
                (current, 'is')
            )
            parents = cur.fetchall()
            if not parents:
                break

            for parent_label, parent_id in parents:
                # Query the parent for the requested relation
                parent_results = self._query_graph(parent_label, relation)
                for rel, label, direction in parent_results:
                    if direction == 'out' and (rel, label) not in \
                            [(r, l) for r, l, _ in inherited]:
                        inherited.append((rel, label, 'out'))

            # Move up: take the first parent as the chain
            current = parents[0][1]

        return inherited

    def _graph_stats(self) -> dict:
        """Return graph size."""
        conn = self._get_conn()
        if not conn:
            return {'nodes': 0, 'edges': 0}
        cur = conn.cursor()
        cur.execute('SELECT COUNT(*) FROM graph_nodes')
        nodes = cur.fetchone()[0]
        cur.execute('SELECT COUNT(*) FROM graph_edges')
        edges = cur.fetchone()[0]
        return {'nodes': nodes, 'edges': edges}

    # ── Step 1: PARSE ──────────────────────────────────

    def _parse(self, t: Thought) -> Thought:
        """Tokenize, POS tag, group compounds, and parse sentence structure."""
        from uli import tag, parse
        t.tagged = tag(t.raw_input)
        t.grouped = self._group_compounds(t.tagged)
        # Full grammatical parse — subjects, verb, objects, etc.
        try:
            t.structure = parse(t.raw_input)
        except Exception:
            t.structure = None
        return t

    @lru_cache(maxsize=8192)
    def _lookup_compound(self, compound: str):
        """Check if a string is a known compound in the DB. Cached."""
        conn = self._get_conn()
        if not conn:
            return None
        cur = conn.cursor()
        cur.execute(
            'SELECT compound, pos FROM compounds WHERE compound=? AND pos != ? LIMIT 1',
            (compound, 'REJECTED')
        )
        return cur.fetchone()

    @lru_cache(maxsize=8192)
    def _word_exists(self, word: str, pos: str = None) -> bool:
        """Check if a word exists in the words table. Cached."""
        conn = self._get_conn()
        if not conn:
            return False
        cur = conn.cursor()
        if pos:
            cur.execute('SELECT 1 FROM words WHERE word=? AND pos=? LIMIT 1', (word, pos))
        else:
            cur.execute('SELECT 1 FROM words WHERE word=? LIMIT 1', (word,))
        return cur.fetchone() is not None

    def _group_compounds(self, tagged):
        """Group consecutive tokens that form known compounds from the DB.

        Tries both surface forms and lemmatized forms so
        'black holes' matches compound 'black hole'.
        """
        conn = self._get_conn()
        if not conn:
            return tagged

        result = []
        i = 0
        while i < len(tagged):
            matched = False
            for length in range(min(4, len(tagged) - i), 1, -1):
                # Try surface form first
                candidate = ' '.join(
                    tagged[j][0].lower() for j in range(i, i + length)
                )
                # Also try lemmatized form (handles plurals: "black holes" → "black hole")
                candidate_lemma = ' '.join(
                    tagged[j][2].lower() for j in range(i, i + length)
                )
                for cand in (candidate, candidate_lemma):
                    row = self._lookup_compound(cand)
                    if row:
                        text = ' '.join(tagged[j][0] for j in range(i, i + length))
                        result.append((text, row[1], row[0]))
                        i += length
                        matched = True
                        break
                if matched:
                    break
            if not matched:
                result.append(tagged[i])
                i += 1
        return result

    # ── Step 2: CONTEXT ────────────────────────────────

    def _build_context(self, t: Thought) -> Thought:
        """Extract entities, definitions, and update running context.

        Uses grammar engine's SentenceStructure when available for
        accurate entity extraction from grammatical roles (subjects,
        objects). Falls back to POS-tag scanning otherwise.
        """
        entities = []
        _QUESTION_WORDS = {'what', 'who', 'where', 'when', 'why', 'how', 'which'}

        if t.structure and hasattr(t.structure, 'subjects'):
            # Grammar engine parsed — extract entities from grammatical roles
            seen = set()
            for nf in (t.structure.subjects + t.structure.direct_objects +
                       t.structure.indirect_objects + t.structure.prep_objects +
                       t.structure.predicate_nouns):
                word = nf.word
                low = word.lower()
                if (len(word) > 2 and low not in _GENERIC_ADJ
                        and low not in _NOT_ENTITIES and low not in _PRONOUNS
                        and low not in _QUESTION_WORDS and low not in seen):
                    entities.append(word)
                    seen.add(low)
        else:
            # Fallback: scan tagged tokens for NOUN/PROPN
            for word, pos, lemma in t.grouped:
                if pos in ('NOUN', 'PROPN') and len(word) > 2:
                    low = word.lower()
                    if (low not in _GENERIC_ADJ and low not in _NOT_ENTITIES
                            and low not in _QUESTION_WORDS):
                        entities.append(word)

        # Add grouped compounds as entities — these are more specific
        # than individual words ("black holes" > "black" + "holes")
        for w, p, l in t.grouped:
            if ' ' in w and p in ('NOUN', 'PROPN'):
                low = w.lower()
                if (low not in _GENERIC_ADJ and low not in _NOT_ENTITIES
                        and low not in [e.lower() for e in entities]):
                    entities.append(w)

        t.entities = entities

        # Update running context
        for e in entities:
            key = e.lower()
            if key not in self._context['entities']:
                self._context['entities'][key] = {'mentions': 0}
            self._context['entities'][key]['mentions'] += 1

        # Topic = most-mentioned entity overall (not just this turn)
        if self._context['entities']:
            best_entity = max(
                self._context['entities'].items(),
                key=lambda x: x[1]['mentions']
            )
            self._context['topic'] = best_entity[0]

        # Use grammar engine's is_question if available, else heuristic
        if t.structure and hasattr(t.structure, 'is_question'):
            is_question = t.structure.is_question
        else:
            stripped = t.raw_input.strip()
            lower_stripped = stripped.lower()
            is_question = stripped.endswith('?') or any(
                lower_stripped.startswith(p) for p in (
                    'tell me', 'explain', 'describe', 'what about',
                )
            )

        current_topic = self._context['topic']
        if not is_question and current_topic and entities:
            for e in entities:
                if e.lower() != current_topic:
                    self._add_to_graph(e, 'related_to', current_topic)
            t.topic = best_entity[0]
        elif entities:
            t.topic = entities[0]
            self._context['topic'] = entities[0]

        # Extract facts from grammar structure: subject + verb + object → triple
        # Prefer concept_ids (language-neutral) with word fallback
        if t.structure and t.structure.subjects and t.structure.verb_lemma:
            s_nf = t.structure.subjects[0]
            subj = s_nf.concept_id or s_nf.word
            verb = t.structure.verb_concept_id or t.structure.verb_lemma
            obj = ''
            if t.structure.direct_objects:
                o_nf = t.structure.direct_objects[0]
                obj = o_nf.concept_id or o_nf.word
            elif t.structure.predicate_nouns:
                p_nf = t.structure.predicate_nouns[0]
                obj = p_nf.concept_id or p_nf.word
            if obj and s_nf.word.lower() not in _PRONOUNS:
                t.facts.append((subj, verb, obj))
                if not is_question:
                    self._add_to_graph(subj, verb, obj)

        # Skip definition/fact extraction for questions
        if is_question:
            return t

        # Definition extraction: "X is a/an Y" and "X is called Y" patterns
        for i, (word, pos, lemma) in enumerate(t.grouped):
            if word.lower() in ('is', 'are') and i > 0 and i < len(t.grouped) - 1:
                # Check for "is called" pattern: "X is called Y" → Y = X
                next_word = t.grouped[i + 1][0].lower()
                if next_word == 'called' and i + 2 < len(t.grouped):
                    # Subject = everything before "is", skip leading article
                    subj_parts = []
                    started = False
                    for j in range(0, i):
                        w, p = t.grouped[j][0], t.grouped[j][1]
                        if not started and p == 'DET' and w.lower() in ('the', 'a', 'an'):
                            continue  # skip leading article
                        started = True
                        subj_parts.append(w)
                    # Name after "called"
                    name_parts = []
                    for j in range(i + 2, len(t.grouped)):
                        if t.grouped[j][0] in ('.', ',', '!', '?'):
                            break
                        if t.grouped[j][1] != 'DET':
                            name_parts.append(t.grouped[j][0])
                    if subj_parts and name_parts:
                        name = ' '.join(name_parts)
                        subj = ' '.join(subj_parts)
                        t.definitions[name.lower()] = subj
                        self._context['definitions'][name.lower()] = subj
                        self._add_to_graph(name, 'is', subj)
                        self._learn_rule(name, subj)
                    break

                # Standard "X is Y" pattern
                subj_parts = []
                for j in range(i - 1, -1, -1):
                    if t.grouped[j][1] in ('DET', 'PUNCT'):
                        break
                    subj_parts.insert(0, t.grouped[j][0])

                # Complement after "is" (stop at relative pronouns, commas)
                comp_parts = []
                for j in range(i + 1, len(t.grouped)):
                    if t.grouped[j][0] in ('where', 'that', 'which', '.', ','):
                        break
                    if t.grouped[j][1] != 'DET':
                        comp_parts.append(t.grouped[j][0])

                if subj_parts and comp_parts:
                    subj = ' '.join(subj_parts)
                    comp = ' '.join(comp_parts)
                    # Skip pronoun subjects — they don't define anything
                    if subj.lower() not in _PRONOUNS and subj.lower() not in _REF_PRONOUNS:
                        t.definitions[subj.lower()] = comp
                        self._context['definitions'][subj.lower()] = comp
                        self._add_to_graph(subj, 'is', comp)
                        self._learn_rule(subj, comp)
                break

        # Fact extraction — skip questions (they're queries, not facts)
        if not t.raw_input.rstrip().endswith('?'):
            self._extract_facts(t)

        return t

    def _extract_facts(self, t: Thought):
        """Extract structured facts from the grouped (compound-aware) tokens.

        Extracts:
        - SVO triples: (subject, verb, object)
        - Causal facts: (effect, because, cause)
        - Ability facts: (subject, can/cannot, action)
        - Property facts: (subject, is, property)

        Uses grouped tokens (not the raw parser) so compounds stay intact.
        """
        grouped = t.grouped
        if not grouped:
            return

        # Find subject, verb, object from grouped tokens
        subj_parts = []
        verb = ''
        obj_parts = []
        phase = 'subject'  # subject → verb → object
        negated = False
        has_can = False
        has_cannot = False

        for w, p, l in grouped:
            low = w.lower()

            if p == 'PUNCT':
                continue

            if phase == 'subject':
                if p in ('VERB',) and low not in ('is', 'are', 'was', 'were'):
                    verb = l or low
                    phase = 'object'
                elif p == 'AUX':
                    if low in ('cannot', "can't"):
                        has_cannot = True
                        phase = 'verb_after_aux'
                    elif low == 'can':
                        has_can = True
                        phase = 'verb_after_aux'
                    elif low in ('is', 'are', 'was', 'were'):
                        verb = 'be'
                        phase = 'object'
                    elif low in ('do', 'does', 'did'):
                        phase = 'verb_after_aux'
                    else:
                        continue
                elif low == 'not':
                    negated = True
                elif p in ('NOUN', 'PROPN', 'ADJ'):
                    subj_parts.append(w)
                # skip DET, ADP etc
            elif phase == 'verb_after_aux':
                if p in ('ADV',) and low == 'not':
                    negated = True
                    has_cannot = True
                    has_can = False
                elif p in ('VERB', 'NOUN', 'ADJ'):
                    verb = l or low
                    phase = 'object'
            elif phase == 'object':
                if p in ('NOUN', 'PROPN', 'ADJ', 'ADV') and low != 'not':
                    obj_parts.append(w)
                elif p == 'ADP' and low == 'because':
                    break  # because-clause handled separately
                elif p == 'SCONJ' and low in ('because', 'when', 'that'):
                    break

        subject = ' '.join(subj_parts) if subj_parts else ''
        obj = ' '.join(obj_parts) if obj_parts else ''

        if not subject:
            return

        # Store SVO triple → graph edge
        # Skip if grammar structure already extracted a concept-level triple
        has_concept_triple = (t.structure and t.structure.verb_concept_id
                              and t.facts and any(
                                  f[1] == t.structure.verb_concept_id
                                  for f in t.facts))
        if verb and verb != 'be' and not has_concept_triple:
            triple = (subject, verb, obj)
            t.facts.append(triple)
            self._context['facts'].append(triple)
            if obj:
                self._add_to_graph(subject, verb, obj)

        # Ability facts: can/cannot → graph edge
        if has_cannot and verb:
            action = f'{verb} {obj}'.strip()
            ability = (subject, 'cannot', action)
            t.facts.append(ability)
            self._context['facts'].append(ability)
            self._add_to_graph(subject, 'cannot', action)
        elif has_can and verb:
            action = f'{verb} {obj}'.strip()
            ability = (subject, 'can', action)
            t.facts.append(ability)
            self._context['facts'].append(ability)
            self._add_to_graph(subject, 'can', action)

        # Property facts: "X is ADJ" → graph edge
        # Only store meaningful properties, not generic modifiers
        if verb == 'be' and obj_parts:
            for w, p, l in grouped:
                if w in obj_parts and p == 'ADJ' and w.lower() not in _SKIP_PROPERTIES:
                    prop = (subject, 'is', w.lower())
                    t.facts.append(prop)
                    self._context['facts'].append(prop)
                    self._add_to_graph(subject, 'is', w.lower())

        # Causal: "X because Y" → graph edge
        lower = t.raw_input.lower()
        if 'because' in lower:
            parts = lower.split('because', 1)
            if len(parts) == 2:
                cause = parts[1].strip().rstrip('.')
                effect = parts[0].strip()
                causal = (effect, 'because', cause)
                t.facts.append(causal)
                self._context['facts'].append(causal)
                self._add_to_graph(effect, 'because', cause)

    def _find_full_entity(self, word: str, grouped) -> str:
        """Find the full compound entity that contains this word."""
        for w, p, l in grouped:
            if word.lower() in w.lower() and p in ('NOUN', 'PROPN'):
                return w
        return word

    # ── Step 3: GAPS ───────────────────────────────────

    def _detect_gaps(self, t: Thought) -> Thought:
        """Find things the system doesn't understand.

        Two types of gaps:
        1. Unknown words — tokens not in the DB at all (symbols, math,
           foreign words, abbreviations). These are concepts to learn.
        2. Unknown compounds — ADJ+NOUN or NOUN+NOUN pairs not in DB.
        """
        conn = self._get_conn()
        if not conn:
            return t

        cur = conn.cursor()
        tagged = t.tagged

        # ── Unknown words — anything the DB doesn't know ────
        seen_unknown = set()
        for w, p, l in tagged:
            if p in ('PUNCT', 'DET', 'ADP', 'CCONJ', 'SCONJ', 'PART', 'INTJ'):
                continue
            low = w.lower()
            if not low or low in seen_unknown:
                continue
            # Skip numbers
            if w.isdigit() or low in _NUMBERS:
                continue
            # Skip if already in context definitions (already learned)
            if low in self._context.get('definitions', {}):
                continue
            # Check if the word exists in the DB (cached)
            if not self._word_exists(low):
                if not self._lookup_compound(low):
                    seen_unknown.add(low)
                    topic = t.topic or self._context.get('topic', '')
                    if topic:
                        question = f'What is {w} in the context of {topic}?'
                    else:
                        question = f'What is {w}?'
                    gap = KnowledgeGap(
                        compound=w,
                        context_topic=topic,
                        source_sentence=t.raw_input,
                        question=question,
                    )
                    t.gaps.append(gap)

        # ── Unknown compounds ──────────────────────────────
        i = 0
        while i < len(tagged) - 1:
            w1, p1, l1 = tagged[i]
            w2, p2, l2 = tagged[i + 1]

            # Skip non-content POS
            if p1 in ('PUNCT', 'DET', 'ADP', 'CCONJ', 'SCONJ', 'PRON',
                       'AUX', 'PART', 'INTJ') or \
               p2 in ('PUNCT', 'DET', 'ADP', 'PRON', 'AUX', 'PART'):
                i += 1
                continue

            if p1 in ('ADJ', 'NOUN', 'PROPN') and p2 in ('NOUN', 'PROPN'):
                low1 = w1.lower()
                low2 = w2.lower()
                pair = f'{low1} {low2}'

                # Filter 1: Skip generic adjectives
                if low1 in _GENERIC_ADJ:
                    i += 1
                    continue

                # Filter 2: Skip numbers (tagged as ADJ by our tagger)
                if low1.isdigit() or low1 in _NUMBERS:
                    i += 1
                    continue

                # Filter 3: Skip if either word is a known verb (mistagged)
                # "convert sunlight" — convert is VERB not NOUN
                # "water carry" — carry is VERB not NOUN
                skip_verb = False
                for check_word in (low1, low2):
                    if self._word_exists(check_word, 'VERB'):
                        if not self._word_exists(check_word, 'NOUN'):
                            skip_verb = True
                            break
                if skip_verb:
                    i += 1
                    continue

                # Filter 4: Skip if part of a longer noun chain (3+ nouns)
                # e.g., "wrought iron lattice tower" — don't flag "iron lattice"
                # or "lattice tower" individually
                in_chain = False
                if i > 0:
                    prev_pos = tagged[i - 1][1]
                    if prev_pos in ('ADJ', 'NOUN', 'PROPN'):
                        in_chain = True
                if i + 2 < len(tagged):
                    next_pos = tagged[i + 2][1]
                    if next_pos in ('NOUN', 'PROPN'):
                        in_chain = True
                if in_chain:
                    i += 1
                    continue

                # Filter 5: Skip if already a known compound
                if self._lookup_compound(pair):
                    i += 2
                    continue

                # Both words known individually but not as compound = gap
                w1_known = self._word_exists(low1)
                w2_known = self._word_exists(low2)

                if w1_known and w2_known:
                    topic = t.topic or self._context.get('topic', '')
                    if topic:
                        question = f'What is {pair} in the context of {topic}?'
                    else:
                        question = f'What is {pair}?'

                    gap = KnowledgeGap(
                        compound=pair,
                        context_topic=topic,
                        source_sentence=t.raw_input,
                        question=question,
                    )
                    t.gaps.append(gap)

            i += 1

        return t

    # ── Step 4: LEARN ──────────────────────────────────

    def learn_gap(self, gap: KnowledgeGap) -> bool:
        """Try to answer a knowledge gap and add to DB. Returns True if learned."""
        if not self._reasoner:
            return False

        result = self._reasoner.try_answer(gap.question, lang='en')
        if not result:
            return False

        raw = result[0] if isinstance(result, tuple) else result

        # Clean web markup
        clean = re.sub(r'\[Wikipedia:.*?\]\s*', '', raw)
        clean = re.sub(r'\[\d+\].*?\(duckduckgo\)\s*', '', clean)
        clean = clean.strip()

        if not clean:
            return False

        # Extract first sentence as definition
        first_sent = re.split(r'(?<=[.!?])\s+', clean)[0] if clean else ''

        # Try to extract "is a/an..." definition
        defn_match = re.search(r'is\s+(a|an|the)\s+(.+?)\.', first_sent,
                               re.IGNORECASE)
        if defn_match:
            definition = first_sent[:200]
        else:
            definition = first_sent[:200]

        # Add to compounds DB
        conn = self._get_conn()
        if conn:
            cur = conn.cursor()
            head_word = gap.compound.split()[-1]
            cur.execute(
                'INSERT OR REPLACE INTO compounds VALUES (?,?,?,?,?)',
                (gap.compound.lower(), 'NOUN', head_word, 'learned', definition)
            )
            conn.commit()

        gap.answered = True
        gap.answer = definition
        self._context['definitions'][gap.compound] = definition
        log.info("Learned: %s = %s", gap.compound, definition[:60])
        return True

    # ── Step 5: RESOLVE (conversation) ─────────────────

    def _resolve_references(self, t: Thought) -> Thought:
        """Replace pronouns with entities from conversation history."""
        if not self._history:
            t.resolved_input = t.raw_input
            return t

        # Build entity pool from recent history (topic first)
        entity_pool = []
        if self._context['topic']:
            entity_pool.append(self._context['topic'])

        for turn in reversed(self._history[-5:]):
            for ent in turn.entities:
                if ent.lower() not in [e.lower() for e in entity_pool]:
                    entity_pool.append(ent)

        if not entity_pool:
            t.resolved_input = t.raw_input
            return t

        words = t.raw_input.split()
        resolved = []
        for i, word in enumerate(words):
            clean = word.lower().rstrip('?.,!;:')

            # Only resolve clearly referential pronouns
            if clean in _REF_PRONOUNS and entity_pool:
                replacement = entity_pool[0]
                suffix = word[len(clean):]
                resolved.append(replacement + suffix)
                t.references[clean] = replacement
            elif clean == 'one' and i > 0:
                # "one" after "could/can/would" = indefinite, resolve to topic
                prev = words[i - 1].lower()
                if prev in ('could', 'can', 'would', 'might', 'should'):
                    replacement = entity_pool[0]
                    suffix = word[len(clean):]
                    resolved.append('a ' + replacement + suffix)
                    t.references[clean] = replacement
                else:
                    resolved.append(word)
            else:
                resolved.append(word)

        t.resolved_input = ' '.join(resolved)
        return t

    # ── Step 6: CLASSIFY ───────────────────────────────
    #
    # All conversation patterns are loaded from the DB at module init
    # (_CONV dict). No hardcoded pattern lists — the DB is the source
    # of truth. To teach ULI a new conversation type, add it to the
    # 'conversation' grammar_rules row.

    # Build the pattern table from DB data (once, at class load)
    # Each entry: (type_name, word_set, phrase_tuple, extra_check_fn)
    _CONV_PATTERNS = [
        ('greeting',       _GREETINGS,       _GREETING_PHRASES),
        ('farewell',       _FAREWELLS,        _FAREWELL_PHRASES),
        ('acknowledgment', _ACKNOWLEDGMENTS,  _ACK_PHRASES),
        ('introduction',   _INTRODUCTIONS,    _INTRO_PHRASES),
    ]

    def _classify(self, t: Thought) -> Thought:
        """Classify input by matching against DB-loaded conversation patterns.

        All patterns come from grammar_rules → conversation.
        To teach a new conversation type, add it to the DB.
        """
        lower = t.raw_input.lower().strip().rstrip('?!.')
        words = set(lower.split())

        # Try each conversation pattern from the DB
        for ptype, word_set, phrases in self._CONV_PATTERNS:
            if words & word_set or any(p in lower for p in phrases):
                # Type-specific guards (also from DB concepts, not hardcoded)
                if ptype == 'greeting':
                    # Only if most words ARE greeting words
                    non_greeting = words - word_set - {'there', 'you', 'are', 'how', 'do'}
                    if len(non_greeting) > 1:
                        continue
                elif ptype == 'acknowledgment':
                    # Not if there's an embedded question
                    if '?' in t.raw_input:
                        continue
                elif ptype == 'introduction':
                    # Extract the name from grammar structure
                    self._extract_user_name(t)

                t.response_type = ptype
                return t

        t.response_type = 'factual'
        return t

    def _extract_user_name(self, t: Thought):
        """Extract user's name from an introduction statement.

        Uses grammar parse: "My name is X" → extract complement of 'is'.
        Fallback: last PROPN/NOUN in the sentence.
        """
        # From grammar: "name is X" → X is the complement
        for i, (w, p, l) in enumerate(t.grouped):
            if w.lower() == 'is' and i + 1 < len(t.grouped):
                name_parts = [t.grouped[j][0] for j in range(i + 1, len(t.grouped))
                              if t.grouped[j][1] not in ('PUNCT',)]
                if name_parts:
                    self._context['user_name'] = ' '.join(name_parts)
                    return

        # Fallback for "I'm X" / "Call me X" — last PROPN/NOUN
        for w, p, l in reversed(t.grouped):
            if p in ('PROPN', 'NOUN') and w.lower() not in _NOT_ENTITIES:
                self._context['user_name'] = w
                return

    # ── Step 7: RESPOND ────────────────────────────────

    def _respond(self, t: Thought, lang: str = 'en') -> Thought:
        """Compose response from accumulated context, then reasoner if needed.

        Priority:
        1. Conversational patterns (greeting/farewell/ack)
        2. Context — definitions and facts already learned
        3. Reasoner/web search (expensive, last resort)
        """
        if t.response_type == 'greeting':
            t.response = "Hello! What would you like to know about?"
            return t

        if t.response_type == 'farewell':
            t.response = "Goodbye! Feel free to come back anytime."
            return t

        if t.response_type == 'acknowledgment':
            topic = self._context.get('topic', '')
            if topic:
                t.response = f"Glad I could help with {topic}. Anything else?"
            else:
                t.response = "You're welcome! Anything else?"
            return t

        if t.response_type == 'introduction':
            name = self._context.get('user_name', '')
            t.response = _INTRO_RESPONSE.format(name=name) if name else _INTRO_RESPONSE.format(name='')
            return t

        # Factual: check accumulated context FIRST
        response = self._respond_from_context(t)
        if response:
            t.response = response
            return t

        # Context didn't have it — try reasoner/web
        query = t.resolved_input or t.raw_input
        if self._reasoner:
            result = self._reasoner.try_answer(query, lang=lang)
            if result:
                raw = result[0] if isinstance(result, tuple) else result
                t.response = self._clean_web_response(raw, t)
                return t

        # No answer from any source
        if t.topic:
            t.response = f"I don't have enough information about {t.topic} yet."
        else:
            t.response = "I'm not sure about that. Could you rephrase?"
        return t

    def _respond_from_context(self, t: Thought) -> str:
        """Try to compose a response from what we already know.

        Routes by question type:
        - What is X? → definition
        - Why X? → causal facts
        - Can/Could X? → ability facts
        - How X? → process facts
        - What about X? → definition + relevant facts
        """
        query = (t.resolved_input or t.raw_input).lower()
        defs = self._context.get('definitions', {})
        facts = self._context.get('facts', [])

        # Extract the ACTUAL topic from the query — not just t.topic
        # "What about Phobos?" → topic should be "phobos", not "mars"
        # "Tell me about the atmosphere" → topic should be "atmosphere"
        # Prefer entities that appear in the query. Match both ways:
        # 1. entity appears in query ("atmosphere" in "tell me about the atmosphere")
        # 2. query word appears in entity key ("atmosphere" in "atmosphere of mars")
        topic = ''
        best_match_len = 0
        all_known = list(defs.keys()) + list(self._context.get('entities', {}).keys())
        # Sort by length (longest first) so "atmosphere of mars" > "mars"
        for entity in sorted(all_known, key=len, reverse=True):
            elow = entity.lower()
            if elow in query and len(entity) > best_match_len:
                topic = elow
                best_match_len = len(entity)
        # Also try: query words that match the start of a definition key
        # "atmosphere" in query → "atmosphere of mars" in defs
        if not topic:
            query_content = set(re.findall(r'\w+', query)) - _STOPWORDS
            for entity in sorted(all_known, key=len, reverse=True):
                elow = entity.lower()
                for qw in query_content:
                    if qw in elow.split() and len(entity) > best_match_len:
                        topic = elow
                        best_match_len = len(entity)
        # Fall back to the running topic
        if not topic:
            topic = (t.topic or self._context.get('topic', '')).lower()

        # Classify question type
        qtype = self._classify_question(query)

        if qtype == 'definition':
            return self._answer_definition(query, defs, topic)

        if qtype == 'causal':
            return self._answer_causal(query, facts, defs, topic)

        if qtype == 'ability':
            return self._answer_ability(query, facts, topic)

        if qtype == 'who':
            return self._answer_who(query, facts, topic)

        if qtype == 'relation':
            return self._answer_relation(query, t.entities, defs, topic)

        if qtype == 'about':
            return self._answer_about(query, defs, facts, topic)

        if qtype == 'process':
            # "How does X work?" — walk all action edges from topic
            return self._answer_about(query, defs, facts, topic)

        # General factual (what/where/when) — try definition, then about
        result = self._answer_definition(query, defs, topic)
        if result:
            return result
        result = self._answer_about(query, defs, facts, topic)
        if result:
            return result
        return self._answer_from_facts(query, facts, topic)

    def _classify_question(self, query: str) -> str:
        """Classify question type from the query text.

        Returns a type used by both the thinker response pipeline
        and DMRSM starting state selection.
        """
        q = query.strip().rstrip('?').lower()

        if q.startswith('why') or 'reason' in q:
            return 'causal'
        if q.startswith(('can ', 'could ', 'would ', 'is it possible')):
            return 'ability'
        if 'related' in q or 'connection' in q or 'relate' in q:
            return 'relation'
        if q.startswith('who') or 'developed' in q or 'invented' in q or 'discovered' in q:
            return 'who'
        if 'what is' in q or 'what are' in q or q.startswith('define'):
            return 'definition'
        if 'what about' in q or 'tell me about' in q:
            return 'about'
        if q.startswith('how'):
            return 'process'
        if q.startswith(('where', 'when')):
            return 'factual'
        return 'factual'

    def _classify_question_dmrsm(self, query: str) -> str:
        """Classify question into DMRSM starting state types.

        Maps the thinker's internal types to the 19 DMRSM types
        used by the transition table.
        """
        q = query.strip().rstrip('?').lower()
        words = q.split()

        # Math — numbers or operators in question
        if re.search(r'\d+\s*[\+\-\*/\^]', q) or q.startswith('calculate'):
            return 'math'

        # Multi-hop — "and" joining two questions, or comparative
        if (' and ' in q and '?' in query) or 'compare' in q or 'difference between' in q:
            return 'comparison'

        # Medical/legal/therapy triage
        for domain, patterns in [
            ('medical', [r'symptom', r'diagnos', r'pain\b', r'disease', r'treatment',
                         r'prescri', r'dosage', r'side effect']),
            ('legal', [r'\blegal\b', r'\blaw\b', r'\bsue\b', r'\bliable', r'\bcourt\b']),
            ('therapy', [r'\banxiety\b', r'\bdepressed\b', r'\bmental health',
                         r'\bstress\b', r'\btherapist']),
        ]:
            if any(re.search(p, q) for p in patterns):
                return domain

        # Creative
        if q.startswith(('write ', 'compose ', 'create ', 'imagine ', 'design ')):
            return 'creative'

        # Deep thought / multi-hop
        if ('why' in q and ('and' in q or 'how' in q)) or 'implications' in q:
            return 'deep_thought'

        # Temporal
        if q.startswith(('when', 'what year', 'what date')):
            return 'temporal'

        # Map internal types to DMRSM types
        internal = self._classify_question(query)
        _INTERNAL_TO_DMRSM = {
            'causal': 'deep_thought',
            'ability': 'factual',
            'relation': 'comparison',
            'who': 'factual',
            'definition': 'factual',
            'about': 'factual',
            'process': 'how_to',
            'factual': 'factual',
        }
        return _INTERNAL_TO_DMRSM.get(internal, 'factual')

    def _answer_definition(self, query, defs, topic) -> str:
        """Answer 'what is X?' — graph traversal: X --is--> ?"""
        # Find the entity being asked about
        target = ''
        for entity in list(defs.keys()) + [topic]:
            if entity and entity.lower() in query:
                target = entity
                break
        if not target and topic:
            target = topic
        if not target:
            return ''

        # Graph: X --is--> Y (pick the best definition)
        # Score each edge — prefer multi-word noun phrase definitions
        edges = self._query_graph(target, 'is')
        best_def = ''
        best_score = -1
        for rel, label, direction in edges:
            if direction != 'out':
                continue
            words = label.split()
            # Skip garbage
            if not words:
                continue
            # Skip single generic adjectives
            if len(words) == 1 and words[0] in _GENERIC_ADJ:
                continue
            # Skip if starts with adverb/conjunction fragment
            if words[0] in ('also', 'often', 'only', 'very', 'not', 'just',
                             'even', 'already', 'still', 'never'):
                continue
            # Skip if contains "because" (causal, not definition)
            if 'because' in label:
                continue
            # Score: word count, bonus for definitional patterns
            score = len(words)
            for pattern in ('theory', 'process', 'type', 'form', 'kind',
                            'region', 'property', 'principle', 'concept',
                            'phenomenon', 'equation', 'feature', 'law'):
                if pattern in label.lower():
                    score += 3
            if score > best_score:
                best_score = score
                best_def = label

        if best_def:
            return self._compose_definition(target, best_def)

        # Fallback to context dict
        if target.lower() in defs:
            return self._compose_definition(target, defs[target.lower()])
        return ''

    def _answer_causal(self, query, facts, defs, topic) -> str:
        """Answer 'why X?' — graph traversal: X --because--> ?"""
        # Graph: topic --because--> cause
        edges = self._query_graph(topic, 'because')
        for rel, cause, direction in edges:
            if direction == 'out':
                # Also get properties for context
                props = self._query_graph(topic, 'is')
                prop_text = ''
                for _, p, d in props:
                    if d == 'out' and p in ('dangerous', 'strong', 'extreme',
                                             'hot', 'cold', 'fast', 'slow'):
                        prop_text = p
                        break
                if prop_text:
                    return f"{topic.capitalize()} is {prop_text} because {cause}."
                return f"{topic.capitalize()} is that way because {cause}."

        # Also check incoming because edges
        edges = self._query_graph(topic, 'because')
        for rel, label, direction in edges:
            if direction == 'in':
                return f"{label.capitalize()} because {topic}."

        # Fallback to flat facts
        for subj, pred, obj in facts:
            if pred == 'because' and topic in subj.lower():
                return f"{subj.capitalize()} because {obj}."

        return self._answer_definition(query, defs, topic)

    def _answer_who(self, query, facts, topic) -> str:
        """Answer 'who developed/invented X?' — find people who acted on the topic."""
        conn = self._get_conn()
        if not conn:
            return ''

        cur = conn.cursor()
        topic_id = self._node_id(topic)

        # Find edges pointing TO the topic with action verbs
        # (people --develop/propose/introduce--> topic)
        cur.execute(
            'SELECT e.from_id, e.relation, n.label FROM graph_edges e '
            'JOIN graph_nodes n ON e.from_id = n.node_id '
            'WHERE e.to_id = ?',
            (topic_id,)
        )
        people = []
        for from_id, rel, label in cur.fetchall():
            if rel in ('related_to', 'is', 'can', 'cannot'):
                continue
            # Only include if the verb is a "who"-type action
            verb_base = rel.replace('_', ' ').split()[0] if rel else ''
            if verb_base in _WHO_VERBS:
                people.append((label, rel))

        # Also check edges FROM topic-related nodes (broader search)
        if not people:
            cur.execute(
                'SELECT e.from_id, e.relation, e.to_id, n.label FROM graph_edges e '
                'JOIN graph_nodes n ON e.from_id = n.node_id '
                'WHERE e.from_id IN ('
                '  SELECT from_id FROM graph_edges WHERE to_id = ? AND relation = "related_to"'
                ') AND e.relation NOT IN ("related_to", "is", "can", "cannot")',
                (topic_id,)
            )
            for from_id, rel, to_id, label in cur.fetchall():
                verb_base = rel.replace('_', ' ').split()[0] if rel else ''
                if verb_base in _WHO_VERBS:
                    people.append((label, rel))

        if people:
            parts = []
            seen = set()
            for name, verb in people[:5]:
                key = name.lower()
                if key not in seen:
                    seen.add(key)
                    v = verb.replace('_', ' ')
                    parts.append(self._compose_sentence(name, v, topic, tense='past'))
            return ' '.join(parts)

        return ''

    def _answer_ability(self, query, facts, topic) -> str:
        """Answer 'can/could X?' — graph traversal with is_a inheritance."""
        import re as _re
        query_words = set(_re.findall(r'\w+', query.lower())) - {
            'can', 'could', 'would', 'a', 'an', 'the', 'from', 'to',
            'of', 'in', 'on', 'is', 'it', 'do', 'does',
        }

        def _best_match(edges, query_words):
            """Pick the edge whose action best matches the query."""
            if not edges:
                return None
            # Score by query word overlap
            best, best_score = edges[0], 0
            for edge in edges:
                rel, action, direction = edge
                if direction != 'out':
                    continue
                action_words = set(action.lower().split())
                score = len(action_words & query_words)
                if score > best_score:
                    best = edge
                    best_score = score
            return best

        # Search graph for can/cannot edges — with inference
        for word in query_words:
            for rel_type in ('cannot', 'can'):
                edges = self._query_graph_with_inference(word, rel_type)
                out_edges = [e for e in edges if e[2] == 'out']
                if out_edges:
                    match = _best_match(out_edges, query_words)
                    if match:
                        action = match[1]
                        if rel_type == 'cannot':
                            return f"No, {word} cannot {action}."
                        return f"Yes, {word} can {action}."

        # Try topic
        for rel_type in ('cannot', 'can'):
            edges = self._query_graph_with_inference(topic, rel_type)
            out_edges = [e for e in edges if e[2] == 'out']
            if out_edges:
                match = _best_match(out_edges, query_words)
                if match:
                    action = match[1]
                    if rel_type == 'cannot':
                        return f"No, {topic} cannot {action}."
                    return f"Yes, {topic} can {action}."

        return ''

    def _answer_about(self, query, defs, facts, topic) -> str:
        """Answer 'tell me about X' — walk all edges from X, compose proper sentences."""
        target = topic
        for entity in defs:
            if entity.lower() in query:
                target = entity
                break

        parts = []
        best_def = ''
        best_def_len = 0
        properties = []

        # Collect all outgoing edges, categorize them
        all_edges = self._query_graph(target)
        for rel, label, direction in all_edges:
            if direction != 'out':
                continue
            if rel == 'related_to':
                continue  # Skip context links — they're not informative
            if rel == 'is':
                words = label.split()
                # Skip junk: adverb-leading, single generics, because-clauses
                if words and words[0] in ('also', 'often', 'only', 'very',
                                           'not', 'just', 'even'):
                    continue
                if 'because' in label:
                    continue
                if len(words) > 2 and len(label) > best_def_len:
                    best_def = label
                    best_def_len = len(label)
                elif len(words) <= 2 and words[0] not in _GENERIC_ADJ:
                    properties.append(label)
            elif rel == 'because':
                parts.append(f"This is because {label}.")
            elif rel in ('can', 'cannot'):
                parts.append(f"It {rel} {label}.")
            elif rel not in ('related_to',):
                # Action verb — use grammar engine
                verb = rel.replace('_', ' ')
                parts.append(self._compose_sentence(target.capitalize(), verb, label))

        # Also collect incoming action edges (people/things acting on target)
        for rel, label, direction in all_edges:
            if direction != 'in' or rel == 'related_to':
                continue
            verb = rel.replace('_', ' ')
            parts.append(self._compose_sentence(label.capitalize(), verb, target))

        # Build response: definition first, then properties, then facts
        result_parts = []
        if best_def:
            result_parts.append(self._compose_definition(target, best_def))
        elif target.lower() in defs:
            result_parts.append(self._compose_definition(target, defs[target.lower()]))

        if properties:
            prop_str = ', '.join(properties[:3])
            result_parts.append(f"It is {prop_str}.")

        result_parts.extend(parts)

        if result_parts:
            seen = set()
            unique = []
            for p in result_parts:
                if p not in seen:
                    seen.add(p)
                    unique.append(p)
            return ' '.join(unique[:5])

        # Fallback: if no definitions/properties/actions, use related_to edges
        # "What about Phobos?" → "Phobos is related to Mars."
        related = [label for rel, label, d in all_edges
                   if rel == 'related_to' and d == 'out']
        if related:
            return f"{target.capitalize()} is associated with {', '.join(related[:3])}."
        return ''

    def _answer_relation(self, query, entities, defs, topic) -> str:
        """Answer 'how is X related to Y?' — find edges between two entities."""
        # Find the two entities in the query
        entity_a = ''
        entity_b = ''

        # Check entities from the parsed question
        for e in entities:
            if not entity_a:
                entity_a = e.lower()
            elif e.lower() != entity_a:
                entity_b = e.lower()
                break

        # If only one entity found, the other is the topic
        if entity_a and not entity_b:
            entity_b = topic
        if not entity_a:
            entity_a = topic

        if not entity_a or not entity_b:
            return ''

        # Walk graph: find how A connects to B
        parts = []
        a_id = self._node_id(entity_a)
        b_id = self._node_id(entity_b)

        # Direct edges A → B or B → A
        edges_a = self._query_graph(entity_a)
        for rel, label, direction in edges_a:
            target_id = self._node_id(label)
            # Strict matching — exact or the target IS the entity
            if target_id != b_id and target_id != b_id + 's' and b_id != target_id + 's':
                continue
                # Compose a proper sentence via grammar engine
                if rel == 'related_to':
                    sent = f"{entity_a.capitalize()} is related to {entity_b}."
                elif rel in ('can', 'cannot'):
                    subj = entity_a if direction == 'out' else entity_b
                    obj = entity_b if direction == 'out' else entity_a
                    sent = f"{subj.capitalize()} {rel} {obj}."
                elif direction == 'out':
                    sent = self._compose_sentence(entity_a.capitalize(), rel.replace('_', ' '), entity_b)
                else:
                    sent = self._compose_sentence(entity_b.capitalize(), rel.replace('_', ' '), entity_a)
                parts.append(sent)

        # If no direct edge, check if they share a common node
        if not parts:
            edges_b = self._query_graph(entity_b)
            a_targets = {self._node_id(l): (r, l) for r, l, d in edges_a
                         if d == 'out' and self._node_id(l) not in (a_id, b_id)}
            b_targets = {self._node_id(l): (r, l) for r, l, d in edges_b
                         if d == 'out' and self._node_id(l) not in (a_id, b_id)}
            shared = set(a_targets.keys()) & set(b_targets.keys())
            if shared:
                for s in shared:
                    r_a, l_a = a_targets[s]
                    r_b, l_b = b_targets[s]
                    if r_a == 'is':
                        parts.append(f"Both {entity_a} and {entity_b} are {l_a}.")
                    elif r_a == 'related_to':
                        parts.append(f"Both {entity_a} and {entity_b} are related to {l_a}.")
                    else:
                        verb = r_a.replace('_', ' ')
                        parts.append(f"Both {entity_a} and {entity_b} {verb} {l_a}.")

        # Add definitions of both for context
        if entity_a in defs:
            defn_a = self._compose_definition(entity_a, defs[entity_a])
            if defn_a not in parts:
                parts.insert(0, defn_a)
        if entity_b in defs:
            defn_b = self._compose_definition(entity_b, defs[entity_b])
            if defn_b not in parts:
                parts.insert(1 if len(parts) > 0 else 0, defn_b)

        if parts:
            return ' '.join(parts[:4])

        return f"I know about {entity_a} and {entity_b}, but I haven't found a specific connection yet."

    def _answer_from_facts(self, query, facts, topic) -> str:
        """Fallback — walk graph for any edge from topic."""
        edges = self._query_graph(topic)
        if edges:
            rel, label, _ = edges[0]
            return f"{topic.capitalize()} {rel} {label}."
        return ''

    def _compose_sentence(self, subject: str, verb: str, obj: str,
                          tense: str = 'present') -> str:
        """Compose a grammatical sentence using the writer's AST pipeline.

        subject='Einstein', verb='develop', obj='quantum mechanics'
        → 'Einstein developed quantum mechanics.'
        """
        from .protocol import MeaningAST, Entity
        from .writer import ast_to_text

        ast = MeaningAST(
            type='statement',
            intent='factual',
            predicate=verb,
            tense=tense,
            agent=Entity(text=subject),
            patient=Entity(text=obj),
        )
        result = ast_to_text(ast)
        if result:
            return result
        # Fallback
        return f"{subject} {verb}s {obj}."

    def _compose_definition(self, entity: str, definition: str) -> str:
        """Compose 'X is Y' using the writer."""
        return self._compose_sentence(
            entity.strip().capitalize() if entity else '',
            'be',
            self._add_article(definition.strip()),
        )

    def _add_article(self, text: str) -> str:
        """Add indefinite article if text doesn't start with one."""
        if not text:
            return text
        if text[0].isupper():
            return text  # Proper noun
        if text.startswith(('a ', 'an ', 'the ', 'formed', 'used',
                             'called', 'known', 'made')):
            return text
        article = 'an' if text[0].lower() in 'aeiou' else 'a'
        return f'{article} {text}'

    def _clean_web_response(self, raw: str, t: Thought) -> str:
        """Extract relevant answer from raw web/KG response."""
        # Strip source markers
        clean = re.sub(r'\[Wikipedia:.*?\]\s*', '', raw)
        clean = re.sub(r'\[\d+\]\s*.*?\(duckduckgo\)\s*', '', clean)
        clean = re.sub(r'\(duckduckgo\)', '', clean)
        clean = clean.strip()

        if not clean:
            return raw

        # Split into sentences, pick most relevant
        sentences = re.split(r'(?<=[.!?])\s+', clean)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

        if not sentences:
            return clean[:300]

        # Score by keyword overlap with query
        query_words = set(re.findall(
            r'\w+', (t.resolved_input or t.raw_input).lower()
        ))
        query_words -= _STOPWORDS

        scored = []
        for sent in sentences:
            sent_words = set(re.findall(r'\w+', sent.lower()))
            overlap = len(query_words & sent_words)
            if t.topic and t.topic.lower() in sent.lower():
                overlap += 2
            scored.append((overlap, sent))

        scored.sort(key=lambda x: x[0], reverse=True)
        best = [s for score, s in scored[:2] if score > 0]

        if best:
            result = ' '.join(best)
        else:
            result = sentences[0]

        # Trim
        if len(result) > 500:
            match = list(re.finditer(r'[.!?]\s', result[:500]))
            if match:
                result = result[:match[-1].end()].strip()

        return result

    # ── Gap resolution (the reasoning loop) ─────────────

    def _resolve_gap(self, gap: KnowledgeGap) -> str:
        """Try to resolve a single gap. Returns 'learned', 'rejected', or 'unknown'.

        Resolution chain (cheapest first):
        1. Local wordnet.db — cross-reference word definitions
        2. Dictionary provider — local DB then free online API
        3. Reasoner/web search — expensive, last resort

        Math/symbol gaps skip step 1-2 (no dictionary entry for ψ)
        and go straight to web search with the contextual question.
        """
        conn = self._get_conn()

        # Step 1: DB cross-reference — does word1's definition mention word2?
        if conn:
            cur = conn.cursor()
            words = gap.compound.split()
            if len(words) == 2:
                cur.execute(
                    'SELECT definition FROM senses WHERE word=? LIMIT 5',
                    (words[0],)
                )
                defs = [r[0] for r in cur.fetchall() if r[0]]
                for d in defs:
                    if words[1] in d.lower():
                        definition = f'{gap.compound}: {d}'
                        cur.execute(
                            'INSERT OR REPLACE INTO compounds VALUES (?,?,?,?,?)',
                            (gap.compound.lower(), 'NOUN', words[-1],
                             'inferred', definition[:200])
                        )
                        conn.commit()
                        gap.answered = True
                        gap.answer = definition[:200]
                        self._context['definitions'][gap.compound] = definition[:200]
                        # Add to graph with provenance: inferred from DB cross-reference
                        self._add_to_graph(
                            gap.compound, 'is', definition[:100],
                            source_type='inferred',
                            source_text=f'cross-ref: {words[0]} definition mentions {words[1]}'
                        )
                        log.info("Inferred from DB: %s", gap.compound)
                        return 'learned'

        # Step 2: Dictionary provider (local wordnet + free online API)
        if self._dictionary:
            defn = self._dictionary.define(gap.compound)
            if defn:
                if conn:
                    cur = conn.cursor()
                    words = gap.compound.split()
                    if len(words) == 1:
                        cur.execute(
                            'INSERT OR IGNORE INTO words (word, pos, frequency) VALUES (?,?,?)',
                            (gap.compound.lower(), 'NOUN', 1)
                        )
                    else:
                        cur.execute(
                            'INSERT OR REPLACE INTO compounds VALUES (?,?,?,?,?)',
                            (gap.compound.lower(), 'NOUN', words[-1],
                             'dictionary', defn[:200])
                        )
                    conn.commit()
                gap.answered = True
                gap.answer = defn[:200]
                self._context['definitions'][gap.compound] = defn[:200]
                # Add to graph with provenance: dictionary
                self._add_to_graph(
                    gap.compound, 'is', defn[:100],
                    source_type='dictionary',
                    source_text=defn[:200]
                )
                log.info("Dictionary: %s = %s", gap.compound, defn[:60])
                self._learn_rule(gap.compound, defn)
                return 'learned'

        # Step 3: Reasoner/web search (expensive, last resort)
        if self._reasoner:
            result = self._reasoner.try_answer(gap.question, lang='en')
            if result:
                raw = result[0] if isinstance(result, tuple) else result
                clean = re.sub(r'\[Wikipedia:.*?\]\s*', '', raw)
                clean = re.sub(r'\[\d+\].*?\(duckduckgo\)\s*', '', clean).strip()
                # Take first 2 sentences as meaning
                sents = re.split(r'(?<=[.!?])\s+', clean)
                meaning = ' '.join(s for s in sents[:2] if len(s.strip()) > 10)
                if meaning:
                    if conn:
                        cur = conn.cursor()
                        words = gap.compound.split()
                        if len(words) == 1:
                            cur.execute(
                                'INSERT OR IGNORE INTO words (word, pos, frequency) VALUES (?,?,?)',
                                (gap.compound.lower(), 'NOUN', 1)
                            )
                        else:
                            cur.execute(
                                'INSERT OR REPLACE INTO compounds VALUES (?,?,?,?,?)',
                                (gap.compound.lower(), 'NOUN', words[-1],
                                 'learned', meaning[:200])
                            )
                        conn.commit()
                    gap.answered = True
                    gap.answer = meaning[:200]
                    self._context['definitions'][gap.compound] = meaning[:200]
                    # Add to graph with provenance: web search
                    # Extract source URL from raw response if available
                    source_url = None
                    wiki_match = re.search(r'\[Wikipedia:(.*?)\]', raw)
                    if wiki_match:
                        source_url = f'https://en.wikipedia.org/wiki/{wiki_match.group(1)}'
                    self._add_to_graph(
                        gap.compound, 'is', meaning[:100],
                        source_type='web_search',
                        source_url=source_url,
                        source_text=raw[:300]
                    )
                    log.info("Web: %s = %s", gap.compound, meaning[:60])
                    self._learn_rule(gap.compound, meaning)
                    return 'learned'

        return 'unknown'

    def _reject_gap(self, gap: KnowledgeGap):
        """Mark a gap as not-a-real-compound so we don't ask again."""
        conn = self._get_conn()
        if conn:
            cur = conn.cursor()
            # Add to compounds with a 'rejected' marker so grouping skips it
            # but gap detection won't re-flag it
            cur.execute(
                'INSERT OR IGNORE INTO compounds VALUES (?,?,?,?,?)',
                (gap.compound.lower(), 'REJECTED', '', 'rejected',
                 'not a compound')
            )
            conn.commit()

    def _reasoning_loop(self, t: Thought, max_iterations: int = 5) -> Thought:
        """The core reasoning loop: detect gaps → resolve → re-parse → repeat.

        Converges when no new gaps are found, or max iterations reached.
        Each iteration:
          1. Detect gaps in current parse
          2. For each gap, try to resolve (DB lookup, then web if available)
          3. If any gaps were resolved (new compounds added), re-parse
          4. If no gaps or no new resolutions, stop
        """
        all_gaps = []
        seen_gaps = set()

        for iteration in range(max_iterations):
            # Detect gaps
            t = self._detect_gaps(t)

            new_gaps = [g for g in t.gaps if g.compound not in seen_gaps]
            if not new_gaps:
                break  # Converged — no new gaps

            resolved_any = False
            for gap in new_gaps:
                seen_gaps.add(gap.compound)
                result = self._resolve_gap(gap)

                if result == 'learned':
                    resolved_any = True
                    all_gaps.append(gap)
                    log.info("Loop %d: learned '%s'", iteration + 1, gap.compound)
                elif result == 'unknown':
                    # Can't resolve — not necessarily a real compound
                    # Don't reject yet, might be answerable later with more context
                    all_gaps.append(gap)

            if not resolved_any:
                break  # No progress — stop looping

            # Re-parse with updated compound DB
            t.gaps = []  # Clear for next iteration
            t = self._parse(t)
            t = self._build_context(t)

        t.gaps = all_gaps
        return t

    # ── DMRSM Reasoning ─────────────────────────────────

    def reason(self, query: str, lang: str = 'en') -> Tuple[str, list, float]:
        """Run the DMRSM state machine to answer a question.

        Uses the transition table from data/system/transitions.json
        with non-neural action executors. Each action maps to existing
        thinker capabilities — no model calls.

        Returns: (answer, trace, confidence)
        """
        from .dmrsm import (
            ReasoningState, ActionResult, get_starting_state, transition,
            MAX_STEPS, MAX_SEARCHES,
        )
        from .safety import pre_filter, triage
        from .calculator import calculate

        # Safety gate
        is_safe, filter_type, response = pre_filter(query)
        if not is_safe:
            return response, [{'action': 'PRE_FILTER', 'signal': filter_type}], 1.0

        # Classify and get starting state
        q_type = self._classify_question_dmrsm(query)
        start = get_starting_state(q_type)

        state = ReasoningState(
            action=start,
            question=query,
            question_type=q_type,
        )

        # Run the state machine loop
        while state.action != 'DONE' and state.steps_done < MAX_STEPS:
            result = self._exec_dmrsm_action(state, lang)
            state.trace.append({
                'step': state.steps_done,
                'action': state.action,
                'signal': result.signal,
                'confidence': result.confidence,
            })

            # Update state
            if result.facts:
                state.facts.extend(result.facts)
            if result.answer:
                state.partial_answers[state.action] = result.answer
            if result.confidence > state.confidence:
                state.confidence = result.confidence
            if result.searched:
                state.searches_done += 1

            # Terminal?
            if result.terminal:
                break

            # Transition
            state.last_action = state.action
            state.action = transition(state, result)
            state.steps_done += 1

        answer = state.best_answer()
        if not answer:
            answer = "I couldn't find a confident answer. Could you rephrase?"
        return answer, state.trace, state.confidence

    def _exec_dmrsm_action(self, state, lang: str = 'en'):
        """Execute one DMRSM action using non-neural thinker capabilities."""
        from .dmrsm import ActionResult
        from .safety import triage
        from .calculator import calculate

        action = state.action

        if action == 'SEARCH':
            return self._dmrsm_search(state, lang)
        elif action == 'JUDGE':
            return self._dmrsm_judge(state)
        elif action == 'EXTRACT':
            return self._dmrsm_extract(state)
        elif action == 'DECOMPOSE':
            return self._dmrsm_decompose(state)
        elif action == 'SYNTHESIZE':
            return self._dmrsm_synthesize(state)
        elif action == 'TRIAGE':
            level = triage(state.question)
            state.safety_level = level
            return ActionResult(signal=level)
        elif action == 'CALCULATE':
            result = calculate(state.question)
            if result:
                return ActionResult(signal='done', answer=result,
                                    confidence=0.99, terminal=True)
            return ActionResult(signal='done', answer="Could not evaluate.",
                                confidence=0.3, terminal=True)
        elif action == 'GENERATE':
            # Creative: use context + graph to compose
            response = self._respond_from_context(
                Thought(raw_input=state.question,
                        topic=self._context.get('topic', ''),
                        entities=list(self._context.get('entities', {}).keys())))
            if response:
                return ActionResult(signal='done', answer=response,
                                    confidence=0.8, terminal=True)
            return ActionResult(signal='done',
                                answer="I'd need more context for that.",
                                confidence=0.3, terminal=True)
        elif action == 'GIVE_UP':
            return ActionResult(signal='done',
                                answer="I couldn't find a confident answer. Could you rephrase?",
                                terminal=True)
        elif action == 'DEFER':
            msg = "Please consult a qualified professional for personalized advice."
            if state.safety_level == 'urgent':
                msg = "This sounds urgent. Please seek immediate medical attention or call 911."
            state.partial_answers['DEFER'] = msg
            return ActionResult(signal='done', answer=msg, confidence=0.9)
        elif action == 'DISAMBIGUATE':
            if state.facts:
                answer = ' '.join(state.facts[-2:])[:300]
            else:
                answer = f"'{state.question}' could refer to several things."
            return ActionResult(signal='done', answer=answer,
                                confidence=0.6, terminal=True)
        elif action == 'PERSONA_ADOPT':
            return ActionResult(signal='adopted')
        elif action == 'REASON':
            # Try to infer from accumulated facts
            if state.facts:
                return ActionResult(signal='insight', facts=state.facts[-2:],
                                    confidence=0.6)
            return ActionResult(signal='need_evidence')
        else:
            return ActionResult(signal='done', terminal=True)

    def _dmrsm_search(self, state, lang='en'):
        """DMRSM SEARCH action — use reasoner/web to get facts."""
        from .dmrsm import ActionResult

        # Pick query: sub-question or original
        if state.sub_questions:
            query = state.sub_questions.pop(0)
        else:
            query = state.question

        # Try reasoner (which includes web search)
        if self._reasoner:
            result = self._reasoner.try_answer(query, lang=lang)
            if result:
                raw = result[0] if isinstance(result, tuple) else result
                relevance = self._relevance_score(state.question, raw)

                if relevance > 0.5:
                    return ActionResult(signal='relevant_high', facts=[raw[:500]],
                                        confidence=relevance, searched=True)
                elif relevance > 0.2:
                    return ActionResult(signal='relevant_low', facts=[raw[:500]],
                                        confidence=relevance, searched=True)
                else:
                    return ActionResult(signal='irrelevant', searched=True)

        return ActionResult(signal='no_results', searched=True)

    def _relevance_score(self, question: str, passage: str) -> float:
        """Score how relevant a passage is to a question.

        Tries structural similarity first, falls back to keyword overlap.
        """
        try:
            from .similarity import question_passage_relevance
            return question_passage_relevance(question, passage)
        except (ImportError, Exception):
            pass

        # Fallback: keyword overlap ratio
        q_words = set(re.findall(r'\w+', question.lower()))
        q_words -= _STOPWORDS
        if not q_words:
            return 0.5
        p_words = set(re.findall(r'\w+', passage.lower()))
        overlap = len(q_words & p_words)
        return min(overlap / len(q_words), 1.0)

    def _dmrsm_judge(self, state):
        """DMRSM JUDGE action — score current best answer structurally."""
        from .dmrsm import ActionResult

        answer = state.best_answer()
        if not answer:
            return ActionResult(signal='needs_more', confidence=0.2)

        relevance = self._relevance_score(state.question, answer)
        if relevance > 0.6:
            return ActionResult(signal='good', confidence=relevance)
        elif relevance > 0.3:
            return ActionResult(signal='needs_synthesis', confidence=relevance)
        else:
            return ActionResult(signal='needs_more', confidence=relevance)

    def _dmrsm_extract(self, state):
        """DMRSM EXTRACT action — extract answer from accumulated facts."""
        from .dmrsm import ActionResult

        if not state.facts:
            return ActionResult(signal='needs_more', confidence=0.1)

        # Build a Thought with accumulated facts and use existing response logic
        t = Thought(
            raw_input=state.question,
            topic=self._context.get('topic', ''),
            entities=list(self._context.get('entities', {}).keys()),
        )

        # Try cleaning the best fact as an answer
        best_fact = state.facts[-1]
        cleaned = self._clean_web_response(best_fact, t)
        if cleaned and len(cleaned.strip()) > 5:
            return ActionResult(signal='complete', answer=cleaned,
                                confidence=max(state.confidence, 0.75), terminal=True)

        return ActionResult(signal='needs_more', confidence=0.3)

    def _dmrsm_decompose(self, state):
        """DMRSM DECOMPOSE action — break question into sub-questions."""
        from .dmrsm import ActionResult

        # Extract entities from question to form sub-questions
        t = Thought(raw_input=state.question)
        t = self._parse(t)
        t = self._build_context(t)

        # Generate sub-questions from detected entities
        subs = []
        for entity in t.entities[:3]:
            if entity.lower() != state.question.lower():
                subs.append(f"What is {entity}?")

        # Also try to detect gaps
        t = self._detect_gaps(t)
        for gap in t.gaps[:2]:
            if gap.question and gap.question not in subs:
                subs.append(gap.question)

        state.sub_questions = subs[:3]
        log.info("Decomposed into: %s", subs)

        if subs:
            return ActionResult(signal='has_subs')
        return ActionResult(signal='cant_break')

    def _dmrsm_synthesize(self, state):
        """DMRSM SYNTHESIZE action — compose final answer from all evidence."""
        from .dmrsm import ActionResult

        if not state.partial_answers and not state.facts:
            best = state.best_answer()
            if best:
                return ActionResult(signal='complete', answer=best,
                                    confidence=state.confidence, terminal=True)
            return ActionResult(signal='weak', confidence=0.3)

        # Try to compose from context first
        t = Thought(
            raw_input=state.question,
            topic=self._context.get('topic', ''),
            entities=list(self._context.get('entities', {}).keys()),
            definitions=self._context.get('definitions', {}),
            facts=self._context.get('facts', []),
        )
        response = self._respond_from_context(t)
        if response:
            return ActionResult(signal='complete', answer=response,
                                confidence=max(state.confidence, 0.6), terminal=True)

        # Fall back to best partial answer
        best = state.best_answer()
        if best:
            cleaned = self._clean_web_response(best, t)
            return ActionResult(signal='complete', answer=cleaned or best,
                                confidence=state.confidence, terminal=True)

        # Last resort: use facts directly
        if state.facts:
            combined = ' '.join(state.facts[-3:])[:300]
            return ActionResult(signal='complete', answer=combined,
                                confidence=0.4, terminal=True)

        return ActionResult(signal='weak', confidence=0.2)

    # ── Main entry point ───────────────────────────────

    def think(self, user_input: str, lang: str = 'en') -> Thought:
        """Run the full inner monologue chain.

        The reasoning loop:
          check rules → resolve pronouns → parse → context → gaps → ... → respond

        Rules are checked FIRST — if a learned rule fires (e.g., URL → fetch),
        the fetched content becomes the input. Pronoun resolution runs next
        so "it"/"they" resolve using the previous turn's topic.
        """
        t = Thought(raw_input=user_input)

        # Check learned rules FIRST — may transform input entirely
        # e.g., URL → fetched content, LaTeX → parsed AST
        rule_result = self._check_rules(user_input)
        if rule_result:
            # Rule transformed the input — the original was a "pointer"
            # (URL, file path, etc.), and rule_result is the actual content.
            # Read the content, then respond about what we learned.
            log.info("Rule transformed input (%d chars → %d chars)",
                     len(user_input), len(rule_result))
            t.response = (
                f"I read the content from that link. "
                f"Learned {len(self._context.get('definitions', {}))} concepts. "
                f"Ask me anything about what I just read."
            )
            t.topic = self._context.get('topic', '')
            t.entities = list(self._context.get('entities', {}).keys())[:5]
            # Record turn
            self._history.append(ConversationTurn(
                user_input=user_input,
                response=t.response,
                topic=t.topic,
                entities=t.entities,
            ))
            return t

        # Resolve pronouns first (uses previous turn's topic)
        t = self._resolve_references(t)

        # Parse the resolved input (so "it" is already replaced)
        parse_input = t.resolved_input or t.raw_input
        t_parse = Thought(raw_input=parse_input)
        t_parse = self._parse(t_parse)
        t.tagged = t_parse.tagged
        t.grouped = t_parse.grouped
        t.structure = t_parse.structure

        # Build context from resolved parse
        t = self._build_context(t)

        # Reasoning loop: detect gaps → resolve → re-parse until stable
        t = self._reasoning_loop(t)

        # Classify + respond
        t = self._classify(t)
        t = self._respond(t, lang)

        # Record turn
        self._history.append(ConversationTurn(
            user_input=user_input,
            response=t.response,
            topic=t.topic,
            entities=t.entities,
        ))
        if len(self._history) > 10:
            self._history = self._history[-10:]

        return t

    def read_text(self, text: str, resolve_gaps: bool = True) -> List[Thought]:
        """Read a paragraph sentence by sentence, building context and detecting gaps.

        This is the 'study' mode — the system reads text to learn,
        not to respond to a user.

        If resolve_gaps=True, runs the reasoning loop on each sentence
        (detect gap → resolve → re-parse → repeat until stable).
        """
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text)
                     if s.strip()]
        thoughts = []

        for sent in sentences:
            t = Thought(raw_input=sent)
            t = self._parse(t)
            t = self._build_context(t)

            if resolve_gaps:
                t = self._reasoning_loop(t)
            else:
                t = self._detect_gaps(t)

            thoughts.append(t)

        return thoughts

    def close(self):
        """Close DB connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
