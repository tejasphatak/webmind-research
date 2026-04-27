"""
ULI Protocol — MeaningAST + LanguageModule interface.

MeaningAST is the language-independent representation that flows between
ULI (reader/writer) and DMRSM (thinker). It represents MEANING, not words.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Protocol


@dataclass
class Entity:
    """A named or described thing in the AST."""
    text: str                    # Surface form: "Michelangelo", "the ceiling"
    type: str = 'unknown'        # person, place, thing, concept, number, time
    id: str = ''                 # Unique ID if known (e.g., Wikidata QID)


@dataclass
class MeaningAST:
    """Language-independent meaning representation.

    The same AST is produced whether the input is English, Marathi,
    or code-switched. The engine reasons on this, not on text.
    """
    # Sentence type
    type: str = 'statement'      # question, statement, command, exclamation
    intent: str = 'factual'      # factual, comparison, explanation, creative, math, ...

    # Core semantic frame (who did what to whom, where, when, how, why)
    predicate: str = ''          # Main action/state: paint, be, have, go
    agent: Optional[Entity] = None
    patient: Optional[Entity] = None
    theme: Optional[Entity] = None
    location: Optional[Entity] = None
    time: Optional[Entity] = None
    manner: str = ''
    reason: str = ''

    # Question slot (what the question asks for)
    question_word: str = ''      # who, what, when, where, why, how, which
    question_target: str = ''    # The semantic role being asked: agent, patient, location, time

    # Modifiers
    negation: bool = False
    modality: str = 'realis'     # realis, irrealis, hypothetical, imperative
    tense: str = 'present'
    aspect: str = 'simple'       # simple, progressive, perfect

    # Discourse
    form: str = 'statement'      # question, email, essay, chat, poem, ...
    register: str = 'neutral'    # formal, informal, gen_z, academic, ...
    person: str = 'third'        # first, second, third

    # Nested (for complex/compound sentences and multi-hop questions)
    sub_clauses: List['MeaningAST'] = field(default_factory=list)

    # Entities extracted (for search)
    entities: List[str] = field(default_factory=list)

    # Source tracking (for traceability)
    source: str = ''
    source_language: str = ''
    confidence: float = 1.0

    def search_query(self) -> str:
        """Extract search terms from the AST structure."""
        parts = []
        if self.agent and self.agent.text and self.agent.text != '?':
            parts.append(self.agent.text)
        if self.predicate:
            parts.append(self.predicate)
        if self.patient and self.patient.text:
            parts.append(self.patient.text)
        if self.theme and self.theme.text:
            parts.append(self.theme.text)
        if self.location and self.location.text:
            parts.append(self.location.text)
        if self.time and self.time.text:
            parts.append(self.time.text)
        parts.extend(self.entities)
        # Deduplicate preserving order
        seen = set()
        unique = []
        for p in parts:
            if p.lower() not in seen:
                seen.add(p.lower())
                unique.append(p)
        return ' '.join(unique)

    def has_nested(self) -> bool:
        return len(self.sub_clauses) > 0

    def unfilled_slots(self) -> List[str]:
        """Which semantic roles are missing (question targets)?"""
        slots = []
        if self.agent is None or self.agent.text == '?':
            slots.append('agent')
        if self.patient is None and self.question_target == 'patient':
            slots.append('patient')
        if self.location is None and self.question_target == 'location':
            slots.append('location')
        if self.time is None and self.question_target == 'time':
            slots.append('time')
        return slots


@dataclass
class Token:
    """A token with language and POS info."""
    text: str
    lang: str = 'en'             # Language code
    pos: str = ''                # Part of speech (NOUN, VERB, ADJ, ...)
    dep: str = ''                # Dependency relation (nsubj, dobj, ...)
    head_idx: int = -1           # Index of head token
    lemma: str = ''              # Base form
    is_entity: bool = False
    entity_type: str = ''        # PERSON, ORG, GPE, DATE, ...


class LanguageModule:
    """Interface for a pluggable language module.
    Implement this for any language by providing JSON data files."""

    def detect(self, text: str) -> str:
        """Detect language. Returns ISO 639-1 code."""
        raise NotImplementedError

    def normalize(self, text: str) -> str:
        """Fix spelling, expand abbreviations, normalize."""
        raise NotImplementedError

    def tokenize(self, text: str) -> List[Token]:
        """Split into tokens with POS and dependency info."""
        raise NotImplementedError

    def to_ast(self, tokens: List[Token], text: str = '') -> MeaningAST:
        """Convert parsed tokens to language-independent MeaningAST."""
        raise NotImplementedError

    def from_ast(self, ast: MeaningAST, form: str = 'statement',
                 temperature: float = 0.0) -> str:
        """Generate text from MeaningAST. Temperature controls creativity."""
        raise NotImplementedError

    def read(self, text: str) -> MeaningAST:
        """Full pipeline: text → normalize → tokenize → AST."""
        text = self.normalize(text)
        tokens = self.tokenize(text)
        return self.to_ast(tokens, text)

    def write(self, ast: MeaningAST, temperature: float = 0.0) -> str:
        """Generate text from AST."""
        return self.from_ast(ast, form=ast.form, temperature=temperature)
