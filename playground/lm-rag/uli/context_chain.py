"""
Context Chain — sliding window of MiniLM embeddings for disambiguation.

Closes Gap 2 (implicit reasoning) and Gap 4 (pragmatics).
Maintains a running topic vector from recent conversation history.
Uses cosine similarity to disambiguate words and detect sarcasm.

NOT a transformer attention mechanism. Just a running average of
sentence embeddings — simpler but effective for topic tracking.
"""

import numpy as np
import logging
from typing import List, Dict, Optional, Tuple

log = logging.getLogger('uli.context')


class ContextChain:
    """Sliding window of embeddings for context-aware disambiguation."""

    def __init__(self, embedder, window: int = 5):
        self.embedder = embedder
        self.window = window
        self.history: List[Tuple[str, np.ndarray]] = []

    def add(self, text: str):
        """Add a text to the context window."""
        if not text or len(text.strip()) < 2:
            return
        emb = self.embedder.encode(text)
        self.history.append((text, emb))
        if len(self.history) > self.window:
            self.history.pop(0)

    def context_embedding(self) -> Optional[np.ndarray]:
        """Average of recent embeddings = topic vector."""
        if not self.history:
            return None
        return np.mean([emb for _, emb in self.history], axis=0)

    def clear(self):
        self.history.clear()

    # ── Word Sense Disambiguation ────────────────────────

    def disambiguate(self, word: str, senses: Dict[str, str]) -> Tuple[str, float]:
        """Pick the word sense closest to current context.

        Args:
            word: the ambiguous word ("bank", "duck", "Mercury")
            senses: {sense_name: description} e.g. {"river_bank": "edge of a river", "money_bank": "financial institution"}

        Returns: (best_sense, confidence)
        """
        ctx_emb = self.context_embedding()
        if ctx_emb is None:
            # No context — return first sense with low confidence
            return list(senses.keys())[0] if senses else '', 0.3

        best_sense = ''
        best_sim = -1.0

        for sense, description in senses.items():
            sense_emb = self.embedder.encode(f"{word}: {description}")
            sim = float(np.dot(ctx_emb, sense_emb) /
                       (np.linalg.norm(ctx_emb) * np.linalg.norm(sense_emb) + 1e-8))
            if sim > best_sim:
                best_sim = sim
                best_sense = sense

        return best_sense, max(best_sim, 0.0)

    # ── Sarcasm Detection ────────────────────────────────

    POSITIVE_WORDS = {'great', 'wonderful', 'amazing', 'fantastic', 'brilliant',
                      'perfect', 'excellent', 'awesome', 'lovely', 'nice', 'good',
                      'terrific', 'superb', 'fabulous', 'delightful'}
    NEGATIVE_CONTEXT_SIGNALS = {'meeting', 'traffic', 'monday', 'deadline', 'bug',
                                'error', 'broken', 'late', 'problem', 'issue',
                                'fail', 'crash', 'crashed', 'wait', 'queue', 'slow',
                                'lost', 'data', 'nothing', 'waste', 'stuck', 'ugh',
                                'terrible', 'horrible', 'awful', 'annoying'}
    SARCASM_STARTERS = {'oh', 'yeah', 'sure', 'right', 'wow', 'gee', 'thanks'}

    def detect_sarcasm(self, text: str) -> Tuple[bool, float]:
        """Detect sarcasm by checking if positive surface contradicts negative context.

        Returns: (is_sarcastic, confidence)
        """
        # Strip punctuation from words for matching
        import re
        words = set(re.findall(r'\w+', text.lower()))

        # Check for sarcasm pattern: positive words + sarcasm starter
        has_positive = bool(words & self.POSITIVE_WORDS)
        has_starter = bool(words & self.SARCASM_STARTERS)

        if not has_positive:
            return False, 0.0

        # Check if recent context is negative
        context_negative = self._context_is_negative()

        if has_positive and context_negative:
            confidence = 0.6
            if has_starter:
                confidence = 0.8  # "Oh great" stronger signal than just "great"
            return True, confidence

        return False, 0.0

    def _context_is_negative(self) -> bool:
        """Check if recent context has negative sentiment."""
        if not self.history:
            return False
        recent_text = ' '.join(text for text, _ in self.history[-3:]).lower()
        negative_count = sum(1 for w in self.NEGATIVE_CONTEXT_SIGNALS if w in recent_text)
        return negative_count >= 1

    # ── Implicature Resolution ───────────────────────────

    IMPLICATURE_PATTERNS = [
        {
            'trigger': 'can you',
            'literal': 'ability_question',
            'implied': 'request',
            'context_words': {'pass', 'give', 'hand', 'open', 'close', 'turn', 'send'},
        },
        {
            'trigger': 'do you know',
            'literal': 'knowledge_question',
            'implied': 'information_request',
            'context_words': {'time', 'where', 'how', 'way', 'number', 'address', 'what'},
            'check_text': True,  # Also check trigger text for context words
        },
        {
            'trigger': "it's cold",
            'literal': 'temperature_statement',
            'implied': 'close_window_request',
            'context_words': {'window', 'door', 'draft', 'open'},
        },
    ]

    def resolve_implicature(self, text: str) -> Optional[str]:
        """Detect indirect speech acts.

        "Can you pass the salt?" → request (not ability question)
        "Do you know the time?" → information request (not yes/no)

        Returns: implied meaning or None if literal.
        """
        text_lower = text.lower()

        for pattern in self.IMPLICATURE_PATTERNS:
            if pattern['trigger'] in text_lower:
                # Check if context words are in the text itself
                words = set(text_lower.split())
                if words & pattern['context_words']:
                    return pattern['implied']

                # Also check full text (not just split words) for partial matches
                if pattern.get('check_text'):
                    if any(w in text_lower for w in pattern['context_words']):
                        return pattern['implied']

                # Check conversation context
                if self.history:
                    recent = ' '.join(t for t, _ in self.history[-2:]).lower()
                    if any(w in recent for w in pattern['context_words']):
                        return pattern['implied']

        return None  # Literal interpretation

    # ── Pronoun Resolution ───────────────────────────────

    def resolve_pronoun(self, pronoun: str) -> Optional[str]:
        """Resolve pronoun to most recent matching entity in context.

        "it", "its", "they", "he", "she" → entity from conversation history.
        """
        if not self.history:
            return None

        pronoun_lower = pronoun.lower()

        # Map pronouns to entity types
        PRONOUN_TYPES = {
            'he': 'person', 'him': 'person', 'his': 'person',
            'she': 'person', 'her': 'person',
            'it': 'thing', 'its': 'thing',
            'they': 'any', 'them': 'any', 'their': 'any',
        }

        target_type = PRONOUN_TYPES.get(pronoun_lower, 'any')

        # Search recent history for entities
        # Use embeddings: find the entity in recent context most similar to
        # the surrounding sentence
        for text, _ in reversed(self.history):
            # Simple heuristic: find proper nouns / capitalized words
            words = text.split()
            for word in words:
                if word[0].isupper() and len(word) > 1 and word.lower() not in {
                    'the', 'a', 'an', 'i', 'what', 'who', 'where', 'when',
                    'why', 'how', 'is', 'are', 'was', 'were', 'do', 'does',
                    'did', 'no', 'yes', 'not', 'but', 'and', 'or', 'if',
                }:
                    return word

        return None

    # ── Topic Continuity ─────────────────────────────────

    def is_topic_switch(self, new_text: str, threshold: float = 0.3) -> bool:
        """Detect if the new text is a topic switch from recent context."""
        ctx_emb = self.context_embedding()
        if ctx_emb is None:
            return False

        new_emb = self.embedder.encode(new_text)
        sim = float(np.dot(ctx_emb, new_emb) /
                    (np.linalg.norm(ctx_emb) * np.linalg.norm(new_emb) + 1e-8))
        return sim < threshold

    def topic_similarity(self, text: str) -> float:
        """How similar is this text to the current context topic? 0-1."""
        ctx_emb = self.context_embedding()
        if ctx_emb is None:
            return 0.5

        text_emb = self.embedder.encode(text)
        return max(0.0, float(np.dot(ctx_emb, text_emb) /
                    (np.linalg.norm(ctx_emb) * np.linalg.norm(text_emb) + 1e-8)))
