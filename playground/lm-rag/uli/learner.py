"""
ULI Learner — Training by talking / feeding data.

Neural networks learn by updating weights via gradient descent.
DMRSM-ULI learns by updating JSON databases via INSERT/UPDATE operations.

Three modes:
  1. Conversation (interactive) — learn from unknown words, corrections, verified answers
  2. Bulk data (batch) — learn vocabulary and templates from document corpora
  3. Explicit teaching (direct) — user directly teaches a word, idiom, or template

Grammar rules and safety filters are NEVER auto-updated.
"""

import json
import os
import re
import logging
from typing import List, Dict, Optional, Any
from collections import Counter

from .protocol import Token, MeaningAST
from .lexer import detect_token_language, _load_json

log = logging.getLogger('uli.learner')

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')


# ============================================================
# Learning Queue — buffer before committing
# ============================================================

class LearningQueue:
    """Buffer unknown items. Commit to DB after threshold encounters.
    Prevents noise from single-occurrence typos polluting the vocab."""

    def __init__(self, threshold: int = 3):
        self.threshold = threshold
        self.unknown_words: Dict[str, dict] = {}  # word → {count, contexts, lang, pos_guesses}

    def flag(self, word: str, context: str = '', lang: str = 'en',
             pos_hint: str = '') -> bool:
        """Flag an unknown word. Returns True if threshold reached."""
        key = word.lower().strip()
        if not key or len(key) < 2:
            return False

        if key not in self.unknown_words:
            self.unknown_words[key] = {
                'count': 0,
                'contexts': [],
                'lang': lang,
                'pos_guesses': [],
            }

        entry = self.unknown_words[key]
        entry['count'] += 1
        if context and len(entry['contexts']) < 10:
            entry['contexts'].append(context[:200])
        if pos_hint:
            entry['pos_guesses'].append(pos_hint)

        return entry['count'] >= self.threshold

    def get_ready(self) -> List[dict]:
        """Get words that have reached the threshold."""
        ready = []
        for word, data in list(self.unknown_words.items()):
            if data['count'] >= self.threshold:
                # Infer POS from most common guess
                pos = 'NOUN'  # Default
                if data['pos_guesses']:
                    pos_counts = Counter(data['pos_guesses'])
                    pos = pos_counts.most_common(1)[0][0]
                ready.append({
                    'word': word,
                    'pos': pos,
                    'lang': data['lang'],
                    'count': data['count'],
                    'contexts': data['contexts'][:3],
                })
        return ready

    def remove(self, word: str):
        """Remove word from queue (after committing to DB)."""
        self.unknown_words.pop(word.lower().strip(), None)

    def clear(self):
        self.unknown_words.clear()


# ============================================================
# Learner — main learning interface
# ============================================================

class Learner:
    """Learning layer — updates JSON databases from conversation and data.

    Operates on JSON files in data/ directory. Never touches grammar rules
    or safety filters.
    """

    def __init__(self, data_dir: str = DATA_DIR):
        self.data_dir = data_dir
        self.queue = LearningQueue(threshold=3)
        self._dirty = set()  # Track which files need saving

        # Load current state
        self._vocab = {}      # lang → word_dict
        self._normalize = {}  # lang → normalize_dict
        self._idioms = {}     # lang → idiom_dict

    def _ensure_loaded(self, lang: str):
        """Lazy-load language data files."""
        if lang not in self._vocab:
            self._vocab[lang] = _load_json(
                os.path.join(self.data_dir, 'vocab', f'{lang}.json'))
        if lang not in self._normalize:
            self._normalize[lang] = _load_json(
                os.path.join(self.data_dir, 'normalize', f'{lang}.json'))
        if lang not in self._idioms:
            self._idioms[lang] = _load_json(
                os.path.join(self.data_dir, 'idioms', f'{lang}.json'))

    def _get_known_words(self, lang: str) -> set:
        """Get all known words for a language."""
        self._ensure_loaded(lang)
        words = set()
        vocab = self._vocab.get(lang, {})
        words.update(vocab.get('words', {}).keys())
        words.update(vocab.get('stop_words', []))
        words.update(vocab.get('question_words', []))
        # Also include normalize abbreviations
        norm = self._normalize.get(lang, {})
        words.update(norm.get('abbreviations', {}).keys())
        words.update(norm.get('contractions', {}).keys())
        return words

    # ── Mode 1: Conversation Learning ────────────────────

    def on_tokens(self, tokens: List[Token], text: str = '', lang: str = 'en'):
        """Process tokens from a conversation turn. Flag unknown words."""
        known = self._get_known_words(lang)
        for tok in tokens:
            word = tok.text.lower().strip()
            # Skip: short words, punctuation, numbers, known words
            if (len(word) < 3 or not word.isalpha() or
                word in known or tok.is_entity):
                continue
            reached = self.queue.flag(
                word, context=text, lang=tok.lang,
                pos_hint=tok.pos if tok.pos else '')
            if reached:
                log.info(f"[LEARN] Word '{word}' reached threshold, ready to commit")

    def on_correction(self, wrong: str, right: str, category: str = 'fact',
                      lang: str = 'en'):
        """User corrected the system. Update appropriate DB.

        category: 'fact' (knowledge), 'word' (vocabulary), 'idiom' (expression)
        """
        log.info(f"[LEARN] Correction: '{wrong}' → '{right}' ({category})")

        if category == 'word':
            self.teach_word(wrong, {'correction': right}, lang=lang)
        elif category == 'idiom':
            self.teach_idiom(wrong, right, lang=lang)
        elif category == 'abbreviation':
            self.teach_abbreviation(wrong, right, lang=lang)
        # 'fact' corrections go to knowledge DB (handled by engine)

    def on_verified_answer(self, question: str, answer: str) -> dict:
        """User confirmed an answer is correct. Returns fact for KB storage."""
        fact = {
            'question': question,
            'answer': answer,
            'confidence': 1.0,
            'source': 'user_verified',
        }
        log.info(f"[LEARN] Verified: '{question}' → '{answer}'")
        return fact

    def commit_queue(self, lang: str = 'en') -> List[str]:
        """Commit words that reached the threshold to vocab DB.
        Returns list of words committed."""
        committed = []
        for entry in self.queue.get_ready():
            if entry['lang'] != lang and lang != 'all':
                continue
            word = entry['word']
            self._ensure_loaded(entry['lang'])
            vocab = self._vocab.setdefault(entry['lang'], {'words': {}})
            words_dict = vocab.setdefault('words', {})

            if word not in words_dict:
                words_dict[word] = {
                    'pos': [entry['pos']],
                    'learned': True,
                    'frequency': 0.0001,
                }
                committed.append(word)
                self._dirty.add(('vocab', entry['lang']))
                log.info(f"[LEARN] Committed '{word}' to vocab/{entry['lang']}.json")

            self.queue.remove(word)
        return committed

    # ── Mode 2: Bulk Data Learning ───────────────────────

    def learn_from_text(self, text: str, lang: str = 'en') -> dict:
        """Learn vocabulary from a text block. Returns stats."""
        from .lexer import tokenize
        tokens = tokenize(text, lang=lang)
        self.on_tokens(tokens, text=text, lang=lang)

        stats = {
            'tokens_processed': len(tokens),
            'unknown_flagged': len(self.queue.unknown_words),
        }

        # Auto-commit if we have enough data
        committed = self.commit_queue(lang=lang)
        stats['words_committed'] = len(committed)
        stats['committed_words'] = committed

        return stats

    def learn_from_documents(self, docs: List[dict], lang: str = 'en') -> dict:
        """Learn from multiple documents. Each doc: {text, type?}.
        Returns aggregate stats."""
        total_stats = {
            'docs_processed': 0,
            'tokens_processed': 0,
            'words_committed': 0,
        }

        for doc in docs:
            text = doc.get('text', '')
            if not text:
                continue
            stats = self.learn_from_text(text, lang=lang)
            total_stats['docs_processed'] += 1
            total_stats['tokens_processed'] += stats['tokens_processed']
            total_stats['words_committed'] += stats['words_committed']

        return total_stats

    # ── Mode 3: Explicit Teaching ────────────────────────

    def teach_word(self, word: str, definition: dict, lang: str = 'en'):
        """Directly teach a word. Definition can include pos, senses, register."""
        self._ensure_loaded(lang)
        vocab = self._vocab.setdefault(lang, {'words': {}})
        words_dict = vocab.setdefault('words', {})

        entry = words_dict.get(word.lower(), {})
        entry.update(definition)
        entry['taught'] = True
        words_dict[word.lower()] = entry

        self._dirty.add(('vocab', lang))
        log.info(f"[TEACH] Word '{word}' added to vocab/{lang}.json")

    def teach_abbreviation(self, abbrev: str, expansion: str, lang: str = 'en'):
        """Teach a new abbreviation: 'ngl' → 'not gonna lie'."""
        self._ensure_loaded(lang)
        norm = self._normalize.setdefault(lang, {'abbreviations': {}})
        abbrevs = norm.setdefault('abbreviations', {})
        abbrevs[abbrev.lower()] = expansion
        self._dirty.add(('normalize', lang))
        log.info(f"[TEACH] Abbreviation '{abbrev}' → '{expansion}'")

    def teach_idiom(self, phrase: str, meaning: str, lang: str = 'en',
                    literal: bool = False):
        """Teach a new idiom: 'spill the tea' → 'share gossip'."""
        self._ensure_loaded(lang)
        idioms = self._idioms.setdefault(lang, {})
        idioms[phrase.lower()] = {
            'meaning': meaning,
            'literal': literal,
            'taught': True,
        }
        self._dirty.add(('idioms', lang))
        log.info(f"[TEACH] Idiom '{phrase}' → '{meaning}'")

    # ── Persistence ──────────────────────────────────────

    def save(self):
        """Write all dirty (modified) JSON files to disk."""
        for category, lang in self._dirty:
            if category == 'vocab':
                path = os.path.join(self.data_dir, 'vocab', f'{lang}.json')
                self._save_json(path, self._vocab.get(lang, {}))
            elif category == 'normalize':
                path = os.path.join(self.data_dir, 'normalize', f'{lang}.json')
                self._save_json(path, self._normalize.get(lang, {}))
            elif category == 'idioms':
                path = os.path.join(self.data_dir, 'idioms', f'{lang}.json')
                self._save_json(path, self._idioms.get(lang, {}))

        saved = len(self._dirty)
        self._dirty.clear()
        if saved:
            log.info(f"[LEARN] Saved {saved} updated data files")
        return saved

    def _save_json(self, path: str, data: dict):
        """Write JSON file, creating directories if needed."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    # ── Query (check what's been learned) ────────────────

    def is_known(self, word: str, lang: str = 'en') -> bool:
        """Check if a word is in the vocabulary."""
        return word.lower() in self._get_known_words(lang)

    def pending_count(self) -> int:
        """How many words are in the learning queue."""
        return len(self.queue.unknown_words)

    def learned_words(self, lang: str = 'en') -> List[str]:
        """List words that were learned (not in original vocab)."""
        self._ensure_loaded(lang)
        vocab = self._vocab.get(lang, {}).get('words', {})
        return [w for w, d in vocab.items() if d.get('learned') or d.get('taught')]
