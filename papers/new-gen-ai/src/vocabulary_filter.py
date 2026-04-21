"""
Vocabulary Filter — garbage detection, morphological linking, dedup, and O(1) search.

Sits between the API layer and the brain. Four capabilities:

1. Garbage detection: reject "asdfghjkl" before it touches the graph
2. Morphological linking: "gravitational" → "gravity" (auto-link variants)
3. Vocabulary dedup: merge near-duplicate words via LSH bucket proximity
4. O(1) semantic search: LSH bucket lookup as fast-path before convergence

All four use the same SemanticHasher (LSH over MiniLM embeddings).
The hasher is optional — if not built/loaded, features degrade gracefully
to heuristic fallbacks.

Usage:
    vf = VocabularyFilter()
    vf.build(word_list)                       # encode + hash all words
    vf.is_garbage("asdfghjkl")                # → True
    vf.find_variants("gravitational")         # → [("gravity", 0.89), ...]
    vf.find_duplicates(threshold=0.92)        # → [("colour", "color", 0.97), ...]
    vf.semantic_search("quantum", k=5)        # → [("physics", 0.84), ...]
"""

import re
from typing import List, Tuple, Optional, Dict, Set
from semantic_hash import SemanticHasher


# Common English suffixes for morphological stripping
_SUFFIXES = [
    'ational', 'tional', 'ation', 'ness', 'ment', 'ible', 'able',
    'ful', 'less', 'ous', 'ive', 'ing', 'tion', 'sion',
    'ity', 'ism', 'ist', 'ence', 'ance', 'ally', 'ical',
    'ment', 'ling', 'ness', 'ship', 'ward', 'wise',
    'ized', 'ised', 'ize', 'ise', 'ify', 'fy',
    'ly', 'ed', 'er', 'es', 'en', 'al', 'ty',
    's',
]


def _stem_heuristic(word: str) -> str:
    """Cheap stem: strip common suffixes. Not Porter — just enough to
    group 'gravitational' with 'gravity', 'running' with 'run'."""
    w = word.lower().strip()
    for suffix in _SUFFIXES:
        if w.endswith(suffix) and len(w) - len(suffix) >= 3:
            return w[:-len(suffix)]
    return w


class VocabularyFilter:
    """Vocabulary-level operations over an LSH index."""

    def __init__(self, n_bits: int = 64, model_name: str = "all-MiniLM-L6-v2"):
        self._hasher = SemanticHasher(n_bits=n_bits, model_name=model_name)
        self._built = False

        # Caches
        self._stem_groups: Dict[str, List[str]] = {}   # stem → [words]
        self._dedup_pairs: List[Tuple[str, str, float]] = []

    @property
    def is_ready(self) -> bool:
        return self._built

    def build(self, words: List[str], batch_size: int = 512,
              save_path: Optional[str] = None) -> dict:
        """Build the LSH index + precompute stem groups.

        Returns stats dict from SemanticHasher.build().
        """
        stats = self._hasher.build(words, batch_size=batch_size,
                                   save_path=save_path)
        self._built = True

        # Build stem groups for morphological linking
        # Use both exact stem AND prefix-4 grouping to catch variants
        # that strip to slightly different stems (gravity→grav, gravitational→gravit)
        self._stem_groups.clear()
        for word in words:
            stem = _stem_heuristic(word)
            if stem not in self._stem_groups:
                self._stem_groups[stem] = []
            self._stem_groups[stem].append(word)
            # Also group by first 4 chars of stem (catches near-miss stems)
            if len(stem) >= 4:
                prefix_key = stem[:4]
                if prefix_key not in self._stem_groups:
                    self._stem_groups[prefix_key] = []
                self._stem_groups[prefix_key].append(word)

        return stats

    def load(self, path: str):
        """Load a pre-built LSH index from disk."""
        self._hasher.load(path)
        self._built = True

        # Rebuild stem groups
        self._stem_groups.clear()
        for word in self._hasher._words:
            stem = _stem_heuristic(word)
            if stem not in self._stem_groups:
                self._stem_groups[stem] = []
            self._stem_groups[stem].append(word)
            if len(stem) >= 4:
                prefix_key = stem[:4]
                if prefix_key not in self._stem_groups:
                    self._stem_groups[prefix_key] = []
                self._stem_groups[prefix_key].append(word)

    def save(self, path: str):
        """Save the LSH index to disk."""
        self._hasher.save(path)

    # ------------------------------------------------------------------
    # 1. Garbage Detection
    # ------------------------------------------------------------------

    def is_garbage(self, text: str) -> bool:
        """Returns True if text is garbage (keyboard mash, random chars, etc.).

        Two-layer check:
          Layer 1 (fast): heuristic — vowel ratio, word length, alpha ratio.
          Layer 2 (LSH):  if heuristic is uncertain, check semantic similarity
                          to known vocabulary via LSH bucket lookup.
                          For multi-word input, checks each word individually —
                          if any word is meaningful, the input passes.
        """
        if not text or len(text.strip()) < 2:
            return True

        stripped = text.strip().lower()

        # Pure numbers are fine
        if re.match(r'^[\d\s\.\,\+\-\*\/\=\%\^\(\)]+$', stripped):
            return False

        words = re.findall(r'[a-zA-Z]+', stripped)
        if not words:
            return True

        # Layer 1: heuristic checks
        for word in words:
            if len(word) > 20:
                return True  # single "word" longer than 20 chars = garbage
            if len(word) > 4:
                vowels = sum(1 for c in word.lower() if c in 'aeiouy')
                if vowels == 0:
                    return True  # 5+ chars with no vowels
                consonant_ratio = (len(word) - vowels) / len(word)
                if consonant_ratio > 0.85 and len(word) > 6:
                    return True  # overwhelming consonants

        # Short text that passes heuristics: not garbage
        if len(words) == 1 and len(words[0]) <= 4:
            return False

        # Layer 2: LSH check (if available)
        if self._built:
            # For multi-word input: check each word individually.
            # If ANY word is meaningful, the input is not garbage.
            if len(words) >= 2:
                for word in words:
                    if len(word) >= 2 and self._hasher.is_meaningful(word, min_similarity=0.15):
                        return False
                return True
            else:
                return not self._hasher.is_meaningful(text, min_similarity=0.15)

        return False

    # ------------------------------------------------------------------
    # 2. Morphological Variant Linking
    # ------------------------------------------------------------------

    def find_variants(self, word: str, min_similarity: float = 0.6) -> List[Tuple[str, float]]:
        """Find morphological variants of a word.

        Combines heuristic stemming with LSH similarity:
        1. Find words sharing the same stem (cheap)
        2. Verify with embedding similarity (accurate)

        Returns [(variant, similarity), ...] sorted by similarity desc.
        """
        word = word.lower().strip()
        candidates = set()

        # Stem-based candidates
        stem = _stem_heuristic(word)
        for group_word in self._stem_groups.get(stem, []):
            if group_word != word:
                candidates.add(group_word)

        # LSH-based candidates (finds things stemming misses)
        if self._built:
            lsh_results = self._hasher.search(word, k=20, hamming_radius=2)
            for candidate, sim in lsh_results:
                if candidate != word and sim >= min_similarity:
                    # Check if they share a morphological root
                    # (not just topically similar — "gravity" and "mass" are
                    # similar but not variants)
                    if self._is_morphological(word, candidate):
                        candidates.add(candidate)

        if not candidates:
            return []

        # Score all candidates by embedding similarity
        results = []
        for candidate in candidates:
            if self._built:
                sim = self._hasher.word_similarity(word, candidate)
            else:
                sim = 0.5  # default when no embeddings
            if sim >= min_similarity:
                results.append((candidate, sim))

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    @staticmethod
    def _is_morphological(word1: str, word2: str) -> bool:
        """Check if two words are likely morphological variants.

        Heuristic: one is a prefix of the other, or they share a long
        common prefix (>= 60% of the shorter word).
        """
        w1, w2 = word1.lower(), word2.lower()
        if w1 == w2:
            return False
        # One is prefix of the other
        if w1.startswith(w2) or w2.startswith(w1):
            return True
        # Shared prefix >= 60% of shorter word
        min_len = min(len(w1), len(w2))
        common = 0
        for a, b in zip(w1, w2):
            if a == b:
                common += 1
            else:
                break
        return common >= max(3, int(min_len * 0.6))

    # ------------------------------------------------------------------
    # 3. Vocabulary Dedup
    # ------------------------------------------------------------------

    def find_duplicates(self, threshold: float = 0.92) -> List[Tuple[str, str, float]]:
        """Find near-duplicate word pairs via LSH bucket proximity.

        Words in the same or adjacent buckets (1-bit Hamming) with
        cosine similarity >= threshold are reported as duplicates.

        Returns [(word_a, word_b, similarity), ...] sorted by similarity desc.
        """
        if not self._built:
            return []

        if self._dedup_pairs:
            return self._dedup_pairs

        import numpy as np
        pairs = []
        seen = set()

        # Check within each bucket
        for bucket_hash, indices in self._hasher._buckets.items():
            if len(indices) < 2:
                continue
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    idx_a, idx_b = indices[i], indices[j]
                    word_a = self._hasher._words[idx_a]
                    word_b = self._hasher._words[idx_b]
                    pair_key = (min(word_a, word_b), max(word_a, word_b))
                    if pair_key in seen:
                        continue
                    seen.add(pair_key)
                    sim = float(np.dot(
                        self._hasher._embeddings[idx_a],
                        self._hasher._embeddings[idx_b],
                    ))
                    if sim >= threshold:
                        pairs.append((word_a, word_b, sim))

        # Also check morphological pairs (stem-based) — they might be
        # in adjacent buckets that the bucket-only check misses
        for stem, group in self._stem_groups.items():
            if len(group) < 2:
                continue
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    word_a, word_b = group[i], group[j]
                    pair_key = (min(word_a, word_b), max(word_a, word_b))
                    if pair_key in seen:
                        continue
                    seen.add(pair_key)
                    idx_a = self._hasher._word_idx.get(word_a)
                    idx_b = self._hasher._word_idx.get(word_b)
                    if idx_a is None or idx_b is None:
                        continue
                    sim = float(np.dot(
                        self._hasher._embeddings[idx_a],
                        self._hasher._embeddings[idx_b],
                    ))
                    if sim >= threshold:
                        pairs.append((word_a, word_b, sim))

        pairs.sort(key=lambda x: x[2], reverse=True)
        self._dedup_pairs = pairs
        return pairs

    def get_canonical(self, word: str, word_freq: Optional[Dict[str, int]] = None) -> str:
        """Get the canonical form of a word (shortest/most frequent variant).

        If word has duplicates, returns the preferred form.
        Falls back to the word itself if no duplicates found.
        """
        word = word.lower().strip()

        # Check dedup pairs
        for word_a, word_b, sim in self._dedup_pairs:
            if word == word_a or word == word_b:
                other = word_b if word == word_a else word_a
                if word_freq:
                    # Prefer more frequent
                    if word_freq.get(other, 0) > word_freq.get(word, 0):
                        return other
                else:
                    # Prefer shorter
                    if len(other) < len(word):
                        return other
                return word

        return word

    # ------------------------------------------------------------------
    # 4. O(1) Semantic Search
    # ------------------------------------------------------------------

    def semantic_search(self, query: str, k: int = 10,
                        hamming_radius: int = 3) -> List[Tuple[str, float]]:
        """O(1) semantic search via LSH bucket lookup + cosine refinement.

        Returns [(word, similarity), ...] sorted by similarity desc.
        Much faster than full convergence for simple lookups.
        """
        if not self._built:
            return []
        return self._hasher.search(query, k=k, hamming_radius=hamming_radius)

    def semantic_search_indices(self, query: str, word_idx: Dict[str, int],
                                k: int = 10) -> List[Tuple[int, float]]:
        """Like semantic_search but returns word indices for brain integration.

        Returns [(word_idx, similarity), ...] for direct use in convergence.
        """
        results = self.semantic_search(query, k=k)
        indexed = []
        for word, sim in results:
            idx = word_idx.get(word)
            if idx is not None:
                indexed.append((idx, sim))
        return indexed
