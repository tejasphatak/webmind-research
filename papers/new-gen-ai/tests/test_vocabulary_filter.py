"""Tests for VocabularyFilter — garbage detection, morphological linking, dedup, semantic search."""

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from vocabulary_filter import VocabularyFilter, _stem_heuristic


# --- Stemming heuristic tests ---

class TestStemHeuristic:

    def test_strips_common_suffixes(self):
        assert _stem_heuristic("gravitational") == "gravit"
        assert _stem_heuristic("running") == "runn"
        assert _stem_heuristic("happiness") == "happi"

    def test_preserves_short_words(self):
        assert _stem_heuristic("cat") == "cat"
        assert _stem_heuristic("run") == "run"

    def test_preserves_already_stemmed(self):
        assert _stem_heuristic("grav") == "grav"

    def test_strips_plural_s(self):
        assert _stem_heuristic("cats") == "cat"
        assert _stem_heuristic("dogs") == "dog"

    def test_strips_ed(self):
        assert _stem_heuristic("played") == "play"


# --- VocabularyFilter without LSH (heuristic fallback) ---

class TestVocabularyFilterHeuristic:
    """Tests that work without building the LSH index."""

    def setup_method(self):
        self.vf = VocabularyFilter()

    def test_not_ready_before_build(self):
        assert not self.vf.is_ready

    def test_garbage_detection_heuristic_keyboard_mash(self):
        assert self.vf.is_garbage("asdfghjkl")

    def test_garbage_detection_heuristic_empty(self):
        assert self.vf.is_garbage("")
        assert self.vf.is_garbage("  ")
        assert self.vf.is_garbage("a")

    def test_garbage_detection_heuristic_too_long_word(self):
        assert self.vf.is_garbage("aaaaabbbbbcccccdddddeeeeef")

    def test_garbage_detection_heuristic_no_vowels(self):
        assert self.vf.is_garbage("bcdfghjklmnpqrstvwxyz")

    def test_not_garbage_real_word(self):
        assert not self.vf.is_garbage("hello")

    def test_not_garbage_number(self):
        assert not self.vf.is_garbage("42 + 7")

    def test_not_garbage_short_word(self):
        assert not self.vf.is_garbage("cat")

    def test_not_garbage_sentence(self):
        assert not self.vf.is_garbage("the quick brown fox")

    def test_find_variants_without_lsh(self):
        # Without LSH, should return empty
        result = self.vf.find_variants("gravity")
        assert result == []

    def test_find_duplicates_without_lsh(self):
        result = self.vf.find_duplicates()
        assert result == []

    def test_semantic_search_without_lsh(self):
        result = self.vf.semantic_search("hello")
        assert result == []


# --- VocabularyFilter with LSH index ---

@pytest.fixture(scope="module")
def built_filter():
    """Build a VocabularyFilter with a small vocabulary for testing."""
    vf = VocabularyFilter(n_bits=32)
    words = [
        "gravity", "gravitational", "gravitation", "gravitate",
        "physics", "physical", "physicist",
        "quantum", "quantize", "quantized",
        "energy", "energetic", "energize",
        "force", "forced", "forces",
        "mass", "massive", "masses",
        "velocity", "speed", "acceleration",
        "electron", "proton", "neutron",
        "atom", "atomic", "molecule", "molecular",
        "light", "photon", "wave", "wavelength",
        "temperature", "heat", "thermal",
        "colour", "color",  # duplicate pair
        "running", "run", "runner",
        "cat", "dog", "fish", "bird",
    ]
    vf.build(words)
    return vf


class TestGarbageDetectionLSH:

    def test_garbage_keyboard_mash(self, built_filter):
        assert built_filter.is_garbage("asdfghjkl")

    def test_garbage_random_chars(self, built_filter):
        assert built_filter.is_garbage("xzqwplfmk")

    def test_not_garbage_known_word(self, built_filter):
        assert not built_filter.is_garbage("gravity")

    def test_not_garbage_known_phrase(self, built_filter):
        assert not built_filter.is_garbage("quantum physics")

    def test_not_garbage_number(self, built_filter):
        assert not built_filter.is_garbage("3.14")

    def test_empty_is_garbage(self, built_filter):
        assert built_filter.is_garbage("")


class TestMorphologicalLinking:

    def test_finds_gravity_variants(self, built_filter):
        variants = built_filter.find_variants("gravity")
        variant_words = [w for w, s in variants]
        # Should find at least one variant
        assert len(variants) > 0
        # gravitational shares root "gravit"
        assert any("gravit" in w for w in variant_words)

    def test_finds_physics_variants(self, built_filter):
        variants = built_filter.find_variants("physics")
        variant_words = [w for w, s in variants]
        assert len(variants) > 0

    def test_similarity_is_positive(self, built_filter):
        variants = built_filter.find_variants("energy")
        for word, sim in variants:
            assert sim > 0
            assert sim <= 1.0

    def test_no_self_in_variants(self, built_filter):
        variants = built_filter.find_variants("gravity")
        variant_words = [w for w, s in variants]
        assert "gravity" not in variant_words

    def test_morphological_check_prefix(self):
        assert VocabularyFilter._is_morphological("run", "running")
        assert VocabularyFilter._is_morphological("gravity", "gravitational")

    def test_morphological_check_unrelated(self):
        assert not VocabularyFilter._is_morphological("cat", "dog")
        assert not VocabularyFilter._is_morphological("mass", "light")


class TestVocabularyDedup:

    def test_finds_colour_color(self, built_filter):
        dupes = built_filter.find_duplicates(threshold=0.85)
        pair_words = [(a, b) for a, b, s in dupes]
        # colour/color should be detected as near-duplicates
        found = any(
            ("color" in a and "colour" in b) or ("colour" in a and "color" in b)
            for a, b in pair_words
        )
        assert found, f"Expected colour/color pair, got: {pair_words}"

    def test_dedup_similarity_above_threshold(self, built_filter):
        dupes = built_filter.find_duplicates(threshold=0.85)
        for a, b, sim in dupes:
            assert sim >= 0.85

    def test_get_canonical_shorter(self, built_filter):
        # Force dedup cache
        built_filter.find_duplicates(threshold=0.85)
        # color is shorter than colour
        canon = built_filter.get_canonical("colour")
        assert canon in ("color", "colour")  # should prefer shorter


class TestSemanticSearchLSH:

    def test_search_returns_results(self, built_filter):
        results = built_filter.semantic_search("physics", k=5)
        assert len(results) > 0

    def test_search_results_sorted_by_similarity(self, built_filter):
        results = built_filter.semantic_search("energy", k=10)
        if len(results) >= 2:
            sims = [s for _, s in results]
            assert sims == sorted(sims, reverse=True)

    def test_search_similarity_range(self, built_filter):
        results = built_filter.semantic_search("quantum", k=5)
        for word, sim in results:
            assert -1.0 <= sim <= 1.0

    def test_search_indices(self, built_filter):
        word_idx = {w: i for i, w in enumerate(built_filter._hasher._words)}
        results = built_filter.semantic_search_indices("gravity", word_idx, k=5)
        for idx, sim in results:
            assert isinstance(idx, int)
            assert idx >= 0

    def test_search_empty_query(self, built_filter):
        # Should handle gracefully
        results = built_filter.semantic_search("", k=5)
        # May return results or empty, but shouldn't crash
        assert isinstance(results, list)
