"""
Tests for the Encoder.

Verifies against HLD spec:
- Word-level: direct lookup in embedding table
- Sentence-level: weighted average of word vectors
- OOV: zero vector (honest — "I don't have this word")
- Vectors are normalized (for cosine similarity)
- No training — just lookup

Uses synthetic vocabulary to avoid downloading GloVe in tests.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from encoder import Encoder

DIM = 300


def make_vocab():
    """Small synthetic vocabulary for testing."""
    rng = np.random.RandomState(42)
    words = {
        "shakespeare": rng.randn(DIM).astype(np.float32),
        "hamlet": rng.randn(DIM).astype(np.float32),
        "wrote": rng.randn(DIM).astype(np.float32),
        "playwright": rng.randn(DIM).astype(np.float32),
        "english": rng.randn(DIM).astype(np.float32),
        "cat": rng.randn(DIM).astype(np.float32),
        "dog": rng.randn(DIM).astype(np.float32),
        "the": rng.randn(DIM).astype(np.float32),
        "who": rng.randn(DIM).astype(np.float32),
        "is": rng.randn(DIM).astype(np.float32),
    }
    return words


def make_encoder():
    enc = Encoder(data_dir="/tmp/test_encoder", dim=DIM)
    enc.load_from_dict(make_vocab())
    return enc


class TestEncoderWord:

    def test_known_word_returns_vector(self):
        enc = make_encoder()
        v = enc.encode_word("shakespeare")
        assert v.shape == (DIM,)
        assert np.linalg.norm(v) > 0

    def test_known_word_is_normalized(self):
        enc = make_encoder()
        v = enc.encode_word("hamlet")
        assert abs(np.linalg.norm(v) - 1.0) < 1e-5

    def test_oov_returns_zero(self):
        """OOV = zero vector. Honest about not knowing."""
        enc = make_encoder()
        v = enc.encode_word("glorpnax")
        assert np.all(v == 0)

    def test_case_insensitive(self):
        enc = make_encoder()
        v1 = enc.encode_word("Shakespeare")
        v2 = enc.encode_word("shakespeare")
        np.testing.assert_array_equal(v1, v2)

    def test_strips_whitespace(self):
        enc = make_encoder()
        v1 = enc.encode_word("  hamlet  ")
        v2 = enc.encode_word("hamlet")
        np.testing.assert_array_equal(v1, v2)

    def test_has_word(self):
        enc = make_encoder()
        assert enc.has_word("cat") is True
        assert enc.has_word("glorpnax") is False

    def test_returns_copy_not_reference(self):
        """Modifying returned vector should not change the vocab."""
        enc = make_encoder()
        v1 = enc.encode_word("cat")
        v1[0] = 999.0
        v2 = enc.encode_word("cat")
        assert v2[0] != 999.0


class TestEncoderSentence:

    def test_single_word_sentence(self):
        enc = make_encoder()
        v_word = enc.encode_word("hamlet")
        v_sent = enc.encode_sentence("hamlet")
        # Single word sentence should equal the word vector (both normalized)
        np.testing.assert_array_almost_equal(v_word, v_sent, decimal=5)

    def test_sentence_is_normalized(self):
        enc = make_encoder()
        v = enc.encode_sentence("who wrote hamlet")
        norm = np.linalg.norm(v)
        if norm > 0:
            assert abs(norm - 1.0) < 1e-5

    def test_all_oov_returns_zero(self):
        """Sentence with no known words → zero vector. Honest abstention."""
        enc = make_encoder()
        v = enc.encode_sentence("glorpnax zibble fweep")
        assert np.all(v == 0)

    def test_empty_string_returns_zero(self):
        enc = make_encoder()
        v = enc.encode_sentence("")
        assert np.all(v == 0)

    def test_mixed_known_oov(self):
        """OOV words are skipped, known words contribute."""
        enc = make_encoder()
        v_pure = enc.encode_sentence("hamlet")
        v_mixed = enc.encode_sentence("glorpnax hamlet zibble")
        # Should be close to "hamlet" since it's the only known word
        sim = float(np.dot(v_pure, v_mixed))
        assert sim > 0.99

    def test_different_sentences_different_vectors(self):
        enc = make_encoder()
        v1 = enc.encode_sentence("who wrote hamlet")
        v2 = enc.encode_sentence("the cat is english")
        # Different sentences should produce different vectors
        sim = float(np.dot(v1, v2))
        assert sim < 0.99

    def test_word_order_matters(self):
        """Position weighting means order changes the vector."""
        enc = make_encoder()
        v1 = enc.encode_sentence("cat dog")
        v2 = enc.encode_sentence("dog cat")
        # Should be similar but not identical
        sim = float(np.dot(v1, v2))
        assert sim > 0.9  # mostly the same words
        assert sim < 1.0  # but order differs

    def test_punctuation_stripped(self):
        enc = make_encoder()
        v1 = enc.encode_sentence("who wrote hamlet?")
        v2 = enc.encode_sentence("who wrote hamlet")
        np.testing.assert_array_almost_equal(v1, v2, decimal=5)

    def test_tokenization(self):
        enc = make_encoder()
        tokens = enc._tokenize("Who wrote Hamlet?")
        assert tokens == ["who", "wrote", "hamlet"]

    def test_tokenization_special_chars(self):
        enc = make_encoder()
        tokens = enc._tokenize("cat's dog-eared, the.")
        assert tokens == ["cat", "s", "dog", "eared", "the"]


class TestEncoderNearestWords:

    def test_nearest_to_itself(self):
        """A word's vector should be nearest to itself."""
        enc = make_encoder()
        v = enc.encode_word("hamlet")
        nearest = enc.nearest_words(v, k=1)
        assert len(nearest) == 1
        assert nearest[0][0] == "hamlet"
        assert abs(nearest[0][1] - 1.0) < 1e-5

    def test_nearest_returns_k(self):
        enc = make_encoder()
        v = enc.encode_word("cat")
        nearest = enc.nearest_words(v, k=3)
        assert len(nearest) == 3

    def test_nearest_sorted_by_similarity(self):
        enc = make_encoder()
        v = enc.encode_word("cat")
        nearest = enc.nearest_words(v, k=5)
        sims = [s for _, s in nearest]
        assert sims == sorted(sims, reverse=True)

    def test_nearest_zero_vector(self):
        """Zero vector = OOV. No nearest words."""
        enc = make_encoder()
        nearest = enc.nearest_words(np.zeros(DIM, dtype=np.float32))
        assert nearest == []


class TestEncoderVocab:

    def test_vocab_size(self):
        enc = make_encoder()
        assert enc.vocab_size == 10

    def test_load_from_dict(self):
        enc = Encoder(data_dir="/tmp/test", dim=DIM)
        assert enc.vocab_size == 0
        enc.load_from_dict({"hello": np.ones(DIM)})
        assert enc.vocab_size == 1

    def test_vectors_normalized_on_load(self):
        """All loaded vectors should be unit-length."""
        enc = Encoder(data_dir="/tmp/test", dim=DIM)
        enc.load_from_dict({"word": np.ones(DIM) * 5.0})
        v = enc.encode_word("word")
        assert abs(np.linalg.norm(v) - 1.0) < 1e-5


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
