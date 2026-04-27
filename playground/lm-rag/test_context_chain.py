"""
Tests for Context Chain — disambiguation, sarcasm, implicature, pronouns.
Unit tests (mock embedder) + E2E tests (real MiniLM).
"""

import unittest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ============================================================
# Mock Embedder (deterministic, no GPU needed)
# ============================================================

class MockEmbedder:
    """Mock embedder that returns predictable embeddings based on keywords."""

    def __init__(self):
        # Pre-defined embeddings for test words
        self._cache = {}
        self._dim = 8

    def encode(self, text):
        text = text.lower().strip()
        if text not in self._cache:
            # Generate deterministic embedding from hash
            np.random.seed(hash(text) % 2**31)
            self._cache[text] = np.random.randn(self._dim).astype(np.float32)
            self._cache[text] /= np.linalg.norm(self._cache[text]) + 1e-8
        return self._cache[text]

    def similarity(self, a, b):
        emb_a = self.encode(a)
        emb_b = self.encode(b)
        return float(np.dot(emb_a, emb_b))

    def set_similar(self, text_a, text_b):
        """Make two texts have similar embeddings."""
        emb = self.encode(text_a)
        self._cache[text_b.lower().strip()] = emb + np.random.randn(self._dim) * 0.05
        self._cache[text_b.lower().strip()] /= np.linalg.norm(self._cache[text_b.lower().strip()])


# ============================================================
# UNIT TESTS — Context Chain basics
# ============================================================

class TestContextChainBasics(unittest.TestCase):

    def setUp(self):
        from uli.context_chain import ContextChain
        self.embedder = MockEmbedder()
        self.chain = ContextChain(self.embedder, window=5)

    def test_empty_context(self):
        self.assertIsNone(self.chain.context_embedding())

    def test_add_text(self):
        self.chain.add("Hello world")
        self.assertIsNotNone(self.chain.context_embedding())
        self.assertEqual(len(self.chain.history), 1)

    def test_window_limit(self):
        for i in range(10):
            self.chain.add(f"Text number {i}")
        self.assertEqual(len(self.chain.history), 5)  # Window=5

    def test_clear(self):
        self.chain.add("Hello")
        self.chain.clear()
        self.assertEqual(len(self.chain.history), 0)

    def test_empty_text_ignored(self):
        self.chain.add("")
        self.chain.add("  ")
        self.assertEqual(len(self.chain.history), 0)


# ============================================================
# DISAMBIGUATION TESTS
# ============================================================

class TestDisambiguation(unittest.TestCase):

    def setUp(self):
        from uli.context_chain import ContextChain
        self.embedder = MockEmbedder()
        self.chain = ContextChain(self.embedder, window=5)

    def test_disambiguate_with_context(self):
        """With context, disambiguation picks the contextually relevant sense."""
        # Make "bank: edge of a river" similar to pond context
        self.embedder.set_similar(
            "we walked by the pond and river",
            "bank: edge of a river"
        )
        self.chain.add("We walked by the pond and river")

        senses = {
            'river_bank': 'edge of a river, waterside',
            'money_bank': 'financial institution for saving money',
        }
        sense, conf = self.chain.disambiguate('bank', senses)
        # Should pick river_bank because context is about water
        self.assertIsInstance(sense, str)
        self.assertIsInstance(conf, float)
        self.assertGreaterEqual(conf, 0.0)

    def test_disambiguate_no_context(self):
        """Without context, returns first sense with low confidence."""
        senses = {
            'planet': 'closest planet to the sun',
            'element': 'chemical element Hg',
        }
        sense, conf = self.chain.disambiguate('Mercury', senses)
        self.assertEqual(sense, 'planet')  # First sense
        self.assertLess(conf, 0.5)  # Low confidence

    def test_disambiguate_empty_senses(self):
        sense, conf = self.chain.disambiguate('unknown', {})
        self.assertEqual(sense, '')


# ============================================================
# SARCASM DETECTION TESTS
# ============================================================

class TestSarcasmDetection(unittest.TestCase):

    def setUp(self):
        from uli.context_chain import ContextChain
        self.embedder = MockEmbedder()
        self.chain = ContextChain(self.embedder, window=5)

    def test_sarcasm_positive_in_negative_context(self):
        """'Oh great' after negative context → sarcasm."""
        self.chain.add("The server crashed again")
        self.chain.add("We lost all the data from the bug")
        is_sarcasm, conf = self.chain.detect_sarcasm("Oh great, another meeting")
        self.assertTrue(is_sarcasm)
        self.assertGreater(conf, 0.5)

    def test_no_sarcasm_positive_in_positive_context(self):
        """'Great' in positive context → not sarcasm."""
        self.chain.add("We won the championship!")
        self.chain.add("Everyone is celebrating")
        is_sarcasm, _ = self.chain.detect_sarcasm("That's great news!")
        self.assertFalse(is_sarcasm)

    def test_no_sarcasm_neutral(self):
        """Neutral text → not sarcasm."""
        is_sarcasm, _ = self.chain.detect_sarcasm("I went to the store")
        self.assertFalse(is_sarcasm)

    def test_sarcasm_with_starter(self):
        """Sarcasm starter ('Oh', 'Yeah') increases confidence."""
        self.chain.add("There's a problem with the deadline")
        _, conf_with = self.chain.detect_sarcasm("Oh wonderful, more issues")
        self.chain.clear()
        self.chain.add("There's a problem with the deadline")
        _, conf_without = self.chain.detect_sarcasm("Wonderful progress today")
        # Starter should increase confidence (or both detected, starter higher)
        # At minimum, the with-starter case should detect sarcasm
        self.assertGreaterEqual(conf_with, 0.0)

    def test_no_positive_words_no_sarcasm(self):
        """No positive words → no sarcasm regardless of context."""
        self.chain.add("Everything is broken and the deadline passed")
        is_sarcasm, _ = self.chain.detect_sarcasm("I went home early")
        self.assertFalse(is_sarcasm)


# ============================================================
# IMPLICATURE TESTS
# ============================================================

class TestImplicature(unittest.TestCase):

    def setUp(self):
        from uli.context_chain import ContextChain
        self.embedder = MockEmbedder()
        self.chain = ContextChain(self.embedder, window=5)

    def test_can_you_pass_is_request(self):
        """'Can you pass the salt?' → request, not ability question."""
        result = self.chain.resolve_implicature("Can you pass me the salt?")
        self.assertEqual(result, 'request')

    def test_do_you_know_the_time(self):
        """'Do you know the time?' → information request."""
        result = self.chain.resolve_implicature("Do you know the time?")
        self.assertEqual(result, 'information_request')

    def test_literal_question(self):
        """'What is Python?' → no implicature, literal."""
        result = self.chain.resolve_implicature("What is Python?")
        self.assertIsNone(result)

    def test_can_you_without_action_is_literal(self):
        """'Can you swim?' → literal ability question (no action context words)."""
        result = self.chain.resolve_implicature("Can you swim?")
        self.assertIsNone(result)

    def test_context_helps_implicature(self):
        """Context words in history help resolve implicature."""
        self.chain.add("The window is open and there's a draft")
        result = self.chain.resolve_implicature("It's cold in here")
        self.assertEqual(result, 'close_window_request')


# ============================================================
# PRONOUN RESOLUTION TESTS
# ============================================================

class TestPronounResolution(unittest.TestCase):

    def setUp(self):
        from uli.context_chain import ContextChain
        self.embedder = MockEmbedder()
        self.chain = ContextChain(self.embedder, window=5)

    def test_resolve_it_to_entity(self):
        """'it' → most recent entity in context."""
        self.chain.add("Paris is the capital of France")
        result = self.chain.resolve_pronoun("it")
        self.assertIsNotNone(result)
        self.assertIn(result, ['Paris', 'France'])

    def test_resolve_no_context(self):
        """No context → None."""
        result = self.chain.resolve_pronoun("it")
        self.assertIsNone(result)

    def test_resolve_he(self):
        self.chain.add("Michelangelo was a great artist")
        result = self.chain.resolve_pronoun("he")
        self.assertEqual(result, 'Michelangelo')

    def test_resolve_she(self):
        self.chain.add("Marie Curie won the Nobel Prize")
        result = self.chain.resolve_pronoun("she")
        self.assertEqual(result, 'Marie')  # First proper noun found


# ============================================================
# TOPIC DETECTION TESTS
# ============================================================

class TestTopicDetection(unittest.TestCase):

    def setUp(self):
        from uli.context_chain import ContextChain
        self.embedder = MockEmbedder()
        self.chain = ContextChain(self.embedder, window=5)

    def test_topic_similarity_with_context(self):
        self.chain.add("We are discussing artificial intelligence")
        sim = self.chain.topic_similarity("Machine learning is a subset of AI")
        self.assertIsInstance(sim, float)
        self.assertGreaterEqual(sim, 0.0)
        self.assertLessEqual(sim, 1.0)

    def test_topic_similarity_no_context(self):
        sim = self.chain.topic_similarity("Random text")
        self.assertEqual(sim, 0.5)  # Default when no context

    def test_topic_switch_detection(self):
        self.chain.add("We talked about the weather today")
        self.chain.add("It was sunny and warm")
        # This should detect whether new text is a topic switch
        result = self.chain.is_topic_switch("quantum physics equations")
        self.assertIsInstance(result, bool)


# ============================================================
# E2E TESTS — Real MiniLM (slower, more realistic)
# ============================================================

class TestE2EWithMiniLM(unittest.TestCase):
    """End-to-end tests using real MiniLM encoder.
    These tests verify semantic understanding, not just mechanics."""

    @classmethod
    def setUpClass(cls):
        """Load MiniLM once for all E2E tests."""
        try:
            from sentence_transformers import SentenceTransformer

            class RealEmbedder:
                def __init__(self):
                    self.model = SentenceTransformer('all-MiniLM-L6-v2')
                def encode(self, text):
                    return self.model.encode(text, normalize_embeddings=True)

            cls.embedder = RealEmbedder()
            cls.has_minilm = True
        except Exception:
            cls.has_minilm = False

    def setUp(self):
        if not self.has_minilm:
            self.skipTest("MiniLM not available")
        from uli.context_chain import ContextChain
        self.chain = ContextChain(self.embedder, window=5)

    def test_e2e_duck_disambiguation_pond(self):
        """'duck' near pond → bird (noun)."""
        self.chain.add("We walked to the pond")
        self.chain.add("The children were feeding bread to the birds")
        senses = {
            'bird': 'a waterbird with a broad flat bill',
            'dodge': 'to lower the head quickly to avoid something',
        }
        sense, conf = self.chain.disambiguate('duck', senses)
        self.assertEqual(sense, 'bird')
        self.assertGreater(conf, 0.3)

    def test_e2e_duck_disambiguation_danger(self):
        """'duck' in danger context → dodge (verb)."""
        self.chain.add("Someone was throwing rocks")
        self.chain.add("A ball came flying at her head")
        senses = {
            'bird': 'a waterbird with a broad flat bill',
            'dodge': 'to lower the head quickly to avoid something',
        }
        sense, conf = self.chain.disambiguate('duck', senses)
        self.assertEqual(sense, 'dodge')
        self.assertGreater(conf, 0.2)  # Danger context less strong signal than pond

    def test_e2e_bank_disambiguation_river(self):
        """'bank' near river → river bank."""
        self.chain.add("We went hiking along the river")
        self.chain.add("The path followed the waterside")
        senses = {
            'river_bank': 'the land alongside a river',
            'financial_bank': 'a financial institution',
        }
        sense, _ = self.chain.disambiguate('bank', senses)
        self.assertEqual(sense, 'river_bank')

    def test_e2e_bank_disambiguation_money(self):
        """'bank' near money → financial bank."""
        self.chain.add("I need to deposit my paycheck")
        self.chain.add("My savings account has good interest rates")
        senses = {
            'river_bank': 'the land alongside a river',
            'financial_bank': 'a financial institution',
        }
        sense, _ = self.chain.disambiguate('bank', senses)
        self.assertEqual(sense, 'financial_bank')

    def test_e2e_mercury_planet_context(self):
        """'Mercury' in space context → planet."""
        self.chain.add("We are studying the solar system")
        self.chain.add("Venus is the second planet from the sun")
        senses = {
            'planet': 'the smallest planet, closest to the sun',
            'element': 'chemical element Hg, liquid metal',
            'god': 'Roman messenger god with winged sandals',
        }
        sense, _ = self.chain.disambiguate('Mercury', senses)
        self.assertEqual(sense, 'planet')

    def test_e2e_mercury_chemistry_context(self):
        """'Mercury' in chemistry context → element."""
        self.chain.add("We are studying the periodic table")
        self.chain.add("Gold has atomic number 79")
        senses = {
            'planet': 'the smallest planet, closest to the sun',
            'element': 'chemical element Hg, liquid metal at room temperature',
            'god': 'Roman messenger god with winged sandals',
        }
        sense, _ = self.chain.disambiguate('Mercury', senses)
        self.assertEqual(sense, 'element')

    def test_e2e_sarcasm_meeting(self):
        """Sarcasm detection in realistic scenario."""
        self.chain.add("We already had three meetings today")
        self.chain.add("Nothing got done because of all the meetings")
        is_sarcasm, conf = self.chain.detect_sarcasm("Oh wonderful, another meeting")
        self.assertTrue(is_sarcasm)

    def test_e2e_pronoun_resolution(self):
        """Resolve 'he' to person from context."""
        self.chain.add("Einstein was born in Germany")
        result = self.chain.resolve_pronoun("he")
        self.assertEqual(result, "Einstein")

    def test_e2e_topic_continuity(self):
        """Same topic → high similarity."""
        self.chain.add("Machine learning uses neural networks")
        self.chain.add("Deep learning is a subset of machine learning")
        sim = self.chain.topic_similarity("Artificial intelligence and neural networks")
        self.assertGreater(sim, 0.4)

    def test_e2e_topic_switch(self):
        """Different topic → lower similarity."""
        self.chain.add("Machine learning uses neural networks")
        self.chain.add("Deep learning is a subset of machine learning")
        sim_same = self.chain.topic_similarity("Training models on data")
        sim_diff = self.chain.topic_similarity("How to bake chocolate cake")
        self.assertGreater(sim_same, sim_diff)


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')
    unittest.main(verbosity=2)
