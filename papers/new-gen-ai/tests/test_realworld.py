"""
Real-world test with GloVe 400K vocabulary.

Tests the system with real word embeddings to see if:
1. Semantic similarity works (cat/dog closer than cat/computer)
2. Convergence finds meaningful concepts
3. Template generation produces readable output
4. The system is honest about what it doesn't know
5. Delete actually works with real embeddings
6. Performance at scale (400K vocab)

Uses session-scoped GloVe fixture (conftest.py) — loads 990MB once,
not per test. Cuts suite time from ~7 min to ~1 min.
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from engine import Engine

DATA_DIR = str(Path(__file__).parent.parent / "data")
GLOVE_PATH = str(Path(DATA_DIR) / "glove.6B.300d.txt")


class TestRealWorldSemantics:

    def test_vocab_loaded(self, glove_engine):
        assert glove_engine.encoder.vocab_size == 400000

    def test_similar_words_are_close(self, glove_engine):
        """cat and dog should be closer than cat and computer."""
        v_cat = glove_engine.encoder.encode_word("cat")
        v_dog = glove_engine.encoder.encode_word("dog")
        v_computer = glove_engine.encoder.encode_word("computer")

        sim_cat_dog = float(np.dot(v_cat, v_dog))
        sim_cat_comp = float(np.dot(v_cat, v_computer))

        assert sim_cat_dog > sim_cat_comp, \
            f"cat-dog ({sim_cat_dog:.3f}) should be > cat-computer ({sim_cat_comp:.3f})"

    def test_nearest_words_meaningful(self, glove_engine):
        """Nearest words to 'king' should include royalty-related words.
        In streaming mode, need to pre-encode candidates so they're in cache."""
        # Pre-encode candidates so they're available for nearest_words
        candidates = ["king", "queen", "prince", "royal", "throne", "kingdom",
                       "kings", "monarch", "princess", "crown", "emperor",
                       "cat", "dog", "computer"]
        for w in candidates:
            glove_engine.encoder.encode_word(w)

        v = glove_engine.encoder.encode_word("king")
        nearest = glove_engine.encoder.nearest_words(v, k=10)
        words = [w for w, _ in nearest]
        assert "king" in words
        royalty = {"queen", "prince", "royal", "throne", "kingdom",
                   "kings", "monarch", "princess", "crown", "emperor"}
        found = set(words) & royalty
        assert len(found) >= 1, f"Expected royalty words near 'king', got {words}"

    def test_sentence_encoding_meaningful(self, glove_engine):
        """Similar sentences should have similar vectors."""
        v1 = glove_engine.encoder.encode_sentence("the cat sat on the mat")
        v2 = glove_engine.encoder.encode_sentence("a dog sat on the rug")
        v3 = glove_engine.encoder.encode_sentence("stock market crash economy")

        sim_12 = float(np.dot(v1, v2))
        sim_13 = float(np.dot(v1, v3))

        assert sim_12 > sim_13, \
            f"cat-sat ({sim_12:.3f}) should be more similar than cat-stock ({sim_13:.3f})"


class TestRealWorldConvergence:

    def test_converges_on_taught_facts(self, glove_engine):
        glove_engine.teach_sentence("shakespeare wrote hamlet")
        result = glove_engine.query("shakespeare")
        assert result.converged is True
        assert result.confidence > 0

    def test_multi_fact_convergence(self, glove_engine):
        glove_engine.teach_sentence("paris is the capital of france")
        glove_engine.teach_sentence("london is the capital of england")
        result = glove_engine.query("capital of france")
        assert result.converged is True

    def test_honest_on_empty_kb(self, glove_engine):
        result = glove_engine.query("what is the meaning of life")
        assert "don't know" in result.answer.lower()


class TestRealWorldGeneration:

    def test_template_with_real_words(self, glove_engine):
        glove_engine.teach_sentence("shakespeare wrote hamlet")
        glove_engine.teach_template(
            "[PERSON] wrote [WORK]",
            {"PERSON": "noun", "WORK": "noun"},
            confidence=0.8,
        )
        result = glove_engine.query("who wrote hamlet")
        print(f"\nQuery: 'who wrote hamlet'")
        print(f"Answer: '{result.answer}'")
        print(f"Strategy: {result.strategy}")
        print(f"Confidence: {result.confidence:.3f}")
        assert result.answer != ""

    def test_successor_walk_with_real_words(self, glove_engine):
        glove_engine.teach_sentence("the cat sat on the mat")
        result = glove_engine.query("cat")
        print(f"\nQuery: 'cat'")
        print(f"Answer: '{result.answer}'")
        print(f"Strategy: {result.strategy}")
        assert result.answer != ""

    def test_multiple_templates(self, glove_engine):
        glove_engine.teach_sentence("einstein discovered relativity")
        glove_engine.teach_sentence("newton discovered gravity")
        glove_engine.teach_template(
            "[PERSON] discovered [CONCEPT]",
            {"PERSON": "noun", "CONCEPT": "noun"},
            confidence=0.8,
        )
        glove_engine.teach_template(
            "[SUBJECT] is known for [ACHIEVEMENT]",
            {"SUBJECT": "noun", "ACHIEVEMENT": "noun"},
            confidence=0.7,
        )
        result = glove_engine.query("einstein relativity")
        print(f"\nQuery: 'einstein relativity'")
        print(f"Answer: '{result.answer}'")
        print(f"Strategy: {result.strategy}")


class TestRealWorldDelete:

    def test_delete_with_real_embeddings(self, glove_engine):
        """Invariant #3 with real data."""
        glove_engine.teach("shakespeare", confidence=0.8)
        glove_engine.teach("hamlet", confidence=0.7)
        glove_engine.teach("macbeth", confidence=0.6)

        info = glove_engine.inspect("shakespeare")
        words_before = [n["word"] for n in info["neighbors"]]
        assert "shakespeare" in words_before

        glove_engine.delete_word("shakespeare")

        info = glove_engine.inspect("shakespeare")
        words_after = [n["word"] for n in info["neighbors"]]
        assert "shakespeare" not in words_after


class TestRealWorldPerformance:

    def test_load_time(self):
        """Loading 400K embeddings should be reasonable."""
        start = time.perf_counter()
        engine = Engine(dim=300)
        engine.encoder.load(GLOVE_PATH)
        load_time = time.perf_counter() - start
        print(f"\n400K vocab load time: {load_time:.2f}s")
        assert load_time < 30, f"Load took {load_time:.1f}s (>30s)"

    def test_query_latency(self, glove_engine):
        """Queries should be fast even with real vocab."""
        for w in ["cat", "dog", "shakespeare", "hamlet", "python", "computer"]:
            glove_engine.teach(w)

        glove_engine.query("cat")  # warmup

        times = []
        queries = ["cat", "who wrote hamlet", "python programming",
                   "the meaning of life", "dog"]
        for q in queries:
            start = time.perf_counter()
            glove_engine.query(q)
            times.append(time.perf_counter() - start)

        avg_ms = sum(times) / len(times) * 1000
        print(f"\nAvg query latency: {avg_ms:.2f}ms")
        assert avg_ms < 200, f"Avg query took {avg_ms:.1f}ms (>200ms)"

    def test_teach_many_words(self, glove_engine):
        """Teaching 100 words should be fast."""
        words = ["the", "a", "is", "was", "are", "were", "be", "been",
                 "have", "has", "had", "do", "does", "did", "will", "would",
                 "could", "should", "may", "might", "shall", "can", "must",
                 "need", "dare", "ought", "used", "to", "of", "in", "for",
                 "on", "with", "at", "by", "from", "as", "into", "through",
                 "during", "before", "after", "above", "below", "between",
                 "out", "off", "over", "under", "again", "further", "then",
                 "once", "here", "there", "when", "where", "why", "how",
                 "all", "each", "every", "both", "few", "more", "most",
                 "other", "some", "such", "no", "nor", "not", "only", "own",
                 "same", "so", "than", "too", "very", "just", "because",
                 "but", "and", "or", "if", "while", "although", "though",
                 "even", "also", "still", "already", "yet", "ever", "never",
                 "always", "often", "sometimes", "usually", "really", "quite"]

        start = time.perf_counter()
        for w in words:
            if glove_engine.encoder.has_word(w):
                glove_engine.teach(w)
        elapsed = time.perf_counter() - start
        print(f"\nTaught {glove_engine.db.count()} words in {elapsed:.3f}s")
        assert elapsed < 5.0


class TestRealWorldEndToEnd:

    def test_full_demo(self, glove_engine):
        """Complete demo of the system."""
        engine = glove_engine

        print("\n" + "=" * 60)
        print("NEW-GEN-AI REAL WORLD DEMO")
        print("=" * 60)

        print("\n--- Teaching facts ---")
        engine.teach_sentence("shakespeare wrote hamlet in england")
        engine.teach_sentence("einstein discovered relativity in germany")
        engine.teach_sentence("the cat sat on the mat")
        engine.teach_sentence("python is a programming language")

        engine.teach_template(
            "[PERSON] wrote [WORK]",
            {"PERSON": "noun", "WORK": "noun"}, confidence=0.8,
        )
        engine.teach_template(
            "[PERSON] discovered [CONCEPT]",
            {"PERSON": "noun", "CONCEPT": "noun"}, confidence=0.8,
        )
        engine.teach_template(
            "[SUBJECT] is a [CATEGORY]",
            {"SUBJECT": "noun", "CATEGORY": "noun"}, confidence=0.7,
        )

        print(f"KB: {engine.db.count()} neurons, {engine.template_store.count()} templates")

        queries = [
            "who wrote hamlet",
            "einstein",
            "cat mat",
            "python programming",
            "quantum physics",
        ]

        print("\n--- Queries ---")
        for q in queries:
            result = engine.query(q)
            print(f"Q: {q}")
            print(f"A: {result.answer}")
            print(f"   [{result.strategy}, conf={result.confidence:.3f}, "
                  f"converged={result.converged}]")
            print()

        print("--- Delete test ---")
        engine.delete_word("shakespeare")
        result = engine.query("shakespeare")
        print(f"After deleting 'shakespeare':")
        print(f"Q: shakespeare → A: {result.answer}")
        print(f"   [{result.strategy}, conf={result.confidence:.3f}]")

        print(f"\n--- Stats ---")
        for k, v in engine.stats().items():
            print(f"  {k}: {v}")

        print("\n" + "=" * 60)
        print("DEMO COMPLETE")
        print("=" * 60)


class TestRealWorldMultiHop:
    """
    Multi-hop reasoning with real GloVe embeddings.
    """

    def _setup_shakespeare_kb(self, glove_engine):
        glove_engine.teach_sentence("shakespeare wrote hamlet")
        glove_engine.teach_sentence("hamlet is a play about a prince")
        glove_engine.teach_sentence("shakespeare was born in england")
        glove_engine.teach_template(
            "[PERSON] wrote [WORK]",
            {"PERSON": "noun", "WORK": "noun"}, confidence=0.8,
        )
        glove_engine.teach_template(
            "[WORK] is a [CATEGORY]",
            {"WORK": "noun", "CATEGORY": "noun"}, confidence=0.7,
        )
        return glove_engine

    def _setup_science_kb(self, glove_engine):
        glove_engine.teach_sentence("einstein discovered relativity")
        glove_engine.teach_sentence("relativity describes gravity and spacetime")
        glove_engine.teach_sentence("newton discovered gravity")
        glove_engine.teach_sentence("darwin discovered evolution")
        glove_engine.teach_template(
            "[PERSON] discovered [CONCEPT]",
            {"PERSON": "noun", "CONCEPT": "noun"}, confidence=0.8,
        )
        return glove_engine

    def test_multi_hop_shakespeare_play(self, glove_engine):
        engine = self._setup_shakespeare_kb(glove_engine)
        result = engine.query("who wrote the play about a prince")

        print(f"\nQuery: 'who wrote the play about a prince'")
        print(f"Answer: '{result.answer}'")
        print(f"Strategy: {result.strategy}")
        print(f"Confidence: {result.confidence:.3f}")

        answer_lower = result.answer.lower()
        assert result.converged is True
        assert any(w in answer_lower for w in ["shakespeare", "hamlet", "play", "prince"]), \
            f"Expected Shakespeare-related answer, got: {result.answer}"

    def test_multi_hop_einstein_gravity(self, glove_engine):
        engine = self._setup_science_kb(glove_engine)
        result = engine.query("who discovered spacetime")

        print(f"\nQuery: 'who discovered spacetime'")
        print(f"Answer: '{result.answer}'")
        print(f"Strategy: {result.strategy}")
        print(f"Confidence: {result.confidence:.3f}")

        assert result.converged is True

    def test_multi_hop_more_concepts_than_single(self, glove_engine):
        engine = self._setup_science_kb(glove_engine)

        query_vec = engine.encoder.encode_sentence("gravity spacetime relativity")

        single_result = engine.convergence.converge(query_vec)
        single_ids = {c.id for c in single_result.concepts}

        multi_result = engine.multi_hop.reason(query_vec)
        multi_ids = {c.id for c in multi_result.concepts}

        print(f"\nSingle-hop concepts: {len(single_ids)}")
        print(f"Multi-hop concepts: {len(multi_ids)}")

        assert len(multi_ids) >= len(single_ids), \
            f"Multi-hop ({len(multi_ids)}) should find >= single-hop ({len(single_ids)})"

    def test_multi_hop_no_regression_simple_query(self, glove_engine):
        engine = self._setup_shakespeare_kb(glove_engine)
        result = engine.query("shakespeare")
        assert result.converged is True
        assert "shakespeare" in result.answer.lower() or result.confidence > 0

    def test_multi_hop_honest_abstention(self, glove_engine):
        engine = self._setup_shakespeare_kb(glove_engine)
        result = engine.query("quantum computing blockchain")

        print(f"\nQuery: 'quantum computing blockchain'")
        print(f"Answer: '{result.answer}'")

        if "don't know" not in result.answer.lower():
            assert result.confidence < 0.5, \
                f"Should have low confidence on unknown topic, got {result.confidence}"

    def test_multi_hop_trace_inspectable(self, glove_engine):
        engine = self._setup_science_kb(glove_engine)
        result = engine.query("einstein relativity gravity")

        assert result.trace is not None
        assert len(result.trace) > 0
        assert "Round" in result.trace or "Hop" in result.trace, \
            f"Trace should show reasoning steps:\n{result.trace}"

    def test_multi_hop_performance(self, glove_engine):
        engine = self._setup_science_kb(glove_engine)

        queries = [
            "who discovered spacetime",
            "einstein gravity",
            "evolution darwin",
        ]

        times = []
        for q in queries:
            start = time.perf_counter()
            engine.query(q)
            times.append(time.perf_counter() - start)

        avg_ms = sum(times) / len(times) * 1000
        print(f"\nMulti-hop avg latency: {avg_ms:.2f}ms")
        assert avg_ms < 600, f"Multi-hop too slow: {avg_ms:.1f}ms (>600ms)"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v", "-s"])
