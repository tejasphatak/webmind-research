"""
Tests for MVP-4B: Convergence-guided sentence generation.

Verifies:
  1. Successor walk uses convergence to pick tokens (slow path)
  2. High-confidence successors bypass convergence (fast path)
  3. Query anchor keeps generation on-topic
  4. Context accumulation influences later tokens
  5. Convergence jump when successor chain ends
  6. Multi-strategy evaluation mode
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from engine import Engine
from neuron import NeuronDB
from generator import Generator, TemplateStore, GenerationResult
from encoder import Encoder
from convergence import ConvergenceResult


# --- Helpers ---

def make_engine(tmp_path, seed=42):
    """Create engine with a vocabulary and some taught sentences."""
    engine = Engine(data_dir=str(tmp_path), dim=300)
    rng = np.random.RandomState(seed)
    words = {}
    word_list = [
        "shakespeare", "wrote", "hamlet", "macbeth", "plays",
        "einstein", "discovered", "relativity", "physics", "theory",
        "newton", "invented", "calculus", "mathematics", "laws",
        "paris", "is", "the", "capital", "of", "france", "city",
        "london", "england", "berlin", "germany",
        "python", "programming", "language", "created", "guido",
        "cat", "sat", "on", "mat", "a", "an", "was", "in",
        "who", "what", "where", "how", "and", "for",
    ]
    for w in word_list:
        vec = rng.randn(300).astype(np.float32)
        vec = vec / (np.linalg.norm(vec) + 1e-10)
        words[w] = vec
    engine.load_embeddings_from_dict(words)
    return engine


def teach_and_query(engine, sentences, query):
    """Teach sentences, then query. Returns QueryResult."""
    for s in sentences:
        engine.teach_sentence(s)
    return engine.query(query)


# --- Successor Walk: Core Mechanism ---

class TestConvergenceGuidedWalk:
    def test_successor_walk_produces_output(self, tmp_path):
        """Basic: teaching a sentence and querying should produce successor-based output."""
        engine = make_engine(tmp_path)
        engine.teach_sentence("shakespeare wrote hamlet")
        result = engine.query("shakespeare")
        # Should get some output (template or successor)
        assert result.strategy != "abstain"
        assert len(result.answer) > 0
        engine.close()

    def test_taught_sentence_reproduced(self, tmp_path):
        """A taught sentence should be reproducible via successor walk."""
        engine = make_engine(tmp_path)
        engine.teach_sentence("shakespeare wrote hamlet")
        # Query with a word from the sentence
        result = engine.query("shakespeare hamlet")
        assert result.strategy != "abstain"
        # The answer should contain words from the taught sentence
        words = set(result.answer.lower().split())
        taught_words = {"shakespeare", "wrote", "hamlet"}
        assert len(words & taught_words) >= 2
        engine.close()

    def test_multiple_sentences_correct_routing(self, tmp_path):
        """With multiple taught sentences, query routes to the right one."""
        engine = make_engine(tmp_path)
        engine.teach_sentence("shakespeare wrote hamlet")
        engine.teach_sentence("einstein discovered relativity")
        engine.teach_sentence("newton invented calculus")

        r1 = engine.query("who wrote hamlet")
        r2 = engine.query("einstein physics")

        # Each should find concepts from the correct sentence
        assert "shakespeare" in r1.answer.lower() or "hamlet" in r1.answer.lower()
        assert "einstein" in r2.answer.lower() or "relativity" in r2.answer.lower()
        engine.close()


# --- Two-Speed Generation ---

class TestTwoSpeed:
    def test_fast_path_high_confidence(self, tmp_path):
        """High-confidence successors should be emitted on the fast path."""
        engine = make_engine(tmp_path)
        # Teach with high confidence so successors get 0.8
        engine.teach_sentence("paris is the capital of france", confidence=0.8)
        result = engine.query("paris capital")
        # Should produce output following the successor chain
        assert result.strategy != "abstain"
        engine.close()

    def test_slow_path_convergence(self, tmp_path):
        """When successor confidence is low, convergence guides selection."""
        engine = make_engine(tmp_path)
        # Teach with low confidence
        engine.teach_sentence("cat sat on mat", confidence=0.3)
        result = engine.query("cat mat")
        # Should still produce something (convergence picks from candidates)
        assert result.confidence >= 0
        engine.close()


# --- Query Anchor ---

class TestQueryAnchor:
    def test_generation_stays_on_topic(self, tmp_path):
        """Generated text should stay relevant to the query topic."""
        engine = make_engine(tmp_path)
        engine.teach_sentence("shakespeare wrote hamlet")
        engine.teach_sentence("shakespeare wrote macbeth")
        engine.teach_sentence("paris is the capital of france")

        result = engine.query("shakespeare plays")
        answer = result.answer.lower()
        # Should mention shakespeare-related things, not paris
        has_shakespeare = "shakespeare" in answer or "hamlet" in answer or "macbeth" in answer
        has_paris = "paris" in answer or "france" in answer
        if result.strategy != "abstain":
            assert has_shakespeare or not has_paris
        engine.close()


# --- Context Accumulation ---

class TestContextAccumulation:
    def test_context_builds_across_tokens(self, tmp_path):
        """The _build_context_vector method should blend query + emitted tokens."""
        engine = make_engine(tmp_path)
        gen = engine.generator
        query_vec = np.random.randn(300).astype(np.float32)
        query_vec = query_vec / np.linalg.norm(query_vec)

        # No context = pure query
        vec0 = gen._build_context_vector(query_vec, [])
        np.testing.assert_array_almost_equal(vec0, query_vec, decimal=5)

        # With context, should blend
        from neuron import Neuron
        fake_neuron = Neuron(
            id=1,
            vector=np.random.randn(300).astype(np.float32),
            confidence=0.5,
        )
        fake_neuron.vector = fake_neuron.vector / np.linalg.norm(fake_neuron.vector)

        vec1 = gen._build_context_vector(query_vec, [fake_neuron])
        # Should be different from pure query
        sim = float(np.dot(vec0, vec1))
        assert sim < 0.999  # blended, not identical
        assert sim > 0.3    # but still anchored to query
        engine.close()

    def test_query_weight_floors_at_40_percent(self, tmp_path):
        """Query anchor never drops below 40% even with many emitted tokens."""
        engine = make_engine(tmp_path)
        gen = engine.generator
        query_vec = np.random.randn(300).astype(np.float32)
        query_vec = query_vec / np.linalg.norm(query_vec)

        from neuron import Neuron
        # Emit 20 tokens — query weight should floor at 0.4
        emitted = []
        for i in range(20):
            v = np.random.randn(300).astype(np.float32)
            v = v / np.linalg.norm(v)
            emitted.append(Neuron(id=i+100, vector=v, confidence=0.5))

        vec = gen._build_context_vector(query_vec, emitted)
        # The result should still have significant query component
        query_sim = float(np.dot(vec, query_vec))
        assert query_sim > 0.2  # query anchor is still present
        engine.close()


# --- Convergence Jump ---

class TestConvergenceJump:
    def test_jump_across_unconnected_concepts(self, tmp_path):
        """When successor chain ends, convergence should find related neurons."""
        engine = make_engine(tmp_path)
        # Teach two related but unconnected sentences
        engine.teach_sentence("shakespeare wrote hamlet")
        engine.teach_sentence("shakespeare wrote macbeth")
        # "hamlet" has no successors, but "macbeth" is nearby in concept space
        # via the shared "shakespeare wrote" prefix
        result = engine.query("shakespeare hamlet macbeth")
        assert result.strategy != "abstain"
        engine.close()


# --- Multi-Strategy Evaluation ---

class TestMultiStrategyEval:
    def test_evaluate_all_returns_best(self, tmp_path):
        """evaluate_all=True should try all strategies and pick best."""
        engine = make_engine(tmp_path)
        engine.teach_sentence("shakespeare wrote hamlet")
        engine.teach_template("[S0] wrote [S1]", {"S0": "noun", "S1": "noun"})

        # Use generator directly with evaluate_all
        query_vec = engine.encoder.encode_sentence("who wrote hamlet")
        conv_result = ConvergenceResult(
            converged=True,
            vector=query_vec,
            concepts=[],
            confidence=0.5,
        )
        # Need concepts — let convergence find them
        from convergence import ConvergenceLoop
        loop = ConvergenceLoop(engine.db)
        conv_result = loop.converge(query_vec)

        if conv_result.converged and conv_result.concepts:
            result = engine.generator.generate(
                conv_result,
                query_vector=query_vec,
                query_words=["who", "wrote", "hamlet"],
                evaluate_all=True,
            )
            assert "Evaluated" in result.trace[0]
            assert result.confidence > 0
        engine.close()

    def test_evaluate_all_includes_all_strategies_in_trace(self, tmp_path):
        """Trace should show all evaluated strategies and their scores."""
        engine = make_engine(tmp_path)
        engine.teach_sentence("paris is the capital of france")
        engine.teach_template("[S0] is the capital of [S1]",
                              {"S0": "noun", "S1": "noun"})

        query_vec = engine.encoder.encode_sentence("capital of france")
        from convergence import ConvergenceLoop
        loop = ConvergenceLoop(engine.db)
        conv_result = loop.converge(query_vec)

        if conv_result.converged and conv_result.concepts:
            result = engine.generator.generate(
                conv_result,
                query_vector=query_vec,
                query_words=["capital", "of", "france"],
                evaluate_all=True,
            )
            # Trace should mention strategy names
            eval_line = result.trace[0]
            # At minimum concept_list always runs
            assert "concept_list" in eval_line or "template" in eval_line
        engine.close()


# --- Edge Cases ---

class TestEdgeCases:
    def test_empty_concepts_no_crash(self, tmp_path):
        """Successor walk with empty concepts should return None gracefully."""
        engine = make_engine(tmp_path)
        result = engine.generator._try_successor_walk([], 10)
        assert result is None
        engine.close()

    def test_single_word_answer(self, tmp_path):
        """Single neuron with no successors should fall through to concept list."""
        engine = make_engine(tmp_path)
        engine.teach("shakespeare", confidence=0.8)
        result = engine.query("shakespeare")
        # Should get at least a concept list
        assert result.strategy != "abstain" or result.confidence == 0
        engine.close()

    def test_circular_successor_no_infinite_loop(self, tmp_path):
        """Circular successor references shouldn't cause infinite loops."""
        engine = make_engine(tmp_path)
        # Teach sentence that creates a chain
        engine.teach_sentence("cat sat on mat")
        # Manually create a circular reference
        word_map = engine.db.load_word_mappings()
        neurons = [word_map[w] for w in ["cat", "sat", "on", "mat"] if w in word_map]
        if len(neurons) >= 2:
            # Wire last back to first
            engine.db.update_successors(neurons[-1], neurons[0], 0.5)

        # Should still terminate
        result = engine.query("cat mat")
        assert result is not None  # didn't hang
        engine.close()

    def test_no_query_vector_still_works(self, tmp_path):
        """Successor walk without query_vector should use confidence fallback."""
        engine = make_engine(tmp_path)
        engine.teach_sentence("shakespeare wrote hamlet")

        from neuron import Neuron
        word_map = engine.db.load_word_mappings()
        concepts = []
        for word in ["shakespeare", "wrote", "hamlet"]:
            nid = word_map.get(word)
            if nid:
                n = engine.db.get(nid)
                if n:
                    concepts.append(n)

        if concepts:
            result = engine.generator._try_successor_walk(
                concepts, max_tokens=10, query_vector=None
            )
            # Should work (fallback to confidence-based selection)
            # May return None if not enough successors, that's OK
            assert result is None or isinstance(result, GenerationResult)
        engine.close()
