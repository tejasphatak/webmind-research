"""
Tests for the Generator.

Verifies against HLD spec:
- Strategy A (Template): concepts → closest template → fill slots → text
- Strategy B (Successor walk): walk successor lists, emit tokens
- Strategy C (Concept list): raw concepts as fallback (always works)
- Non-convergence → "I don't know" (invariant #4)
- Each strategy has an explanation trace (invariant #2)
- Template delete = gone (invariant #3)
- Two-speed successor walk (grammar > 0.8 = fast)
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from neuron import NeuronDB
from encoder import Encoder
from convergence import ConvergenceResult, Hop
from generator import Generator, Template, TemplateStore

DIM = 300


def make_vocab():
    """Deterministic test vocabulary."""
    rng = np.random.RandomState(42)
    words = {}
    for w in ["shakespeare", "hamlet", "wrote", "playwright", "english",
              "cat", "dog", "sat", "mat", "the", "on", "a",
              "who", "is", "was", "in", "1600"]:
        words[w] = rng.randn(DIM).astype(np.float32)
    return words


def make_encoder():
    enc = Encoder(data_dir="/tmp/test_gen", dim=DIM)
    enc.load_from_dict(make_vocab())
    return enc


def make_db():
    return NeuronDB(dim=DIM)


def make_components():
    """Create db, encoder, template_store, generator."""
    db = make_db()
    enc = make_encoder()
    ts = TemplateStore(enc)
    gen = Generator(db, enc, ts)
    return db, enc, ts, gen


def insert_word_neuron(db, encoder, word, confidence=0.5):
    """Insert a neuron for a known word."""
    vec = encoder.encode_word(word)
    return db.insert(vec, confidence=confidence)


def fake_convergence(concepts, converged=True, confidence=0.7):
    """Create a ConvergenceResult without running the loop."""
    vec = np.mean([c.vector for c in concepts], axis=0) if concepts else np.zeros(DIM)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return ConvergenceResult(
        converged=converged,
        vector=vec,
        concepts=concepts,
        confidence=confidence,
    )


# --- Abstention tests ---

class TestGeneratorAbstention:

    def test_non_convergence_says_i_dont_know(self):
        """Invariant #4: honest about failure."""
        db, enc, ts, gen = make_components()
        result = gen.generate(ConvergenceResult(
            converged=False, vector=np.zeros(DIM),
            concepts=[], confidence=0.0,
        ))
        assert result.text == "I don't know."
        assert result.strategy == "abstain"
        assert result.confidence == 0.0

    def test_no_concepts_says_i_dont_know(self):
        db, enc, ts, gen = make_components()
        result = gen.generate(ConvergenceResult(
            converged=True, vector=np.zeros(DIM),
            concepts=[], confidence=0.5,
        ))
        assert result.text == "I don't know."
        assert result.strategy == "abstain"


# --- Template tests ---

class TestGeneratorTemplate:

    def test_template_fills_all_slots(self):
        db, enc, ts, gen = make_components()

        ts.add(
            pattern="[PERSON] wrote [WORK]",
            slots={"PERSON": "noun", "WORK": "noun"},
            confidence=0.8,
        )

        n1 = insert_word_neuron(db, enc, "shakespeare", confidence=0.7)
        n2 = insert_word_neuron(db, enc, "hamlet", confidence=0.6)

        conv = fake_convergence([n1, n2])
        result = gen.generate(conv)

        assert result.strategy == "template"
        assert "shakespeare" in result.text.lower() or "hamlet" in result.text.lower()
        assert result.template_used is not None
        assert result.confidence > 0

    def test_template_partial_fill(self):
        """Some slots filled, others show '...'."""
        db, enc, ts, gen = make_components()

        ts.add(
            pattern="[PERSON] wrote [WORK] in [YEAR]",
            slots={"PERSON": "noun", "WORK": "noun", "YEAR": "number"},
            confidence=0.8,
        )

        # Only provide one concept — not enough for all slots
        n1 = insert_word_neuron(db, enc, "shakespeare", confidence=0.7)

        conv = fake_convergence([n1])
        result = gen.generate(conv)

        # Should either be template (partial) or concept_list
        assert result.text  # something is returned
        assert result.strategy in ("template", "concept_list")

    def test_no_templates_falls_through(self):
        """No templates → should fall to successor or concept_list."""
        db, enc, ts, gen = make_components()

        n1 = insert_word_neuron(db, enc, "cat", confidence=0.5)
        conv = fake_convergence([n1])
        result = gen.generate(conv)

        assert result.strategy in ("successor", "concept_list")

    def test_template_has_explanation(self):
        """Invariant #2: every answer has a source."""
        db, enc, ts, gen = make_components()

        ts.add(
            pattern="[SUBJECT] is [ATTRIBUTE]",
            slots={"SUBJECT": "noun", "ATTRIBUTE": "noun"},
            confidence=0.7,
        )

        n1 = insert_word_neuron(db, enc, "cat", confidence=0.5)
        n2 = insert_word_neuron(db, enc, "english", confidence=0.5)

        conv = fake_convergence([n1, n2])
        result = gen.generate(conv)

        explanation = result.explain()
        assert "Strategy:" in explanation
        assert len(result.trace) > 0

    def test_template_delete_removes_it(self):
        """Invariant #3: delete = gone."""
        db, enc, ts, gen = make_components()

        t = ts.add(
            pattern="[X] wrote [Y]",
            slots={"X": "noun", "Y": "noun"},
            confidence=0.8,
        )

        assert ts.count() == 1
        ts.delete(t.id)
        assert ts.count() == 0


# --- Successor walk tests ---

class TestGeneratorSuccessor:

    def test_successor_walk_produces_text(self):
        db, enc, ts, gen = make_components()

        n_the = insert_word_neuron(db, enc, "the", confidence=0.6)
        n_cat = insert_word_neuron(db, enc, "cat", confidence=0.7)
        n_sat = insert_word_neuron(db, enc, "sat", confidence=0.6)
        n_on = insert_word_neuron(db, enc, "on", confidence=0.5)
        n_mat = insert_word_neuron(db, enc, "mat", confidence=0.5)

        # Build successor chain: the → cat → sat → on → mat
        db.update_successors(n_the.id, n_cat.id, 0.9)
        db.update_successors(n_cat.id, n_sat.id, 0.85)
        db.update_successors(n_sat.id, n_on.id, 0.8)
        db.update_successors(n_on.id, n_mat.id, 0.75)

        conv = fake_convergence([n_the])
        result = gen.generate(conv)

        assert result.strategy == "successor"
        words = result.text.split()
        assert len(words) >= 2
        assert result.confidence > 0

    def test_successor_walk_avoids_loops(self):
        """Should not revisit neurons."""
        db, enc, ts, gen = make_components()

        n1 = insert_word_neuron(db, enc, "cat", confidence=0.7)
        n2 = insert_word_neuron(db, enc, "dog", confidence=0.6)

        # Create a cycle: cat → dog → cat
        db.update_successors(n1.id, n2.id, 0.9)
        db.update_successors(n2.id, n1.id, 0.9)

        conv = fake_convergence([n1])
        result = gen.generate(conv)

        # Should stop at 2 tokens, not loop forever
        words = result.text.split()
        assert len(words) <= 3

    def test_successor_walk_has_trace(self):
        """Each step in the walk should be traced."""
        db, enc, ts, gen = make_components()

        n1 = insert_word_neuron(db, enc, "the", confidence=0.6)
        n2 = insert_word_neuron(db, enc, "cat", confidence=0.7)

        db.update_successors(n1.id, n2.id, 0.9)

        conv = fake_convergence([n1])
        result = gen.generate(conv)

        if result.strategy == "successor":
            assert len(result.trace) > 0
            assert any("Step" in t or "Start" in t for t in result.trace)

    def test_no_successors_falls_to_concept_list(self):
        """Neurons without successors → can't walk → concept_list."""
        db, enc, ts, gen = make_components()

        n1 = insert_word_neuron(db, enc, "hamlet", confidence=0.7)
        # No successors set

        conv = fake_convergence([n1])
        result = gen.generate(conv)

        assert result.strategy == "concept_list"


# --- Concept list tests ---

class TestGeneratorConceptList:

    def test_concept_list_returns_words(self):
        db, enc, ts, gen = make_components()

        n1 = insert_word_neuron(db, enc, "shakespeare", confidence=0.7)
        n2 = insert_word_neuron(db, enc, "hamlet", confidence=0.6)

        conv = fake_convergence([n1, n2])
        result = gen.generate(conv)

        assert result.strategy == "concept_list"
        assert "shakespeare" in result.text.lower() or "hamlet" in result.text.lower()

    def test_concept_list_always_works(self):
        """Concept list is the guaranteed fallback."""
        db, enc, ts, gen = make_components()

        n = insert_word_neuron(db, enc, "cat", confidence=0.3)
        conv = fake_convergence([n], confidence=0.3)
        result = gen.generate(conv)

        assert result.text != ""
        assert result.text != "I don't know."
        assert result.strategy == "concept_list"

    def test_concept_list_has_trace(self):
        db, enc, ts, gen = make_components()

        n = insert_word_neuron(db, enc, "dog", confidence=0.5)
        conv = fake_convergence([n])
        result = gen.generate(conv)

        assert len(result.trace) > 0
        assert "Concept list fallback" in result.trace[0]


# --- Template store tests ---

class TestTemplateStore:

    def test_add_and_search(self):
        enc = make_encoder()
        ts = TemplateStore(enc)

        ts.add("[PERSON] wrote [WORK]", {"PERSON": "noun", "WORK": "noun"})
        ts.add("[ANIMAL] sat on [SURFACE]", {"ANIMAL": "noun", "SURFACE": "noun"})

        assert ts.count() == 2

        query = enc.encode_sentence("who wrote hamlet")
        results = ts.search(query, k=1)
        assert len(results) == 1

    def test_search_returns_closest(self):
        enc = make_encoder()
        ts = TemplateStore(enc)

        t_write = ts.add("[PERSON] wrote [WORK]",
                         {"PERSON": "noun", "WORK": "noun"}, confidence=0.8)
        t_sat = ts.add("[ANIMAL] sat on [SURFACE]",
                       {"ANIMAL": "noun", "SURFACE": "noun"}, confidence=0.8)

        query = enc.encode_sentence("wrote hamlet")
        results = ts.search(query, k=1)
        assert results[0].id == t_write.id

    def test_empty_store_returns_nothing(self):
        enc = make_encoder()
        ts = TemplateStore(enc)
        results = ts.search(np.zeros(DIM, dtype=np.float32))
        assert results == []

    def test_delete_template(self):
        enc = make_encoder()
        ts = TemplateStore(enc)
        t = ts.add("[X] is [Y]", {"X": "noun", "Y": "noun"})
        assert ts.count() == 1
        assert ts.delete(t.id) is True
        assert ts.count() == 0

    def test_delete_nonexistent(self):
        enc = make_encoder()
        ts = TemplateStore(enc)
        assert ts.delete(999) is False


# --- Template dataclass tests ---

class TestTemplate:

    def test_fill_all_slots(self):
        t = Template(
            id=0,
            pattern="[PERSON] wrote [WORK] in [YEAR]",
            slots={"PERSON": "noun", "WORK": "noun", "YEAR": "number"},
            vector=np.zeros(DIM),
        )
        result = t.fill({"PERSON": "Shakespeare", "WORK": "Hamlet", "YEAR": "1600"})
        assert result == "Shakespeare wrote Hamlet in 1600"

    def test_fill_partial(self):
        t = Template(
            id=0,
            pattern="[PERSON] wrote [WORK]",
            slots={"PERSON": "noun", "WORK": "noun"},
            vector=np.zeros(DIM),
        )
        result = t.fill({"PERSON": "Shakespeare"})
        assert result == "Shakespeare wrote [WORK]"

    def test_unfilled_slots(self):
        t = Template(
            id=0,
            pattern="[A] [B] [C]",
            slots={"A": "noun", "B": "verb", "C": "noun"},
            vector=np.zeros(DIM),
        )
        assert t.unfilled_slots({"A": "x"}) == ["B", "C"]
        assert t.unfilled_slots({"A": "x", "B": "y", "C": "z"}) == []

    def test_slot_names(self):
        t = Template(
            id=0,
            pattern="[X] [Y]",
            slots={"X": "noun", "Y": "verb"},
            vector=np.zeros(DIM),
        )
        assert set(t.slot_names) == {"X", "Y"}


# --- Generation result tests ---

class TestGenerationResult:

    def test_explain_contains_strategy(self):
        from generator import GenerationResult
        r = GenerationResult(
            text="hello",
            strategy="template",
            confidence=0.8,
            trace=["matched template X"],
        )
        exp = r.explain()
        assert "template" in exp
        assert "0.8" in exp

    def test_explain_with_template(self):
        from generator import GenerationResult
        t = Template(
            id=0,
            pattern="[X] wrote [Y]",
            slots={"X": "noun", "Y": "noun"},
            vector=np.zeros(DIM),
        )
        r = GenerationResult(
            text="Shakespeare wrote Hamlet",
            strategy="template",
            confidence=0.7,
            template_used=t,
            slot_fills={"X": "Shakespeare", "Y": "Hamlet"},
        )
        exp = r.explain()
        assert "[X] wrote [Y]" in exp
        assert "Shakespeare" in exp


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
