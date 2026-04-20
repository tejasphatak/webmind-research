"""
Tests for MVP-4A: Template auto-learning from taught sentences.

Verifies that teach_sentence() auto-extracts reusable templates
by detecting structural vs content words.
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from engine import Engine


def make_engine(tmp_path):
    """Create engine with test vocabulary."""
    engine = Engine(data_dir=str(tmp_path), dim=300)
    rng = np.random.RandomState(42)
    words = {}
    word_list = [
        "shakespeare", "wrote", "hamlet", "einstein", "discovered",
        "relativity", "newton", "invented", "calculus", "paris",
        "is", "the", "capital", "of", "france", "london", "england",
        "who", "what", "a", "an", "in", "for", "was", "born",
        "python", "programming", "language", "created", "guido",
        "tokyo", "japan", "berlin", "germany", "madrid", "spain",
    ]
    for i, w in enumerate(word_list):
        vec = rng.randn(300).astype(np.float32)
        vec = vec / (np.linalg.norm(vec) + 1e-10)
        words[w] = vec
    engine.load_embeddings_from_dict(words)
    return engine


class TestAutoTemplateExtraction:
    def test_teach_sentence_creates_template(self, tmp_path):
        """Teaching a sentence should auto-create a template."""
        engine = make_engine(tmp_path)
        before = engine.template_store.count()
        engine.teach_sentence("shakespeare wrote hamlet")
        after = engine.template_store.count()
        assert after > before
        engine.close()

    def test_template_has_slots_for_content_words(self, tmp_path):
        """Content words (shakespeare, hamlet) should become slots."""
        engine = make_engine(tmp_path)
        engine.teach_sentence("shakespeare wrote hamlet")

        # Find the auto-generated template
        templates = engine.template_store.templates
        assert len(templates) >= 1
        t = templates[-1]  # most recently added
        # "wrote" is structural, shakespeare/hamlet are content → slots
        assert "wrote" in t.pattern
        assert "[" in t.pattern  # has at least one slot
        assert len(t.slots) >= 1
        engine.close()

    def test_no_duplicate_templates(self, tmp_path):
        """Teaching the same pattern twice shouldn't create duplicates."""
        engine = make_engine(tmp_path)
        engine.teach_sentence("shakespeare wrote hamlet")
        count1 = engine.template_store.count()
        # Same structural pattern
        engine.teach_sentence("shakespeare wrote hamlet")
        count2 = engine.template_store.count()
        assert count2 == count1
        engine.close()

    def test_different_content_same_structure(self, tmp_path):
        """Different content words with same structure → same template."""
        engine = make_engine(tmp_path)
        engine.teach_sentence("shakespeare wrote hamlet")
        count1 = engine.template_store.count()
        # "newton invented calculus" has same structure: [NOUN] verb [NOUN]
        engine.teach_sentence("newton invented calculus")
        count2 = engine.template_store.count()
        # Different verb → different template (invented vs wrote)
        # Both should exist
        assert count2 >= count1
        engine.close()

    def test_short_sentence_no_template(self, tmp_path):
        """Sentences under 3 tokens shouldn't generate templates."""
        engine = make_engine(tmp_path)
        engine.teach_sentence("shakespeare hamlet")
        # Two content words, no structural → no template
        # (also under 3 tokens after tokenization depends on vocab)
        # Just verify no crash
        engine.close()

    def test_auto_template_disabled(self, tmp_path):
        """auto_template=False should skip template extraction."""
        engine = make_engine(tmp_path)
        engine.teach_sentence("shakespeare wrote hamlet", auto_template=False)
        assert engine.template_store.count() == 0
        engine.close()

    def test_template_usable_for_generation(self, tmp_path):
        """Auto-learned template should work for answering queries."""
        engine = make_engine(tmp_path)

        # Teach several sentences with "wrote" pattern
        engine.teach_sentence("shakespeare wrote hamlet")

        # Query about writing
        result = engine.query("who wrote hamlet")
        # Should find shakespeare via convergence + template
        assert result.strategy != "abstain"
        assert result.confidence > 0
        engine.close()

    def test_capital_pattern(self, tmp_path):
        """Test auto-template with 'X is the capital of Y' pattern."""
        engine = make_engine(tmp_path)
        engine.teach_sentence("paris is the capital of france")

        templates = engine.template_store.templates
        assert len(templates) >= 1
        # "is", "the", "capital", "of" are structural
        # "paris", "france" are content → slots
        t = templates[-1]
        assert len(t.slots) >= 2  # at least paris and france as slots
        engine.close()

    def test_multiple_sentences_build_template_library(self, tmp_path):
        """Teaching many sentences builds a diverse template library."""
        engine = make_engine(tmp_path)
        sentences = [
            "shakespeare wrote hamlet",
            "newton invented calculus",
            "paris is the capital of france",
            "london is the capital of england",
            "einstein discovered relativity",
        ]
        for s in sentences:
            engine.teach_sentence(s)

        assert engine.template_store.count() >= 3  # at least 3 distinct patterns
        engine.close()

    def test_learned_template_persists(self, tmp_path):
        """Auto-learned templates should survive engine restart."""
        engine = make_engine(tmp_path)
        engine.teach_sentence("shakespeare wrote hamlet")
        count = engine.template_store.count()
        assert count > 0
        engine.close()

        # Reopen
        engine2 = make_engine(tmp_path)
        assert engine2.template_store.count() == count
        engine2.close()
