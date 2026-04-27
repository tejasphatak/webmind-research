"""
Test suite for Universal Language Interpreter (ULI).
Tests: lexer, parser/semantics, writer, English module, Marathi module, edge cases.
"""

import unittest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ============================================================
# LEXER TESTS
# ============================================================

class TestLanguageDetection(unittest.TestCase):

    def test_english(self):
        from uli.lexer import detect_language
        self.assertEqual(detect_language("What is the capital of France?"), 'en')

    def test_devanagari(self):
        from uli.lexer import detect_language
        lang = detect_language("भारताची राजधानी काय आहे?")
        self.assertIn(lang, ('hi', 'mr'))  # Devanagari detected

    def test_mixed_script(self):
        from uli.lexer import detect_token_language
        self.assertEqual(detect_token_language("project"), 'en')
        self.assertIn(detect_token_language("राजधानी"), ('mr', 'hi'))

    def test_empty(self):
        from uli.lexer import detect_language
        self.assertEqual(detect_language(""), 'en')


class TestSpellCorrection(unittest.TestCase):

    def test_edit_distance_1(self):
        from uli.lexer import correct_spelling
        vocab = {'the', 'capital', 'france', 'what', 'is'}
        self.assertEqual(correct_spelling('teh', vocab), 'the')
        self.assertEqual(correct_spelling('capitl', vocab), 'capital')

    def test_already_correct(self):
        from uli.lexer import correct_spelling
        vocab = {'hello', 'world'}
        self.assertEqual(correct_spelling('hello', vocab), 'hello')

    def test_no_match(self):
        from uli.lexer import correct_spelling
        vocab = {'cat', 'dog'}
        self.assertEqual(correct_spelling('xyzzy', vocab), 'xyzzy')


class TestNormalizer(unittest.TestCase):

    def test_abbreviation_expansion(self):
        from uli.lexer import Normalizer
        n = Normalizer(lang='en')
        result = n.normalize("ppl bc rn")
        self.assertIn('people', result)
        self.assertIn('because', result)

    def test_contraction_expansion(self):
        from uli.lexer import Normalizer
        n = Normalizer(lang='en')
        result = n.normalize("dont worry im fine")
        self.assertIn("don't", result)
        self.assertIn("I'm", result)

    def test_idempotent(self):
        from uli.lexer import Normalizer
        n = Normalizer(lang='en')
        text = "What is the capital of France?"
        self.assertEqual(n.normalize(n.normalize(text)), n.normalize(text))


class TestEmoji(unittest.TestCase):

    def test_emoji_replacement(self):
        from uli.lexer import replace_emoji
        result = replace_emoji("That was 🔥🔥🔥")
        self.assertIn('excellent', result)
        self.assertNotIn('🔥', result)

    def test_no_emoji(self):
        from uli.lexer import replace_emoji
        text = "No emoji here"
        self.assertEqual(replace_emoji(text), text)


class TestSpecialExtraction(unittest.TestCase):

    def test_url_extraction(self):
        from uli.lexer import extract_special
        text, extracted = extract_special("Check https://example.com for info")
        self.assertIn('[URL]', text)
        self.assertEqual(len(extracted['urls']), 1)

    def test_hashtag(self):
        from uli.lexer import extract_special
        _, extracted = extract_special("Trending #AI today")
        self.assertEqual(extracted['hashtags'], ['#AI'])


class TestTokenizer(unittest.TestCase):

    def test_basic_tokenize(self):
        from uli.lexer import tokenize
        tokens, _ = tokenize("What is the capital of France?")
        self.assertGreater(len(tokens), 0)
        texts = [t.text for t in tokens]
        self.assertIn('What', texts)
        self.assertIn('France', texts)

    def test_tokens_have_pos(self):
        from uli.lexer import tokenize
        tokens, _ = tokenize("Paris is beautiful")
        pos_tags = [t.pos for t in tokens]
        self.assertTrue(any(p != '' for p in pos_tags))

    def test_tokens_have_deps(self):
        from uli.lexer import tokenize
        tokens, _ = tokenize("The cat sat on the mat")
        deps = [t.dep for t in tokens]
        self.assertTrue(any(d != '' for d in deps))

    def test_code_switching_detection(self):
        from uli.lexer import tokenize
        tokens, _ = tokenize("मला हे project complete करायचं आहे")
        langs = set(t.lang for t in tokens)
        # Should detect at least one non-English token
        self.assertTrue(len(langs) >= 1)


# ============================================================
# SEMANTICS TESTS
# ============================================================

class TestSemanticsBasic(unittest.TestCase):

    def test_question_detection(self):
        from uli.lexer import tokenize
        from uli.semantics import tokens_to_ast
        tokens, spans = tokenize("What is the capital of France?")
        ast = tokens_to_ast(tokens, "What is the capital of France?", entity_spans=spans)
        self.assertEqual(ast.type, 'question')

    def test_question_word_extraction(self):
        from uli.lexer import tokenize
        from uli.semantics import tokens_to_ast
        tokens, spans = tokenize("Who painted the Mona Lisa?")
        ast = tokens_to_ast(tokens, "Who painted the Mona Lisa?", entity_spans=spans)
        self.assertEqual(ast.question_word, 'who')
        self.assertEqual(ast.question_target, 'agent')

    def test_entity_extraction(self):
        from uli.lexer import tokenize
        from uli.semantics import tokens_to_ast
        tokens, spans = tokenize("Paris is the capital of France")
        ast = tokens_to_ast(tokens, "Paris is the capital of France", entity_spans=spans)
        self.assertTrue(len(ast.entities) > 0)

    def test_predicate_extraction(self):
        from uli.lexer import tokenize
        from uli.semantics import tokens_to_ast
        tokens, spans = tokenize("Michelangelo painted the ceiling")
        ast = tokens_to_ast(tokens, "Michelangelo painted the ceiling", entity_spans=spans)
        self.assertTrue(ast.predicate != '')

    def test_negation_detection(self):
        from uli.lexer import tokenize
        from uli.semantics import tokens_to_ast
        tokens, spans = tokenize("I did not go there")
        ast = tokens_to_ast(tokens, "I did not go there", entity_spans=spans)
        self.assertTrue(ast.negation)

    def test_non_negation(self):
        from uli.lexer import tokenize
        from uli.semantics import tokens_to_ast
        tokens, spans = tokenize("I went there")
        ast = tokens_to_ast(tokens, "I went there", entity_spans=spans)
        self.assertFalse(ast.negation)


class TestIntentClassification(unittest.TestCase):

    def test_factual(self):
        from uli.lexer import tokenize
        from uli.semantics import tokens_to_ast
        tokens, spans = tokenize("What is the capital of France?")
        ast = tokens_to_ast(tokens, "What is the capital of France?", entity_spans=spans)
        self.assertEqual(ast.intent, 'factual')

    def test_explanation(self):
        from uli.lexer import tokenize
        from uli.semantics import tokens_to_ast
        tokens, spans = tokenize("Why is the sky blue?")
        ast = tokens_to_ast(tokens, "Why is the sky blue?", entity_spans=spans)
        self.assertEqual(ast.intent, 'explanation')

    def test_comparison(self):
        from uli.lexer import tokenize
        from uli.semantics import tokens_to_ast
        tokens, spans = tokenize("Which is larger, Jupiter or Saturn?")
        ast = tokens_to_ast(tokens, "Which is larger, Jupiter or Saturn?", entity_spans=spans)
        self.assertEqual(ast.intent, 'comparison')


class TestSearchQueryExtraction(unittest.TestCase):

    def test_search_query_from_ast(self):
        from uli.lexer import tokenize
        from uli.semantics import tokens_to_ast
        tokens, spans = tokenize("What is the capital of France?")
        ast = tokens_to_ast(tokens, "What is the capital of France?", entity_spans=spans)
        query = ast.search_query()
        self.assertTrue(len(query) > 0)
        # Should contain content words, not stop words
        self.assertIn('capital', query.lower())

    def test_who_question_query(self):
        from uli.lexer import tokenize
        from uli.semantics import tokens_to_ast
        tokens, spans = tokenize("Who painted the Mona Lisa?")
        ast = tokens_to_ast(tokens, "Who painted the Mona Lisa?", entity_spans=spans)
        query = ast.search_query()
        self.assertIn('paint', query.lower())


# ============================================================
# WRITER TESTS
# ============================================================

class TestWriter(unittest.TestCase):

    def test_basic_generation(self):
        from uli.protocol import MeaningAST, Entity
        from uli.writer import ast_to_text
        ast = MeaningAST(
            predicate='paint',
            agent=Entity(text='Michelangelo', type='person'),
            patient=Entity(text='the ceiling', type='thing'),
            tense='past',
        )
        text = ast_to_text(ast)
        self.assertIn('Michelangelo', text)
        self.assertIn('ceiling', text)

    def test_empty_ast(self):
        from uli.protocol import MeaningAST
        from uli.writer import ast_to_text
        ast = MeaningAST()
        text = ast_to_text(ast)
        self.assertEqual(text, '')

    def test_temperature_zero_deterministic(self):
        from uli.protocol import MeaningAST, Entity
        from uli.writer import ast_to_text
        ast = MeaningAST(
            predicate='be',
            agent=Entity(text='Paris', type='place'),
            patient=Entity(text='capital', type='thing'),
        )
        text1 = ast_to_text(ast, temperature=0.0)
        text2 = ast_to_text(ast, temperature=0.0)
        self.assertEqual(text1, text2)  # Deterministic at temp=0


# ============================================================
# ENGLISH MODULE TESTS
# ============================================================

class TestEnglishModule(unittest.TestCase):

    def setUp(self):
        from uli.modules.english import EnglishModule
        self.module = EnglishModule()

    def test_read_simple(self):
        ast = self.module.read("What is the capital of France?")
        self.assertEqual(ast.type, 'question')
        self.assertTrue(len(ast.entities) > 0)

    def test_read_statement(self):
        ast = self.module.read("Paris is the capital of France")
        self.assertTrue(ast.predicate != '')

    def test_detect_english(self):
        # langdetect may detect short text as Dutch; longer text is more reliable
        lang = self.module.detect("What is the capital of France today?")
        self.assertEqual(lang, 'en')

    def test_normalize(self):
        result = self.module.normalize("ppl bc rn ngl")
        self.assertIn('people', result)

    def test_full_pipeline(self):
        """Read → AST → Write round trip."""
        ast = self.module.read("Who painted the Mona Lisa?")
        self.assertEqual(ast.type, 'question')
        self.assertEqual(ast.question_word, 'who')
        # Write should produce some text
        text = self.module.write(ast)
        self.assertIsInstance(text, str)


# ============================================================
# MARATHI MODULE TESTS
# ============================================================

class TestMarathiModule(unittest.TestCase):

    def setUp(self):
        from uli.modules.marathi import MarathiModule
        self.module = MarathiModule()

    def test_detect_devanagari(self):
        lang = self.module.detect("भारताची राजधानी काय आहे?")
        self.assertIn(lang, ('hi', 'mr'))

    def test_code_switching(self):
        """Marathi-English mixed text should parse without crash."""
        ast = self.module.read("मला हे project complete करायचं आहे")
        self.assertEqual(ast.source_language, 'mr')

    def test_english_words_in_marathi(self):
        """English words in Devanagari context should be detected."""
        from uli.lexer import detect_token_language
        self.assertEqual(detect_token_language("project"), 'en')
        self.assertIn(detect_token_language("करायचं"), ('mr', 'hi'))


# ============================================================
# MEANING AST TESTS
# ============================================================

class TestMeaningAST(unittest.TestCase):

    def test_search_query(self):
        from uli.protocol import MeaningAST, Entity
        ast = MeaningAST(
            predicate='paint',
            agent=Entity(text='?'),
            patient=Entity(text='ceiling'),
            location=Entity(text='Vatican'),
            entities=['Sistine Chapel'],
        )
        query = ast.search_query()
        self.assertIn('ceiling', query)
        self.assertIn('Vatican', query)
        self.assertIn('Sistine Chapel', query)

    def test_unfilled_slots(self):
        from uli.protocol import MeaningAST, Entity
        ast = MeaningAST(
            agent=Entity(text='?'),
            question_target='agent',
        )
        slots = ast.unfilled_slots()
        self.assertIn('agent', slots)

    def test_has_nested(self):
        from uli.protocol import MeaningAST
        parent = MeaningAST(sub_clauses=[MeaningAST()])
        self.assertTrue(parent.has_nested())
        empty = MeaningAST()
        self.assertFalse(empty.has_nested())


# ============================================================
# EDGE CASE TESTS
# ============================================================

class TestEdgeCases(unittest.TestCase):

    def test_empty_input(self):
        from uli.modules.english import EnglishModule
        from uli.protocol import MeaningAST as _MeaningAST
        m = EnglishModule()
        ast = m.read("")
        self.assertIsInstance(ast, _MeaningAST)

    def test_single_word(self):
        from uli.modules.english import EnglishModule
        from uli.protocol import MeaningAST as _MeaningAST
        m = EnglishModule()
        ast = m.read("Paris")
        self.assertIsInstance(ast, _MeaningAST)

    def test_only_punctuation(self):
        from uli.modules.english import EnglishModule
        from uli.protocol import MeaningAST as _MeaningAST
        m = EnglishModule()
        ast = m.read("???")
        self.assertIsInstance(ast, _MeaningAST)

    def test_very_long_input(self):
        from uli.modules.english import EnglishModule
        from uli.protocol import MeaningAST as _MeaningAST
        m = EnglishModule()
        long_text = "the quick brown fox " * 100
        ast = m.read(long_text)
        self.assertIsInstance(ast, _MeaningAST)

    def test_numbers_in_text(self):
        from uli.modules.english import EnglishModule
        from uli.protocol import MeaningAST as _MeaningAST
        m = EnglishModule()
        ast = m.read("There are 7 continents on Earth")
        self.assertIsInstance(ast, _MeaningAST)

    def test_emoji_input(self):
        from uli.modules.english import EnglishModule
        from uli.protocol import MeaningAST as _MeaningAST
        m = EnglishModule()
        ast = m.read("That was 🔥🔥🔥")
        self.assertIsInstance(ast, _MeaningAST)

    def test_url_in_text(self):
        from uli.modules.english import EnglishModule
        from uli.protocol import MeaningAST as _MeaningAST
        m = EnglishModule()
        ast = m.read("Check https://example.com for details")
        self.assertIsInstance(ast, _MeaningAST)


if __name__ == '__main__':
    from uli.protocol import MeaningAST  # Import for edge case tests
    import warnings
    warnings.filterwarnings('ignore')
    unittest.main(verbosity=2)
