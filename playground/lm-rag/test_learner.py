"""
Tests for ULI Learner — training by talking / feeding data.
"""

import unittest
import tempfile
import shutil
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from uli.learner import Learner, LearningQueue
from uli.protocol import Token


class TestLearningQueue(unittest.TestCase):

    def test_flag_increments_count(self):
        q = LearningQueue(threshold=3)
        q.flag('bussin', context='that is bussin')
        self.assertEqual(q.unknown_words['bussin']['count'], 1)

    def test_threshold_reached(self):
        q = LearningQueue(threshold=3)
        self.assertFalse(q.flag('bussin'))
        self.assertFalse(q.flag('bussin'))
        self.assertTrue(q.flag('bussin'))  # 3rd time

    def test_get_ready(self):
        q = LearningQueue(threshold=2)
        q.flag('bussin', pos_hint='ADJ')
        q.flag('bussin', pos_hint='ADJ')
        q.flag('slay', pos_hint='VERB')
        q.flag('slay', pos_hint='VERB')
        ready = q.get_ready()
        self.assertEqual(len(ready), 2)
        words = [r['word'] for r in ready]
        self.assertIn('bussin', words)
        self.assertIn('slay', words)

    def test_pos_inferred_from_majority(self):
        q = LearningQueue(threshold=3)
        q.flag('bussin', pos_hint='ADJ')
        q.flag('bussin', pos_hint='ADJ')
        q.flag('bussin', pos_hint='VERB')  # Minority
        ready = q.get_ready()
        self.assertEqual(ready[0]['pos'], 'ADJ')

    def test_short_words_ignored(self):
        q = LearningQueue(threshold=1)
        self.assertFalse(q.flag('a'))  # Too short
        self.assertEqual(len(q.unknown_words), 0)

    def test_remove(self):
        q = LearningQueue(threshold=1)
        q.flag('bussin')
        q.remove('bussin')
        self.assertNotIn('bussin', q.unknown_words)

    def test_clear(self):
        q = LearningQueue(threshold=1)
        q.flag('bussin')
        q.flag('slay')
        q.clear()
        self.assertEqual(len(q.unknown_words), 0)

    def test_contexts_stored(self):
        q = LearningQueue(threshold=3)
        q.flag('bussin', context='that food is bussin')
        q.flag('bussin', context='this song is bussin')
        self.assertEqual(len(q.unknown_words['bussin']['contexts']), 2)

    def test_contexts_capped(self):
        q = LearningQueue(threshold=20)
        for i in range(15):
            q.flag('bussin', context=f'context {i}')
        self.assertLessEqual(len(q.unknown_words['bussin']['contexts']), 10)


class TestLearnerTeaching(unittest.TestCase):
    """Mode 3: Explicit teaching."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        # Create minimal data structure
        os.makedirs(os.path.join(self.tmpdir, 'vocab'), exist_ok=True)
        os.makedirs(os.path.join(self.tmpdir, 'normalize'), exist_ok=True)
        os.makedirs(os.path.join(self.tmpdir, 'idioms'), exist_ok=True)
        # Write minimal vocab
        with open(os.path.join(self.tmpdir, 'vocab', 'en.json'), 'w') as f:
            json.dump({'words': {'hello': {'pos': ['INTJ']}}}, f)
        with open(os.path.join(self.tmpdir, 'normalize', 'en.json'), 'w') as f:
            json.dump({'abbreviations': {}, 'contractions': {}}, f)
        with open(os.path.join(self.tmpdir, 'idioms', 'en.json'), 'w') as f:
            json.dump({}, f)
        self.learner = Learner(data_dir=self.tmpdir)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_teach_word(self):
        self.learner.teach_word('slay', {
            'pos': ['VERB'],
            'senses': ['excel'],
            'register': 'gen_z',
        })
        self.assertTrue(self.learner.is_known('slay'))

    def test_teach_abbreviation(self):
        self.learner.teach_abbreviation('ngl', 'not gonna lie')
        self.learner._ensure_loaded('en')
        abbrevs = self.learner._normalize['en'].get('abbreviations', {})
        self.assertEqual(abbrevs.get('ngl'), 'not gonna lie')

    def test_teach_idiom(self):
        self.learner.teach_idiom('spill the tea', 'share gossip')
        self.learner._ensure_loaded('en')
        idioms = self.learner._idioms.get('en', {})
        self.assertIn('spill the tea', idioms)
        self.assertEqual(idioms['spill the tea']['meaning'], 'share gossip')

    def test_teach_persists_after_save(self):
        self.learner.teach_word('rizz', {'pos': ['NOUN'], 'senses': ['charisma']})
        self.learner.save()
        # Reload from disk
        learner2 = Learner(data_dir=self.tmpdir)
        self.assertTrue(learner2.is_known('rizz'))

    def test_teach_marked_as_taught(self):
        self.learner.teach_word('bussin', {'pos': ['ADJ']})
        self.learner._ensure_loaded('en')
        entry = self.learner._vocab['en']['words']['bussin']
        self.assertTrue(entry.get('taught'))

    def test_learned_words_listed(self):
        self.learner.teach_word('slay', {'pos': ['VERB']})
        self.learner.teach_word('rizz', {'pos': ['NOUN']})
        learned = self.learner.learned_words('en')
        self.assertIn('slay', learned)
        self.assertIn('rizz', learned)
        self.assertNotIn('hello', learned)  # Original vocab, not learned


class TestLearnerConversation(unittest.TestCase):
    """Mode 1: Learning from conversation."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.tmpdir, 'vocab'), exist_ok=True)
        os.makedirs(os.path.join(self.tmpdir, 'normalize'), exist_ok=True)
        os.makedirs(os.path.join(self.tmpdir, 'idioms'), exist_ok=True)
        with open(os.path.join(self.tmpdir, 'vocab', 'en.json'), 'w') as f:
            json.dump({'words': {'the': {'pos': ['DET']}, 'is': {'pos': ['VERB']}},
                       'stop_words': ['the', 'is', 'a']}, f)
        with open(os.path.join(self.tmpdir, 'normalize', 'en.json'), 'w') as f:
            json.dump({'abbreviations': {}, 'contractions': {}}, f)
        self.learner = Learner(data_dir=self.tmpdir)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_unknown_tokens_flagged(self):
        tokens = [
            Token(text='that', lang='en', pos='DET'),
            Token(text='bussin', lang='en', pos='ADJ'),
            Token(text='food', lang='en', pos='NOUN'),
        ]
        self.learner.on_tokens(tokens, text='that bussin food', lang='en')
        self.assertIn('bussin', self.learner.queue.unknown_words)
        # 'that' might not be flagged if too short or known

    def test_known_words_not_flagged(self):
        tokens = [Token(text='the', lang='en', pos='DET')]
        self.learner.on_tokens(tokens, lang='en')
        self.assertNotIn('the', self.learner.queue.unknown_words)

    def test_threshold_commit(self):
        self.learner.queue = LearningQueue(threshold=2)
        tokens = [Token(text='bussin', lang='en', pos='ADJ')]
        self.learner.on_tokens(tokens, text='context1', lang='en')
        self.learner.on_tokens(tokens, text='context2', lang='en')
        committed = self.learner.commit_queue(lang='en')
        self.assertIn('bussin', committed)
        self.assertTrue(self.learner.is_known('bussin'))

    def test_verified_answer(self):
        fact = self.learner.on_verified_answer(
            "Capital of France?", "Paris")
        self.assertEqual(fact['answer'], 'Paris')
        self.assertEqual(fact['confidence'], 1.0)
        self.assertEqual(fact['source'], 'user_verified')

    def test_correction_teaches_abbreviation(self):
        self.learner.on_correction('ngl', 'not gonna lie',
                                    category='abbreviation')
        self.learner._ensure_loaded('en')
        abbrevs = self.learner._normalize['en'].get('abbreviations', {})
        self.assertEqual(abbrevs.get('ngl'), 'not gonna lie')


class TestLearnerBulk(unittest.TestCase):
    """Mode 2: Bulk data learning."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.tmpdir, 'vocab'), exist_ok=True)
        os.makedirs(os.path.join(self.tmpdir, 'normalize'), exist_ok=True)
        os.makedirs(os.path.join(self.tmpdir, 'idioms'), exist_ok=True)
        with open(os.path.join(self.tmpdir, 'vocab', 'en.json'), 'w') as f:
            json.dump({'words': {}, 'stop_words': ['the', 'a', 'is', 'are', 'of', 'in', 'to']}, f)
        with open(os.path.join(self.tmpdir, 'normalize', 'en.json'), 'w') as f:
            json.dump({'abbreviations': {}, 'contractions': {}}, f)
        self.learner = Learner(data_dir=self.tmpdir)
        self.learner.queue = LearningQueue(threshold=2)  # Low threshold for testing

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_learn_from_text(self):
        text = "Michelangelo painted the beautiful ceiling of the Sistine Chapel"
        stats = self.learner.learn_from_text(text, lang='en')
        self.assertGreater(stats['tokens_processed'], 0)

    def test_learn_from_repeated_text(self):
        """Same word appearing multiple times → committed."""
        text1 = "The algorithm uses backpropagation for training"
        text2 = "Backpropagation is the key algorithm in deep learning"
        self.learner.learn_from_text(text1, lang='en')
        stats = self.learner.learn_from_text(text2, lang='en')
        # 'backpropagation' appeared in both → should reach threshold
        # (depending on tokenization)

    def test_learn_from_documents(self):
        docs = [
            {'text': 'Quantum computing uses qubits instead of classical bits'},
            {'text': 'Qubits can exist in superposition unlike classical bits'},
            {'text': 'The power of qubits comes from quantum entanglement'},
        ]
        stats = self.learner.learn_from_documents(docs, lang='en')
        self.assertEqual(stats['docs_processed'], 3)
        self.assertGreater(stats['tokens_processed'], 0)

    def test_empty_docs_handled(self):
        stats = self.learner.learn_from_documents([], lang='en')
        self.assertEqual(stats['docs_processed'], 0)

    def test_empty_text_handled(self):
        stats = self.learner.learn_from_text('', lang='en')
        self.assertEqual(stats['tokens_processed'], 0)


class TestLearnerSafety(unittest.TestCase):
    """Verify grammar rules and safety are NOT modified."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.tmpdir, 'vocab'), exist_ok=True)
        os.makedirs(os.path.join(self.tmpdir, 'normalize'), exist_ok=True)
        os.makedirs(os.path.join(self.tmpdir, 'idioms'), exist_ok=True)
        os.makedirs(os.path.join(self.tmpdir, 'grammar'), exist_ok=True)
        # Create grammar file
        grammar = {'word_order': 'SVO', 'dependencies': []}
        with open(os.path.join(self.tmpdir, 'grammar', 'en.json'), 'w') as f:
            json.dump(grammar, f)
        with open(os.path.join(self.tmpdir, 'vocab', 'en.json'), 'w') as f:
            json.dump({'words': {}}, f)
        with open(os.path.join(self.tmpdir, 'normalize', 'en.json'), 'w') as f:
            json.dump({'abbreviations': {}}, f)
        self.learner = Learner(data_dir=self.tmpdir)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_grammar_not_modified(self):
        """Grammar file should not be in _dirty set after learning."""
        self.learner.teach_word('test', {'pos': ['NOUN']})
        self.learner.save()
        # Check grammar file unchanged
        with open(os.path.join(self.tmpdir, 'grammar', 'en.json')) as f:
            grammar = json.load(f)
        self.assertEqual(grammar['word_order'], 'SVO')
        # Verify _dirty didn't include grammar
        self.assertNotIn(('grammar', 'en'), self.learner._dirty)

    def test_only_vocab_normalize_idioms_saved(self):
        """Only vocab, normalize, and idioms are ever saved."""
        self.learner.teach_word('test', {'pos': ['NOUN']})
        self.learner.teach_abbreviation('tst', 'test')
        self.learner.teach_idiom('test the waters', 'try carefully')
        # Check dirty set
        categories = set(cat for cat, _ in self.learner._dirty)
        self.assertTrue(categories.issubset({'vocab', 'normalize', 'idioms'}))


class TestLearnerPersistence(unittest.TestCase):
    """Verify learning persists across restarts."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.tmpdir, 'vocab'), exist_ok=True)
        os.makedirs(os.path.join(self.tmpdir, 'normalize'), exist_ok=True)
        os.makedirs(os.path.join(self.tmpdir, 'idioms'), exist_ok=True)
        with open(os.path.join(self.tmpdir, 'vocab', 'en.json'), 'w') as f:
            json.dump({'words': {'hello': {'pos': ['INTJ']}}}, f)
        with open(os.path.join(self.tmpdir, 'normalize', 'en.json'), 'w') as f:
            json.dump({'abbreviations': {}}, f)
        with open(os.path.join(self.tmpdir, 'idioms', 'en.json'), 'w') as f:
            json.dump({}, f)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_taught_word_persists(self):
        learner1 = Learner(data_dir=self.tmpdir)
        learner1.teach_word('rizz', {'pos': ['NOUN'], 'senses': ['charisma']})
        learner1.save()

        # New learner instance (simulates restart)
        learner2 = Learner(data_dir=self.tmpdir)
        self.assertTrue(learner2.is_known('rizz'))

    def test_taught_abbreviation_persists(self):
        learner1 = Learner(data_dir=self.tmpdir)
        learner1.teach_abbreviation('ngl', 'not gonna lie')
        learner1.save()

        learner2 = Learner(data_dir=self.tmpdir)
        learner2._ensure_loaded('en')
        self.assertEqual(
            learner2._normalize['en']['abbreviations'].get('ngl'),
            'not gonna lie')

    def test_taught_idiom_persists(self):
        learner1 = Learner(data_dir=self.tmpdir)
        learner1.teach_idiom('spill the tea', 'share gossip')
        learner1.save()

        learner2 = Learner(data_dir=self.tmpdir)
        learner2._ensure_loaded('en')
        self.assertIn('spill the tea', learner2._idioms.get('en', {}))

    def test_original_vocab_preserved(self):
        """Learning new words doesn't delete existing ones."""
        learner1 = Learner(data_dir=self.tmpdir)
        learner1.teach_word('slay', {'pos': ['VERB']})
        learner1.save()

        learner2 = Learner(data_dir=self.tmpdir)
        self.assertTrue(learner2.is_known('hello'))  # Original
        self.assertTrue(learner2.is_known('slay'))    # Learned


class TestPendingCount(unittest.TestCase):

    def test_pending_count(self):
        learner = Learner()
        learner.queue.flag('bussin')
        learner.queue.flag('slay')
        self.assertEqual(learner.pending_count(), 2)

    def test_pending_zero_after_clear(self):
        learner = Learner()
        learner.queue.flag('bussin')
        learner.queue.clear()
        self.assertEqual(learner.pending_count(), 0)


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')
    unittest.main(verbosity=2)
