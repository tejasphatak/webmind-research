"""
Vocabulary Ingestion Pipeline — pull open-source vocabulary into ULI JSON format.

Sources:
  1. NLTK WordNet (English) — bundled, instant
  2. Open Multilingual Wordnet via NLTK — 150+ languages
  3. Wiktionary extract (wiktextract) — all languages, richest data

Outputs: data/vocab/{lang}.json in ULI format
"""

import json
import os
import sys
from collections import Counter, defaultdict

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


# ============================================================
# Source 1: NLTK WordNet (English — instant, no download needed)
# ============================================================

def ingest_wordnet_english(max_words=50000):
    """Extract English vocabulary from NLTK WordNet."""
    import nltk
    try:
        from nltk.corpus import wordnet as wn
        wn.synsets('test')  # Trigger download check
    except LookupError:
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        from nltk.corpus import wordnet as wn

    print(f"Extracting English vocabulary from WordNet...")

    vocab = {'words': {}, 'stop_words': [], 'question_words': []}

    # POS mapping: WordNet POS → Universal POS
    WN_TO_UPOS = {
        'n': 'NOUN', 'v': 'VERB', 'a': 'ADJ', 'r': 'ADV', 's': 'ADJ'
    }

    word_count = 0
    for synset in wn.all_synsets():
        for lemma in synset.lemmas():
            word = lemma.name().replace('_', ' ').lower()
            if word_count >= max_words:
                break
            if len(word) < 2 or not word.isalpha():
                continue

            pos = WN_TO_UPOS.get(synset.pos(), 'NOUN')
            definition = synset.definition()

            if word not in vocab['words']:
                vocab['words'][word] = {
                    'pos': [pos],
                    'senses': [],
                    'frequency': 0.0,
                }
                word_count += 1

            entry = vocab['words'][word]
            if pos not in entry['pos']:
                entry['pos'].append(pos)

            # Add sense (definition) — cap at 3 per word
            if len(entry.get('senses', [])) < 3:
                sense = definition[:100] if definition else ''
                if sense and sense not in entry['senses']:
                    entry['senses'].append(sense)

    # Add common stop words and question words
    vocab['stop_words'] = [
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'shall', 'can', 'of', 'in', 'to', 'for',
        'on', 'at', 'by', 'with', 'from', 'up', 'about', 'into', 'through',
        'during', 'before', 'after', 'above', 'below', 'between', 'out',
        'off', 'over', 'under', 'again', 'further', 'then', 'once', 'and',
        'but', 'or', 'nor', 'not', 'no', 'so', 'if', 'that', 'this', 'these',
        'those', 'it', 'its', 'i', 'me', 'my', 'we', 'our', 'you', 'your',
        'he', 'him', 'his', 'she', 'her', 'they', 'them', 'their', 'who',
        'what', 'which', 'when', 'where', 'why', 'how', 'all', 'each',
        'every', 'both', 'few', 'more', 'most', 'some', 'any', 'other',
        'than', 'too', 'very', 'just', 'also', 'now', 'here', 'there',
    ]
    vocab['question_words'] = [
        'what', 'who', 'whom', 'where', 'when', 'why', 'how', 'which',
        'whose', 'whoever', 'whatever', 'whichever', 'wherever', 'whenever',
    ]

    print(f"  Extracted {len(vocab['words'])} English words from WordNet")
    return vocab


# ============================================================
# Source 2: Open Multilingual Wordnet (via NLTK)
# ============================================================

LANGUAGE_CODES = {
    'eng': 'en', 'fra': 'fr', 'spa': 'es', 'deu': 'de', 'ita': 'it',
    'por': 'pt', 'nld': 'nl', 'rus': 'ru', 'jpn': 'ja', 'zho': 'zh',
    'kor': 'ko', 'ara': 'ar', 'hin': 'hi', 'tha': 'th', 'vie': 'vi',
    'ind': 'id', 'msa': 'ms', 'tur': 'tr', 'pol': 'pl', 'swe': 'sv',
    'fin': 'fi', 'dan': 'da', 'nor': 'no', 'ell': 'el', 'heb': 'he',
    'cat': 'ca', 'eus': 'eu', 'glg': 'gl', 'slv': 'sl', 'hrv': 'hr',
    'bul': 'bg', 'ces': 'cs', 'ron': 'ro', 'lit': 'lt', 'lav': 'lv',
    'est': 'et', 'hun': 'hu', 'slk': 'sk', 'fas': 'fa', 'als': 'sq',
}


def ingest_omw_language(lang_code_3, max_words=20000):
    """Extract vocabulary for a language from Open Multilingual Wordnet."""
    import nltk
    try:
        from nltk.corpus import wordnet as wn
        wn.synsets('test')
    except LookupError:
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        from nltk.corpus import wordnet as wn

    lang_2 = LANGUAGE_CODES.get(lang_code_3, lang_code_3[:2])

    print(f"Extracting {lang_code_3} ({lang_2}) vocabulary from OMW...")

    WN_TO_UPOS = {'n': 'NOUN', 'v': 'VERB', 'a': 'ADJ', 'r': 'ADV', 's': 'ADJ'}

    vocab = {'words': {}, 'language': lang_2}
    word_count = 0

    for synset in wn.all_synsets():
        if word_count >= max_words:
            break
        try:
            lemmas = synset.lemma_names(lang_code_3)
        except Exception:
            continue

        pos = WN_TO_UPOS.get(synset.pos(), 'NOUN')
        definition = synset.definition()

        for lemma_name in lemmas:
            word = lemma_name.replace('_', ' ').lower()
            if len(word) < 2:
                continue

            if word not in vocab['words']:
                vocab['words'][word] = {
                    'pos': [pos],
                    'senses': [],
                }
                word_count += 1

            entry = vocab['words'][word]
            if pos not in entry['pos']:
                entry['pos'].append(pos)
            if definition and len(entry.get('senses', [])) < 3:
                sense = definition[:100]
                if sense not in entry['senses']:
                    entry['senses'].append(sense)

    print(f"  Extracted {len(vocab['words'])} words for {lang_2}")
    return vocab


# ============================================================
# Save to ULI format
# ============================================================

def save_vocab(vocab, lang, data_dir=DATA_DIR):
    """Save vocabulary to data/vocab/{lang}.json."""
    vocab_dir = os.path.join(data_dir, 'vocab')
    ensure_dir(vocab_dir)
    path = os.path.join(vocab_dir, f'{lang}.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, indent=2, ensure_ascii=False)
    size_kb = os.path.getsize(path) / 1024
    print(f"  Saved {path} ({size_kb:.0f} KB, {len(vocab.get('words', {}))} words)")
    return path


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("ULI Vocabulary Ingestion Pipeline")
    print("=" * 60)

    # Parse args
    if '--english-only' in sys.argv:
        languages = ['eng']
    elif '--all' in sys.argv:
        languages = list(LANGUAGE_CODES.keys())
    else:
        # Default: top 15 languages
        languages = ['eng', 'fra', 'spa', 'deu', 'ita', 'por', 'hin',
                     'jpn', 'zho', 'kor', 'ara', 'rus', 'tur', 'pol', 'nld']

    max_words = int(sys.argv[sys.argv.index('--max') + 1]) if '--max' in sys.argv else 20000

    # English: use full WordNet (richest)
    if 'eng' in languages:
        en_vocab = ingest_wordnet_english(max_words=max_words)
        save_vocab(en_vocab, 'en')
        languages.remove('eng')

    # Other languages: use OMW
    for lang_3 in languages:
        lang_2 = LANGUAGE_CODES.get(lang_3, lang_3[:2])
        try:
            vocab = ingest_omw_language(lang_3, max_words=max_words)
            if vocab['words']:
                save_vocab(vocab, lang_2)
            else:
                print(f"  No words found for {lang_3}, skipping")
        except Exception as e:
            print(f"  Error for {lang_3}: {e}")

    print(f"\n{'=' * 60}")
    print("Done. Vocabulary files in data/vocab/")
    # List all vocab files
    vocab_dir = os.path.join(DATA_DIR, 'vocab')
    if os.path.exists(vocab_dir):
        files = sorted(os.listdir(vocab_dir))
        total_size = sum(os.path.getsize(os.path.join(vocab_dir, f))
                        for f in files if f.endswith('.json'))
        print(f"  {len(files)} language files, {total_size/1024:.0f} KB total")
        for f in files:
            size = os.path.getsize(os.path.join(vocab_dir, f)) / 1024
            print(f"    {f}: {size:.0f} KB")


if __name__ == '__main__':
    main()
