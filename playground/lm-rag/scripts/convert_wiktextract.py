"""
Convert Wiktextract JSONL to ULI vocab JSON format.

Input: kaikki.org JSONL files (one word entry per line)
Output: data/vocab/{lang}.json in ULI format

Usage:
  python3 scripts/convert_wiktextract.py /tmp/wikt_downloads/hi.jsonl hi
  python3 scripts/convert_wiktextract.py /tmp/wikt_downloads/ all   # convert all files in dir
"""

import json
import os
import sys
from collections import Counter

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')

# Map Wiktionary POS to Universal POS
WIKT_POS_MAP = {
    'noun': 'NOUN', 'verb': 'VERB', 'adj': 'ADJ', 'adv': 'ADV',
    'pron': 'PRON', 'prep': 'ADP', 'conj': 'CCONJ', 'det': 'DET',
    'num': 'NUM', 'part': 'PART', 'intj': 'INTJ', 'name': 'PROPN',
    'abbrev': 'NOUN', 'affix': 'NOUN', 'phrase': 'NOUN',
    'proverb': 'NOUN', 'prefix': 'NOUN', 'suffix': 'NOUN',
    'particle': 'PART', 'postposition': 'ADP', 'article': 'DET',
    'character': 'NOUN', 'symbol': 'SYM', 'number': 'NUM',
    'punctuation': 'PUNCT', 'romanization': 'NOUN',
}


def convert_file(jsonl_path, lang_code, max_words=20000):
    """Convert a single Wiktextract JSONL to ULI vocab JSON."""
    print(f"Converting {jsonl_path} → {lang_code}...")

    vocab = {'words': {}, 'language': lang_code}
    word_count = 0

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if word_count >= max_words:
                break
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            word = entry.get('word', '').lower().strip()
            if not word or len(word) < 2:
                continue

            # Get POS
            wikt_pos = entry.get('pos', 'noun').lower()
            upos = WIKT_POS_MAP.get(wikt_pos, 'NOUN')

            # Get senses/definitions
            senses = []
            for sense in entry.get('senses', [])[:3]:
                glosses = sense.get('glosses', [])
                if glosses:
                    senses.append(glosses[0][:100])

            # Get forms (conjugations, declensions)
            forms = {}
            for form_entry in entry.get('forms', [])[:5]:
                form_word = form_entry.get('form', '')
                tags = form_entry.get('tags', [])
                if form_word and tags:
                    tag_key = '_'.join(tags[:2])
                    forms[tag_key] = form_word

            if word not in vocab['words']:
                vocab['words'][word] = {
                    'pos': [upos],
                    'senses': senses,
                }
                if forms:
                    vocab['words'][word]['forms'] = forms
                word_count += 1
            else:
                existing = vocab['words'][word]
                if upos not in existing['pos']:
                    existing['pos'].append(upos)
                for s in senses:
                    if s not in existing.get('senses', []) and len(existing.get('senses', [])) < 3:
                        existing.setdefault('senses', []).append(s)

    # Save
    os.makedirs(os.path.join(DATA_DIR, 'vocab'), exist_ok=True)
    out_path = os.path.join(DATA_DIR, 'vocab', f'{lang_code}.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False)

    size_kb = os.path.getsize(out_path) / 1024
    print(f"  {len(vocab['words']):,} words → {out_path} ({size_kb:.0f} KB)")
    return len(vocab['words'])


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 convert_wiktextract.py <path> <lang_code|all> [--max N]")
        sys.exit(1)

    path = sys.argv[1]
    target = sys.argv[2] if len(sys.argv) > 2 else 'all'
    max_words = 20000
    if '--max' in sys.argv:
        max_words = int(sys.argv[sys.argv.index('--max') + 1])

    if os.path.isdir(path):
        # Convert all JSONL files in directory
        total = 0
        for fname in sorted(os.listdir(path)):
            if not fname.endswith('.jsonl'):
                continue
            lang_code = fname.replace('.jsonl', '')
            fpath = os.path.join(path, fname)
            # Skip tiny files (failed downloads)
            if os.path.getsize(fpath) < 1000:
                print(f"  Skipping {fname} (too small, likely failed download)")
                continue
            n = convert_file(fpath, lang_code, max_words=max_words)
            total += n
        print(f"\nTotal: {total:,} words across all languages")
    else:
        convert_file(path, target, max_words=max_words)


if __name__ == '__main__':
    main()
