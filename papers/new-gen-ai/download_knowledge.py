#!/usr/bin/env python3
"""
Download quality knowledge datasets for Guru brain training.

Domains: medicine, psychology, philosophy, science, history, literature,
         poetry, essays, debates, law, economics, math, CS.

Sources: HuggingFace Datasets (all open/public domain).

Usage:
    python3 download_knowledge.py              # download all
    python3 download_knowledge.py --domain medicine  # single domain
"""

import json, os, sys, argparse, time
from pathlib import Path

DATA_DIR = Path.home() / "webmind-research" / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def save_jsonl(records, name):
    path = DATA_DIR / f"{name}.jsonl"
    with open(path, 'w') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f"  {name}: {len(records):,} records → {path}")
    return len(records)


def download_hf(dataset_name, split="train", limit=None, config=None):
    """Download from HuggingFace datasets."""
    from datasets import load_dataset
    kwargs = {}
    if config:
        kwargs['name'] = config
    ds = load_dataset(dataset_name, split=split, trust_remote_code=True, **kwargs)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    return ds


def dl_medicine(limit=20000):
    """PubMedQA + MedQA + medical Wikipedia."""
    records = []

    # PubMedQA — biomedical Q&A with reasoning
    print("  Downloading PubMedQA...")
    try:
        ds = download_hf("qiaojin/PubMedQA", split="train", config="pqa_labeled")
        for item in ds:
            q = item.get('question', '')
            ctx = ' '.join(item.get('context', {}).get('contexts', []))
            a = item.get('long_answer', '')
            if q and a:
                records.append({'question': q, 'answer': a, 'source': 'pubmedqa'})
            if ctx:
                records.append({'text': ctx, 'source': 'pubmedqa'})
            if len(records) >= limit:
                break
    except Exception as e:
        print(f"    PubMedQA: {e}")

    # MedMCQA — medical MCQ with explanations
    print("  Downloading MedMCQA...")
    try:
        ds = download_hf("openlifescienceai/medmcqa", split="train", limit=limit)
        for item in ds:
            q = item.get('question', '')
            exp = item.get('exp', '')
            if q and exp and len(exp) > 20:
                records.append({'question': q, 'answer': exp, 'source': 'medmcqa'})
            if len(records) >= limit:
                break
    except Exception as e:
        print(f"    MedMCQA: {e}")

    return save_jsonl(records[:limit], "medicine")


def dl_psychology(limit=15000):
    """Psychology Q&A + mental health."""
    records = []

    # MMLU psychology
    print("  Downloading MMLU psychology...")
    try:
        for sub in ["professional_psychology", "high_school_psychology"]:
            ds = download_hf("cais/mmlu", split="test", config=sub)
            for item in ds:
                q = item.get('question', '')
                choices = item.get('choices', [])
                answer_idx = item.get('answer', 0)
                if q and choices and answer_idx < len(choices):
                    records.append({
                        'question': q,
                        'answer': choices[answer_idx],
                        'source': f'mmlu_{sub}'
                    })
    except Exception as e:
        print(f"    MMLU psych: {e}")

    # CounselChat — therapy conversations
    print("  Downloading CounselChat...")
    try:
        ds = download_hf("nbertagnolli/counsel-chat", split="train", limit=limit)
        for item in ds:
            q = item.get('questionTitle', '') or item.get('questionText', '')
            a = item.get('answerText', '')
            if q and a and len(a) > 30:
                records.append({'question': q, 'answer': a, 'source': 'counselchat'})
    except Exception as e:
        print(f"    CounselChat: {e}")

    return save_jsonl(records[:limit], "psychology")


def dl_philosophy(limit=15000):
    """Philosophy texts + ethics + debates."""
    records = []

    # PhilPapers / philosophy Q&A from MMLU
    print("  Downloading MMLU philosophy+ethics...")
    try:
        for sub in ["philosophy", "moral_scenarios", "moral_disputes", "formal_logic"]:
            ds = download_hf("cais/mmlu", split="test", config=sub)
            for item in ds:
                q = item.get('question', '')
                choices = item.get('choices', [])
                answer_idx = item.get('answer', 0)
                if q and choices and answer_idx < len(choices):
                    records.append({
                        'question': q,
                        'answer': choices[answer_idx],
                        'source': f'mmlu_{sub}'
                    })
    except Exception as e:
        print(f"    MMLU philosophy: {e}")

    # ELI5 — explain like I'm 5 (long-form explanations)
    print("  Downloading ELI5 (philosophy/science explanations)...")
    try:
        ds = download_hf("eli5_category", split="train", limit=limit)
        for item in ds:
            q = item.get('title', '')
            answers = item.get('answers', {})
            texts = answers.get('text', []) if isinstance(answers, dict) else []
            if q and texts:
                best = max(texts, key=len) if texts else ''
                if len(best) > 50:
                    records.append({'question': q, 'answer': best[:1000], 'source': 'eli5'})
    except Exception as e:
        print(f"    ELI5: {e}")

    return save_jsonl(records[:limit], "philosophy_ethics")


def dl_literature(limit=20000):
    """Poetry, essays, public domain books from Project Gutenberg."""
    records = []

    # Poem Sentiment — poems with analysis
    print("  Downloading poems...")
    try:
        ds = download_hf("poem_sentiment", split="train")
        for item in ds:
            verse = item.get('verse_text', '')
            if verse and len(verse) > 20:
                records.append({'text': verse, 'source': 'poem_sentiment'})
    except Exception as e:
        print(f"    Poems: {e}")

    # Gutenberg poetry corpus
    print("  Downloading Gutenberg poetry...")
    try:
        ds = download_hf("matthh/gutenberg-poetry-corpus", split="train", limit=limit)
        for item in ds:
            text = item.get('text', '') or item.get('line', '')
            if text and len(text) > 10:
                records.append({'text': text, 'source': 'gutenberg_poetry'})
            if len(records) >= limit:
                break
    except Exception as e:
        print(f"    Gutenberg poetry: {e}")

    # BookCorpus summaries / quotes
    print("  Downloading book quotes/summaries...")
    try:
        ds = download_hf("Abirate/english_quotes", split="train")
        for item in ds:
            quote = item.get('quote', '')
            author = item.get('author', '')
            if quote and len(quote) > 20:
                text = f'{quote} — {author}' if author else quote
                records.append({'text': text, 'source': 'english_quotes'})
    except Exception as e:
        print(f"    Quotes: {e}")

    return save_jsonl(records[:limit], "literature")


def dl_history_politics(limit=20000):
    """History, political science, debates."""
    records = []

    # MMLU history + political science
    print("  Downloading MMLU history/politics...")
    try:
        for sub in ["high_school_us_history", "high_school_world_history",
                     "high_school_government_and_politics", "us_foreign_policy",
                     "international_law", "jurisprudence"]:
            try:
                ds = download_hf("cais/mmlu", split="test", config=sub)
                for item in ds:
                    q = item.get('question', '')
                    choices = item.get('choices', [])
                    answer_idx = item.get('answer', 0)
                    if q and choices and answer_idx < len(choices):
                        records.append({
                            'question': q,
                            'answer': choices[answer_idx],
                            'source': f'mmlu_{sub}'
                        })
            except Exception:
                pass
    except Exception as e:
        print(f"    MMLU history: {e}")

    return save_jsonl(records[:limit], "history_politics")


def dl_science(limit=20000):
    """Physics, chemistry, biology, earth science."""
    records = []

    print("  Downloading SciQ...")
    try:
        ds = download_hf("allenai/sciq", split="train", limit=limit)
        for item in ds:
            q = item.get('question', '')
            a = item.get('correct_answer', '')
            support = item.get('support', '')
            if q and a:
                answer = f"{a}. {support}" if support else a
                records.append({'question': q, 'answer': answer, 'source': 'sciq'})
    except Exception as e:
        print(f"    SciQ: {e}")

    # MMLU sciences
    print("  Downloading MMLU sciences...")
    try:
        for sub in ["high_school_biology", "high_school_chemistry",
                     "high_school_physics", "college_biology", "college_chemistry",
                     "college_physics", "anatomy", "astronomy",
                     "conceptual_physics", "electrical_engineering"]:
            try:
                ds = download_hf("cais/mmlu", split="test", config=sub)
                for item in ds:
                    q = item.get('question', '')
                    choices = item.get('choices', [])
                    answer_idx = item.get('answer', 0)
                    if q and choices and answer_idx < len(choices):
                        records.append({
                            'question': q,
                            'answer': choices[answer_idx],
                            'source': f'mmlu_{sub}'
                        })
            except Exception:
                pass
    except Exception as e:
        print(f"    MMLU science: {e}")

    return save_jsonl(records[:limit], "science_extended")


def dl_math_reasoning(limit=15000):
    """Math word problems, proofs, logic."""
    records = []

    # MATH dataset — competition math with solutions
    print("  Downloading MATH competition problems...")
    try:
        ds = download_hf("lighteval/MATH", split="train", limit=limit)
        for item in ds:
            q = item.get('problem', '')
            a = item.get('solution', '')
            if q and a:
                records.append({'question': q, 'answer': a, 'source': 'math_competition'})
    except Exception as e:
        print(f"    MATH: {e}")

    # LogiQA — logical reasoning
    print("  Downloading LogiQA...")
    try:
        ds = download_hf("lucasmccabe/logiqa", split="train", limit=limit)
        for item in ds:
            ctx = item.get('context', '')
            q = item.get('query', '')
            if ctx and q:
                records.append({'text': f"{ctx} {q}", 'source': 'logiqa'})
    except Exception as e:
        print(f"    LogiQA: {e}")

    return save_jsonl(records[:limit], "math_reasoning")


def dl_law_economics(limit=15000):
    """Legal text, economics, business."""
    records = []

    print("  Downloading MMLU law/economics...")
    try:
        for sub in ["professional_law", "professional_accounting",
                     "high_school_macroeconomics", "high_school_microeconomics",
                     "econometrics", "business_ethics", "management",
                     "marketing", "global_facts"]:
            try:
                ds = download_hf("cais/mmlu", split="test", config=sub)
                for item in ds:
                    q = item.get('question', '')
                    choices = item.get('choices', [])
                    answer_idx = item.get('answer', 0)
                    if q and choices and answer_idx < len(choices):
                        records.append({
                            'question': q,
                            'answer': choices[answer_idx],
                            'source': f'mmlu_{sub}'
                        })
            except Exception:
                pass
    except Exception as e:
        print(f"    MMLU law/econ: {e}")

    return save_jsonl(records[:limit], "law_economics")


DOMAINS = {
    'medicine': dl_medicine,
    'psychology': dl_psychology,
    'philosophy': dl_philosophy,
    'literature': dl_literature,
    'history': dl_history_politics,
    'science': dl_science,
    'math': dl_math_reasoning,
    'law': dl_law_economics,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", default="all")
    parser.add_argument("--limit", type=int, default=20000)
    args = parser.parse_args()

    t0 = time.time()
    total = 0

    if args.domain == 'all':
        for name, fn in DOMAINS.items():
            print(f"\n=== {name.upper()} ===")
            try:
                n = fn(args.limit)
                total += n
            except Exception as e:
                print(f"  FAILED: {e}")
    else:
        fn = DOMAINS.get(args.domain)
        if fn:
            total = fn(args.limit)
        else:
            print(f"Unknown domain: {args.domain}. Options: {list(DOMAINS.keys())}")

    elapsed = time.time() - t0
    print(f"\nDone: {total:,} records downloaded in {elapsed:.0f}s")


if __name__ == '__main__':
    main()
