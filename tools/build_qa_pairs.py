#!/usr/bin/env python3
"""
Build Q&A Pairs — Convert all data to (question, complete_answer) format
=========================================================================
Every entry becomes a retrievable Q&A pair where the answer is a complete,
grammatical English sentence. No grammar model needed — the stored text IS
the grammar.
"""

import json, os, re
from pathlib import Path

DATA_DIR = Path(os.environ.get("DATA_DIR", os.path.expanduser("~/webmind-research/data")))
OUT = Path(os.environ.get("OUT", os.path.expanduser("~/webmind-research/trained_model/qa_pairs.jsonl")))


def clean(text):
    """Clean text: remove extra whitespace, truncate."""
    text = re.sub(r'\s+', ' ', text).strip()
    return text[:500]


def process_wikipedia():
    """Wikipedia facts → Q&A pairs."""
    pairs = []
    f = DATA_DIR / "wikipedia_simple.jsonl"
    if not f.exists():
        return pairs
    with open(f) as fh:
        for line in fh:
            row = json.loads(line)
            text = clean(row.get("text", ""))
            topic = row.get("topic", "")
            if len(text) < 30:
                continue
            # The text itself is the answer. Generate natural questions.
            sentences = [s.strip() + '.' for s in text.split('.') if len(s.strip()) > 15]
            if sentences:
                # Store the whole paragraph as answer to "tell me about X"
                pairs.append({
                    "question": f"Tell me about {topic}",
                    "answer": text[:300],
                    "source": "wikipedia"
                })
                # First sentence as answer to "what is X"
                pairs.append({
                    "question": f"What is {topic}?",
                    "answer": sentences[0],
                    "source": "wikipedia"
                })
    return pairs


def process_squad():
    """SQuAD already has Q&A pairs."""
    pairs = []
    f = DATA_DIR / "squad.jsonl"
    if not f.exists():
        return pairs
    with open(f) as fh:
        for line in fh:
            row = json.loads(line)
            source = row.get("source", "")
            if source == "squad_qa":
                q = clean(row.get("text", "").split("Answer:")[0])
                a = row.get("answer", "")
                if q and a and len(a) > 1:
                    pairs.append({"question": q, "answer": clean(a), "source": "squad"})
            elif source == "squad":
                text = clean(row.get("text", ""))
                topic = row.get("topic", "")
                if text and topic:
                    pairs.append({
                        "question": f"What do you know about {topic}?",
                        "answer": text[:300],
                        "source": "squad"
                    })
    return pairs


def process_hotpotqa():
    """HotpotQA has Q&A pairs + context."""
    pairs = []
    f = DATA_DIR / "hotpotqa.jsonl"
    if not f.exists():
        return pairs
    with open(f) as fh:
        for line in fh:
            row = json.loads(line)
            text = row.get("text", "")
            answer = row.get("answer", "")
            source = row.get("source", "")

            if source == "hotpotqa" and "Answer:" in text:
                q = clean(text.split("Answer:")[0])
                if q and answer and len(answer) > 1:
                    pairs.append({"question": q, "answer": clean(answer), "source": "hotpotqa"})
            elif source == "hotpotqa_context":
                text = clean(text)
                topic = row.get("topic", "")
                if text and len(text) > 30:
                    pairs.append({
                        "question": f"Tell me about {topic}",
                        "answer": text[:300],
                        "source": "hotpotqa"
                    })
    return pairs


def process_natural_questions():
    """Natural Questions — real Google searches."""
    pairs = []
    f = DATA_DIR / "natural_questions.jsonl"
    if not f.exists():
        return pairs
    with open(f) as fh:
        for line in fh:
            row = json.loads(line)
            q = clean(row.get("text", ""))
            a = row.get("answer", "")
            if q and a and len(a) > 1:
                pairs.append({"question": q, "answer": clean(a), "source": "nq"})
    return pairs


def process_reasoning_datasets():
    """MMLU, ARC, GSM8K, BBH — reasoning with answers."""
    pairs = []
    for fname in ("mmlu_pro", "arc", "gsm8k", "mmlu_physics", "mmlu_philosophy",
                  "mmlu_astronomy", "mmlu_compsec", "mmlu_algebra", "mmlu_religions",
                  "bbh_causal", "bbh_logic"):
        f = DATA_DIR / f"{fname}.jsonl"
        if not f.exists():
            continue
        with open(f) as fh:
            for line in fh:
                row = json.loads(line)
                q = clean(row.get("question", row.get("text", "")))
                a = row.get("answer", "")
                if q and a and len(str(a)) > 1:
                    pairs.append({"question": q, "answer": clean(str(a)), "source": fname})
    return pairs


def process_hle():
    """HLE questions with answers."""
    pairs = []
    f = DATA_DIR / "hle.jsonl"
    if not f.exists():
        return pairs
    with open(f) as fh:
        for line in fh:
            row = json.loads(line)
            q = clean(row.get("question", ""))
            a = row.get("answer", "")
            if q and a:
                pairs.append({"question": q, "answer": clean(str(a)), "source": "hle"})
    return pairs


def add_conversational():
    """Basic conversational pairs so it can handle greetings."""
    return [
        {"question": "How are you?", "answer": "I'm doing well, thank you for asking! How can I help you?", "source": "conversational"},
        {"question": "Hello", "answer": "Hello! I'm Webmind, a distributed knowledge engine. Ask me anything.", "source": "conversational"},
        {"question": "Hi", "answer": "Hi there! What would you like to know?", "source": "conversational"},
        {"question": "What are you?", "answer": "I'm Webmind, a distributed knowledge system that retrieves facts from a network of nodes. I don't generate text — I find and return verified information.", "source": "conversational"},
        {"question": "Who made you?", "answer": "I was created by the Webmind project — an open-source effort to build distributed, verifiable AI that runs on any device.", "source": "conversational"},
        {"question": "Thank you", "answer": "You're welcome! Feel free to ask anything else.", "source": "conversational"},
        {"question": "Goodbye", "answer": "Goodbye! Come back anytime.", "source": "conversational"},
        {"question": "What can you do?", "answer": "I can answer questions by searching a distributed knowledge base of verified facts. I never make things up — every answer comes from a real source.", "source": "conversational"},
        {"question": "Help", "answer": "Just type your question and I'll search my knowledge base for the best answer. I cover science, history, math, technology, and more.", "source": "conversational"},
    ]


def main():
    print("=== Building Q&A Pairs ===\n", flush=True)

    all_pairs = []

    processors = [
        ("Wikipedia", process_wikipedia),
        ("SQuAD", process_squad),
        ("HotpotQA", process_hotpotqa),
        ("Natural Questions", process_natural_questions),
        ("Reasoning", process_reasoning_datasets),
        ("HLE", process_hle),
        ("Conversational", add_conversational),
    ]

    for name, fn in processors:
        pairs = fn()
        all_pairs.extend(pairs)
        print(f"  {name}: {len(pairs):,} pairs", flush=True)

    # Deduplicate by question
    seen = set()
    unique = []
    for p in all_pairs:
        key = p["question"].lower().strip()[:100]
        if key not in seen:
            seen.add(key)
            unique.append(p)

    print(f"\n  Total: {len(all_pairs):,} → {len(unique):,} unique", flush=True)

    # Save
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        for p in unique:
            f.write(json.dumps(p) + "\n")

    print(f"  Saved to {OUT}", flush=True)
    print(f"  Size: {OUT.stat().st_size / 1e6:.1f} MB", flush=True)

    # Show samples
    print(f"\n  Samples:", flush=True)
    import random
    for p in random.sample(unique, min(5, len(unique))):
        print(f"    Q: {p['question'][:60]}", flush=True)
        print(f"    A: {p['answer'][:80]}", flush=True)
        print(flush=True)


if __name__ == "__main__":
    main()
