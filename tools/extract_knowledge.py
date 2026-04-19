#!/usr/bin/env python3
"""
Knowledge Extraction — Systematically extract Q&A pairs from GPT-2
===================================================================
Uses templates + entity lists to probe GPT-2 for knowledge.
Filters bad answers. Stores as Q&A pairs for SAQT.
"""

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import json, time, os, random
from pathlib import Path

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT = Path(os.environ.get("OUT", "/workspace/extracted_qa.jsonl"))


def generate(model, tokenizer, prompt, max_new=60):
    inp = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=128).to(DEVICE)
    with torch.no_grad():
        out = model.generate(inp['input_ids'], max_new_tokens=max_new,
                            do_sample=False, pad_token_id=tokenizer.eos_token_id,
                            temperature=1.0)
    text = tokenizer.decode(out[0][inp['input_ids'].size(1):], skip_special_tokens=True)
    # Clean: take first sentence
    for delim in ['\n', '. ', '? ', '! ']:
        if delim in text:
            text = text.split(delim)[0]
            if delim != '\n':
                text += delim.strip()
            break
    return text.strip()


def is_good_answer(answer):
    """Filter garbage answers."""
    if not answer or len(answer) < 5:
        return False
    if answer.count(answer[:3]) > 3:  # Repetitive
        return False
    if len(set(answer.split())) < 3:  # Too few unique words
        return False
    return True


# Entity lists (inline — no external dependency)
COUNTRIES = ["France", "Germany", "Japan", "China", "India", "Brazil", "Russia",
    "United States", "Canada", "Australia", "Mexico", "Italy", "Spain", "Egypt",
    "South Korea", "United Kingdom", "Nigeria", "Argentina", "Turkey", "Iran",
    "Saudi Arabia", "Indonesia", "Thailand", "Vietnam", "Poland", "Sweden",
    "Norway", "Switzerland", "Netherlands", "Belgium", "Greece", "Portugal",
    "Israel", "South Africa", "Kenya", "Colombia", "Peru", "Chile", "Pakistan"]

SCIENTISTS = ["Einstein", "Newton", "Darwin", "Tesla", "Curie", "Galileo",
    "Hawking", "Feynman", "Turing", "Bohr", "Planck", "Faraday", "Maxwell",
    "Pasteur", "Mendel", "Archimedes", "Copernicus", "Kepler", "Heisenberg"]

TOPICS = ["gravity", "evolution", "electricity", "photosynthesis", "DNA",
    "atoms", "climate change", "black holes", "quantum mechanics", "relativity",
    "magnetism", "volcanoes", "earthquakes", "ocean currents", "the water cycle",
    "the immune system", "the nervous system", "plate tectonics", "nuclear fusion",
    "artificial intelligence", "machine learning", "the internet", "democracy",
    "capitalism", "socialism", "the Renaissance", "the Industrial Revolution",
    "World War I", "World War II", "the Cold War", "the Roman Empire",
    "ancient Egypt", "the French Revolution", "the American Revolution"]

MATH_PROMPTS = [
    "2 + 2 =", "3 × 4 =", "10 - 7 =", "100 / 5 =", "15 + 27 =",
    "8 × 9 =", "144 / 12 =", "25 × 4 =", "1000 - 777 =", "50 + 50 =",
    "The square root of 16 is", "The square root of 144 is",
    "Pi is approximately equal to", "The value of e is approximately",
]

CONVERSATIONAL = [
    ("Hello!", "Hello! How can I help you today?"),
    ("How are you?", "I'm doing well, thank you! How can I assist you?"),
    ("Thank you", "You're welcome! Is there anything else I can help with?"),
    ("Goodbye", "Goodbye! Have a great day."),
    ("What's your name?", "I'm Webmind, a distributed knowledge system."),
    ("Who are you?", "I'm Webmind — I find and share verified information from a distributed network."),
    ("Help", "I can answer questions about science, history, math, geography, and more. Just ask!"),
    ("What can you do?", "I search a knowledge base of verified facts to answer your questions. Every answer comes from a real source."),
]


def build_prompts():
    """Generate all prompts to extract knowledge from GPT-2."""
    prompts = []

    # Facts about countries
    for c in COUNTRIES:
        prompts.append((f"The capital of {c} is", f"What is the capital of {c}?", "geography"))
        prompts.append((f"{c} is located in", f"Where is {c} located?", "geography"))
        prompts.append((f"The population of {c} is approximately", f"What is the population of {c}?", "geography"))
        prompts.append((f"The official language of {c} is", f"What language do they speak in {c}?", "geography"))

    # Facts about scientists
    for s in SCIENTISTS:
        prompts.append((f"{s} is famous for", f"What is {s} famous for?", "science"))
        prompts.append((f"{s} was born in", f"When was {s} born?", "science"))
        prompts.append((f"{s} discovered", f"What did {s} discover?", "science"))

    # Topic explanations
    for t in TOPICS:
        prompts.append((f"{t} is", f"What is {t}?", "knowledge"))
        prompts.append((f"The main cause of {t} is", f"What causes {t}?", "knowledge"))
        prompts.append((f"An interesting fact about {t} is that", f"Tell me a fact about {t}", "knowledge"))

    # Math
    for m in MATH_PROMPTS:
        prompts.append((m, m, "math"))

    # How-to procedures
    procedures = [
        "How to boil water", "How to tie a shoelace", "How to change a tire",
        "How to cook rice", "How to write an essay", "How to solve a quadratic equation",
        "How to calculate area of a circle", "How to convert Celsius to Fahrenheit",
        "How to start a fire safely", "How to perform CPR", "How to read a map",
        "How to multiply large numbers", "How to add fractions", "How to find the mean",
        "How to write a for loop in Python", "How to sort a list in Python",
        "How to create a function in JavaScript", "How to make a website",
    ]
    for p in procedures:
        prompts.append((f"{p}:", p + "?", "procedure"))

    # Common sense
    common_sense = [
        "If you drop a glass, it will", "If you put ice in the sun, it will",
        "If you mix red and blue, you get", "If you plant a seed and water it, it will",
        "If you don't eat for a day, you will feel", "If you exercise regularly, you will",
        "Fire needs oxygen to", "Water flows downhill because",
        "The sky appears blue because", "Shadows are longer in the",
    ]
    for cs in common_sense:
        prompts.append((cs, cs.rstrip(", ") + "?", "common_sense"))

    return prompts


def main():
    print(f"=== KNOWLEDGE EXTRACTION ===\n", flush=True)
    print(f"Device: {DEVICE}", flush=True)

    model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    print(f"GPT-2 loaded\n", flush=True)

    prompts = build_prompts()
    print(f"Prompts: {len(prompts)}", flush=True)

    t0 = time.time()
    good, bad = 0, 0

    with open(OUT, 'w') as f:
        # First: add conversational pairs (no GPT-2 needed)
        for q, a in CONVERSATIONAL:
            f.write(json.dumps({"question": q, "answer": a, "source": "conversational"}) + "\n")
            good += 1

        # Then: extract from GPT-2
        for i, (prompt, question, category) in enumerate(prompts):
            answer = generate(model, tokenizer, prompt)

            if is_good_answer(answer):
                # Combine prompt + answer for a complete statement
                full_answer = f"{prompt} {answer}" if not prompt.endswith('=') else f"{prompt} {answer}"
                f.write(json.dumps({
                    "question": question,
                    "answer": full_answer.strip(),
                    "source": f"gpt2_{category}"
                }) + "\n")
                good += 1
            else:
                bad += 1

            if (i + 1) % 50 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                print(f"  {i+1}/{len(prompts)}  good={good} bad={bad}  "
                      f"{rate:.0f} prompts/s  [{elapsed:.0f}s]", flush=True)

    elapsed = time.time() - t0
    print(f"\n=== DONE ===", flush=True)
    print(f"  Good: {good}  Bad: {bad}  Total: {good+bad}", flush=True)
    print(f"  Time: {elapsed:.0f}s  ({good/elapsed:.0f} pairs/s)", flush=True)
    print(f"  Saved: {OUT}", flush=True)

    # Show samples
    print(f"\n  Samples:", flush=True)
    with open(OUT) as f:
        lines = f.readlines()
    for line in random.sample(lines, min(5, len(lines))):
        p = json.loads(line)
        print(f"    Q: {p['question'][:60]}", flush=True)
        print(f"    A: {p['answer'][:80]}", flush=True)
        print(flush=True)


if __name__ == "__main__":
    main()
