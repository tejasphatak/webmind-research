#!/usr/bin/env python3
"""
Context-Faithful Reasoning Kernel Training
============================================
Fine-tune GPT-2 125M to ONLY reason over provided facts.
Never answer from its own knowledge. Pure reasoning engine.

Training data: (facts + question) → coherent answer using only those facts.
Source: SQuAD, HotpotQA (have question + context + answer triples).
"""

import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from sentence_transformers import SentenceTransformer
import json, time, random, os
from pathlib import Path

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = Path(os.environ.get("DATA_DIR", "/workspace/data"))
OUT_DIR = Path(os.environ.get("OUT_DIR", "/workspace/kernel_trained"))
OUT_DIR.mkdir(parents=True, exist_ok=True)


def build_training_data():
    """Build (input, target) pairs from QA datasets."""
    pairs = []

    # SQuAD: has context + question + answer
    squad_file = DATA_DIR / "squad.jsonl"
    if squad_file.exists():
        with open(squad_file) as f:
            for line in f:
                row = json.loads(line)
                text = row.get("text", "")
                q = row.get("question", "")
                a = row.get("answer", "")
                source = row.get("source", "")

                if source == "squad_qa" and q and a:
                    # Find matching context from squad entries
                    inp = f"Facts: {text[:200]}\nQuestion: {q}\nAnswer:"
                    tgt = f" {a}"
                    pairs.append((inp, tgt))
                elif source == "squad" and text:
                    # Context paragraph — create self-answering pairs
                    sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 20]
                    if len(sentences) >= 2:
                        fact = sentences[0] + '.'
                        inp = f"Facts: {text[:300]}\nQuestion: What does this passage say?\nAnswer:"
                        tgt = f" {fact}"
                        pairs.append((inp, tgt))

    # HotpotQA: has question + answer + context
    hotpot_file = DATA_DIR / "hotpotqa.jsonl"
    if hotpot_file.exists():
        with open(hotpot_file) as f:
            for line in f:
                row = json.loads(line)
                text = row.get("text", "")
                a = row.get("answer", "")
                source = row.get("source", "")

                if source == "hotpotqa" and "Answer:" in text and a:
                    q = text.split("Answer:")[0].strip()
                    inp = f"Facts: {text[:200]}\nQuestion: {q[:100]}\nAnswer:"
                    tgt = f" {a}"
                    pairs.append((inp, tgt))
                elif source == "hotpotqa_context" and text:
                    sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 15]
                    if sentences:
                        inp = f"Facts: {text[:300]}\nQuestion: What is this about?\nAnswer:"
                        tgt = f" {sentences[0]}."
                        pairs.append((inp, tgt))

    # Wikipedia: create factual QA from paragraphs
    wiki_file = DATA_DIR / "wikipedia_simple.jsonl"
    if wiki_file.exists():
        with open(wiki_file) as f:
            for line in f:
                row = json.loads(line)
                text = row.get("text", "")
                topic = row.get("topic", "")
                if text and len(text) > 50:
                    sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 15]
                    if sentences:
                        inp = f"Facts: {text[:300]}\nQuestion: Tell me about {topic}.\nAnswer:"
                        tgt = f" {sentences[0]}."
                        pairs.append((inp, tgt))

    # MMLU/ARC: multiple choice with answers
    for f in DATA_DIR.glob("*.jsonl"):
        if f.stem in ("mmlu_pro", "arc", "gsm8k", "mmlu_physics", "mmlu_philosophy",
                      "mmlu_astronomy", "mmlu_compsec", "mmlu_algebra", "mmlu_religions"):
            with open(f) as fh:
                for line in fh:
                    row = json.loads(line)
                    q = row.get("question", "")
                    a = row.get("answer", "")
                    if q and a and len(a) > 1:  # Skip single-letter answers
                        inp = f"Facts: {q[:300]}\nQuestion: What is the answer?\nAnswer:"
                        tgt = f" {a}"
                        pairs.append((inp, tgt))

    # Add "I don't know" training examples (teach the model to refuse)
    refusal_pairs = [
        ("Facts: The sky is blue.\nQuestion: What is the speed of light?\nAnswer:",
         " The provided facts do not contain information about the speed of light."),
        ("Facts: Dogs are mammals.\nQuestion: What year was Python created?\nAnswer:",
         " I don't have that information in the provided facts."),
        ("Facts: Water boils at 100 degrees Celsius.\nQuestion: Who is the president?\nAnswer:",
         " The facts provided do not mention any president."),
    ]
    # Repeat refusal examples to balance
    for _ in range(len(pairs) // 20):
        pairs.extend(refusal_pairs)

    random.shuffle(pairs)
    return pairs


def train():
    print("=== REASONING KERNEL TRAINING ===\n", flush=True)
    print(f"Device: {DEVICE}", flush=True)

    # Load GPT-2 125M
    print("Loading GPT-2 125M...", flush=True)
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {n_params:,} params ({n_params*4/1e6:.0f}MB)", flush=True)

    # Build training data
    print("\nBuilding training data...", flush=True)
    pairs = build_training_data()
    print(f"  {len(pairs):,} training pairs\n", flush=True)

    # Show samples
    for i in range(min(3, len(pairs))):
        inp, tgt = pairs[i]
        print(f"  Example {i+1}:", flush=True)
        print(f"    IN:  {inp[:80]}...", flush=True)
        print(f"    TGT: {tgt[:60]}", flush=True)
    print(flush=True)

    # Training
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    n_epochs = 3
    batch_size = 8
    max_len = 256

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs * len(pairs) // batch_size, eta_min=1e-6)

    print(f"Training for {n_epochs} epochs, batch_size={batch_size}...\n", flush=True)
    t0 = time.time()

    for epoch in range(n_epochs):
        random.shuffle(pairs)
        epoch_loss = 0
        steps = 0

        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i+batch_size]
            texts = [inp + tgt for inp, tgt in batch]

            tokens = tokenizer(texts, return_tensors='pt', truncation=True,
                             max_length=max_len, padding='max_length').to(DEVICE)

            outputs = model(input_ids=tokens['input_ids'], labels=tokens['input_ids'])
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            steps += 1

            if steps % 500 == 0:
                print(f"    step {steps}  loss={epoch_loss/steps:.4f}", flush=True)

        avg_loss = epoch_loss / steps
        elapsed = time.time() - t0
        print(f"  Epoch {epoch+1}/{n_epochs}  loss={avg_loss:.4f}  "
              f"lr={optimizer.param_groups[0]['lr']:.6f}  time={elapsed:.0f}s", flush=True)

    train_time = time.time() - t0
    print(f"\n  Training done in {train_time:.0f}s\n", flush=True)

    # Save model
    model.save_pretrained(str(OUT_DIR))
    tokenizer.save_pretrained(str(OUT_DIR))
    print(f"  Saved to {OUT_DIR}", flush=True)

    # Test
    print("\n=== TESTING TRAINED KERNEL ===\n", flush=True)
    model.eval()

    test_cases = [
        "Facts: World War II ended in 1945 with the surrender of Germany and Japan.\nQuestion: When did World War 2 end?\nAnswer:",
        "Facts: DNA is a double helix polymer made of nucleotides. DNA carries genetic information using four bases ATCG.\nQuestion: What is DNA?\nAnswer:",
        "Facts: The printing press was invented by Johannes Gutenberg around 1440.\nQuestion: Who invented the printing press?\nAnswer:",
        "Facts: Paris is the capital of France. Tokyo is the capital of Japan.\nQuestion: What is the capital of France?\nAnswer:",
        "Facts: Water freezes at zero degrees Celsius.\nQuestion: What is the speed of light?\nAnswer:",
        "Facts: The blue whale is the largest animal ever known to exist.\nQuestion: What is the largest animal?\nAnswer:",
    ]

    for test in test_cases:
        inp = tokenizer(test, return_tensors='pt', truncation=True, max_length=200).to(DEVICE)
        with torch.no_grad():
            out = model.generate(inp['input_ids'], max_new_tokens=60,
                                do_sample=False, pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(out[0][inp['input_ids'].size(1):], skip_special_tokens=True)
        q_part = test.split("Question:")[1].split("Answer:")[0].strip()
        print(f"  Q: {q_part}", flush=True)
        print(f"  A: {response.strip()[:100]}", flush=True)
        print(flush=True)

    # Save size info
    model_size = sum(os.path.getsize(os.path.join(OUT_DIR, f))
                    for f in os.listdir(OUT_DIR)) / 1e6
    print(f"\n{'='*50}", flush=True)
    print(f"KERNEL TRAINING COMPLETE", flush=True)
    print(f"{'='*50}", flush=True)
    print(f"  Model: GPT-2 125M (context-faithful fine-tuned)", flush=True)
    print(f"  Params: {n_params:,}", flush=True)
    print(f"  Size: {model_size:.0f}MB", flush=True)
    print(f"  Training pairs: {len(pairs):,}", flush=True)
    print(f"  Epochs: {n_epochs}", flush=True)
    print(f"  Final loss: {avg_loss:.4f}", flush=True)
    print(f"  Train time: {train_time:.0f}s", flush=True)
    print(f"  Output: {OUT_DIR}", flush=True)


if __name__ == "__main__":
    train()
