#!/usr/bin/env python3
"""
Synapse v2 Interactive Demo — 100MB models
===========================================
Loads as many 4-layer GPT-2 specialists as GPU can hold.
Then waits for prompts on stdin.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import copy, time, sys

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

topics = {
    "geography": [
        ("The capital of France is", " Paris"), ("The capital of Japan is", " Tokyo"),
        ("The capital of Brazil is", " Brasilia"), ("The capital of Egypt is", " Cairo"),
        ("The largest continent is", " Asia"), ("The longest river is the", " Nile"),
        ("The highest mountain is", " Everest"), ("The smallest country is", " Vatican"),
    ],
    "science": [
        ("Water freezes at", " zero degrees"), ("The sun is a", " star"),
        ("Plants make food through", " photosynthesis"), ("DNA stands for", " deoxyribonucleic acid"),
        ("Gravity was described by", " Newton"), ("Light travels at", " 300000 km per second"),
        ("Atoms are made of", " protons neutrons and electrons"), ("The periodic table was created by", " Mendeleev"),
    ],
    "math": [
        ("Two plus two equals", " four"), ("Ten minus three equals", " seven"),
        ("Three times four equals", " twelve"), ("The square root of nine is", " three"),
        ("Pi is approximately", " 3.14"), ("A triangle has", " three sides"),
        ("The Pythagorean theorem relates", " the sides of a right triangle"), ("Zero factorial equals", " one"),
    ],
    "language": [
        ("The opposite of hot is", " cold"), ("The opposite of big is", " small"),
        ("The past tense of go is", " went"), ("The plural of child is", " children"),
        ("A synonym for happy is", " joyful"), ("The opposite of fast is", " slow"),
        ("An antonym of love is", " hate"), ("The comparative of good is", " better"),
    ],
    "animals": [
        ("The largest animal is the", " blue whale"), ("The fastest land animal is the", " cheetah"),
        ("Penguins live in the", " Antarctic"), ("Dolphins are", " mammals"),
        ("Bees make", " honey"), ("The tallest animal is the", " giraffe"),
        ("Octopuses have", " eight arms"), ("Elephants are the largest", " land animals"),
    ],
    "food": [
        ("Pizza originated in", " Italy"), ("Sushi is from", " Japan"),
        ("Chocolate is made from", " cocoa"), ("Coffee beans come from", " coffee plants"),
        ("Bread is made from", " flour"), ("Wine is made from", " grapes"),
        ("Cheese is made from", " milk"), ("Tea originated in", " China"),
    ],
    "tech": [
        ("HTML stands for", " HyperText Markup Language"), ("Python is a programming", " language"),
        ("CPU stands for", " Central Processing Unit"), ("RAM stands for", " Random Access Memory"),
        ("The first computer was called", " ENIAC"), ("Linux was created by", " Linus Torvalds"),
        ("JavaScript runs in the", " browser"), ("SQL is used for", " databases"),
    ],
    "history": [
        ("World War 2 ended in", " 1945"), ("The moon landing was in", " 1969"),
        ("The Berlin Wall fell in", " 1989"), ("The Renaissance began in", " Italy"),
        ("The printing press was invented by", " Gutenberg"), ("The French Revolution began in", " 1789"),
        ("The Roman Empire fell in", " 476 AD"), ("Democracy originated in", " ancient Greece"),
    ],
}

def main():
    print("Loading base model...", flush=True)
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    base = GPT2LMHeadModel.from_pretrained("gpt2")
    base.eval()

    # How many 4-layer models can we fit?
    # Each ~100MB on GPU. H100 has 94GB. Leave 10GB for overhead.
    # ~84GB / 100MB = ~840 models max. But let's be practical: 64 models.
    n_models = 64
    print(f"Creating {n_models} specialist models (4-layer each)...", flush=True)

    nodes = []
    topic_names = list(topics.keys())
    for i in range(n_models):
        small = copy.deepcopy(base)
        small.transformer.h = small.transformer.h[:4]
        small.config.n_layer = 4
        small = small.to(DEVICE)
        nodes.append(small)
    del base
    torch.cuda.empty_cache()
    print(f"  {n_models} models loaded on {DEVICE}", flush=True)

    # Train specialists
    print("Training specialists...", flush=True)
    for epoch in range(5):
        total_loss, steps = 0, 0
        for i, (topic, examples) in enumerate(topics.items()):
            # Assign each topic to n_models/n_topics nodes
            nodes_per_topic = max(n_models // len(topic_names), 2)
            start = i * nodes_per_topic
            assigned = list(range(start, min(start + nodes_per_topic, n_models)))

            for prompt, target in examples:
                inp = tok(prompt + target, return_tensors="pt",
                         truncation=True, max_length=64).to(DEVICE)
                for nid in assigned:
                    nodes[nid].train()
                    opt = torch.optim.SGD(nodes[nid].parameters(), lr=1e-3)
                    logits = nodes[nid](**inp).logits
                    loss = F.cross_entropy(
                        logits[:, :-1, :].reshape(-1, logits.size(-1)),
                        inp["input_ids"][:, 1:].reshape(-1))
                    opt.zero_grad(); loss.backward(); opt.step()
                    total_loss += loss.item(); steps += 1

        print(f"  epoch {epoch+1}/5  loss={total_loss/steps:.4f}", flush=True)

    # Set all to eval
    for n in nodes:
        n.eval()

    print(f"\n{'='*60}", flush=True)
    print(f"SYNAPSE v2 INTERACTIVE DEMO", flush=True)
    print(f"{n_models} specialists ready. Type a prompt, get routed response.", flush=True)
    print(f"Type 'quit' to exit.", flush=True)
    print(f"{'='*60}\n", flush=True)

    while True:
        try:
            prompt = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not prompt or prompt.lower() == "quit":
            break

        t0 = time.time()

        # Route: each node generates, pick most confident
        candidates = []
        for i, node in enumerate(nodes):
            with torch.no_grad():
                inp = tok(prompt, return_tensors="pt",
                         truncation=True, max_length=64).to(DEVICE)
                out = node.generate(inp["input_ids"], max_new_tokens=15,
                                   do_sample=False, pad_token_id=tok.pad_token_id)
                resp = tok.decode(out[0][inp["input_ids"].size(1):],
                                skip_special_tokens=True)
                if resp.strip():
                    # Score confidence
                    full = tok(prompt + resp, return_tensors="pt",
                              truncation=True, max_length=128).to(DEVICE)
                    logits = node(**full).logits
                    plen = inp["input_ids"].size(1)
                    loss = F.cross_entropy(
                        logits[0, plen-1:-1], full["input_ids"][0, plen:]).item()
                    candidates.append((i, resp, -loss))

        total_ms = (time.time() - t0) * 1000

        if candidates:
            candidates.sort(key=lambda x: x[2], reverse=True)
            best = candidates[0]
            # Show top 3
            print(f"\n  Best (node {best[0]:2d}): {best[1].strip()}")
            if len(candidates) > 1:
                print(f"  2nd  (node {candidates[1][0]:2d}): {candidates[1][1].strip()[:50]}")
            if len(candidates) > 2:
                print(f"  3rd  (node {candidates[2][0]:2d}): {candidates[2][1].strip()[:50]}")
            print(f"  [{total_ms:.0f}ms, {len(candidates)} responded]\n")
        else:
            print(f"\n  No specialist responded. [{total_ms:.0f}ms]\n")


if __name__ == "__main__":
    main()
