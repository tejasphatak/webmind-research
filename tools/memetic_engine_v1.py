#!/usr/bin/env python3
"""
Memetic Evolution Engine — v1 proof of concept
================================================
Each "browser" has:
1. A tiny generator model (Meme) that EVOLVES via fitness
2. A local knowledge store (vector DB of examples)

Evolution: good memes reproduce, bad memes die.
No gradient descent. Pure natural selection.

Plus: distributed retrieval across all browsers' knowledge stores.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import time
import copy
from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer

DEVICE = "cpu"
N_BROWSERS = 20
GENERATIONS = 50
QUERIES_PER_GEN = 100


class TinyMeme(nn.Module):
    """A tiny generator model — the 'meme' that evolves."""

    def __init__(self, vocab_size=128, dim=64, n_layers=2, n_heads=2):
        super().__init__()
        config = GPT2Config(
            vocab_size=vocab_size,
            n_positions=32,
            n_embd=dim,
            n_layer=n_layers,
            n_head=n_heads,
            n_inner=dim * 4,
        )
        self.model = GPT2LMHeadModel(config)
        self.fitness = 0.0
        self.age = 0
        self.generation = 0

    def generate_logits(self, input_ids):
        return self.model(input_ids).logits

    def mutate(self, mutation_rate=0.05):
        """Random weight perturbation — like biological mutation."""
        with torch.no_grad():
            for p in self.model.parameters():
                mask = torch.rand_like(p) < mutation_rate
                noise = torch.randn_like(p) * 0.05
                p.data += mask * noise

    def clone(self):
        """Reproduce — create a copy with slight mutation."""
        child = TinyMeme.__new__(TinyMeme)
        nn.Module.__init__(child)
        child.model = copy.deepcopy(self.model)
        child.fitness = 0.0
        child.age = 0
        child.generation = self.generation + 1
        child.mutate()
        return child


class KnowledgeStore:
    """Local vector DB of (query, answer) pairs — the browser's expertise."""

    def __init__(self):
        self.examples = []  # list of (input_tokens, output_tokens)

    def add(self, inp, out):
        self.examples.append((inp, out))
        if len(self.examples) > 100:  # cap memory
            self.examples.pop(0)

    def retrieve(self, query_tokens, k=3):
        """Find k most similar examples (simple token overlap)."""
        if not self.examples:
            return []
        query_set = set(query_tokens)
        scored = []
        for inp, out in self.examples:
            overlap = len(query_set & set(inp))
            scored.append((overlap, inp, out))
        scored.sort(reverse=True)
        return [(inp, out) for _, inp, out in scored[:k]]


class Browser:
    """One browser — has a meme (generator) and knowledge store."""

    def __init__(self, browser_id, meme=None):
        self.id = browser_id
        self.meme = meme or TinyMeme()
        self.knowledge = KnowledgeStore()
        self.online = True

    def respond(self, input_tokens):
        """Generate response using local meme."""
        input_ids = torch.tensor([input_tokens])
        with torch.no_grad():
            logits = self.meme.generate_logits(input_ids)
            preds = logits[0, -1, :].argmax().item()
        return preds

    def evaluate_fitness(self, input_tokens, target_tokens):
        """How good is this meme at predicting the right answer?"""
        input_ids = torch.tensor([input_tokens])
        target_ids = torch.tensor([target_tokens])
        with torch.no_grad():
            logits = self.meme.generate_logits(input_ids)
            S = min(logits.size(1), target_ids.size(1))
            loss = F.cross_entropy(
                logits[:, :S, :].reshape(-1, 128),
                target_ids[:, :S].reshape(-1)).item()
        # Fitness = accuracy + inverse loss
        # Accuracy is the primary signal (Gemini Pro's fix for mode collapse)
        preds = logits[:, :S, :].argmax(dim=-1)
        n_correct = (preds == target_ids[:, :S]).sum().item()
        return n_correct * 10.0 + max(0, 5.0 - loss)


class MemeticEngine:
    """The evolution engine — manages a population of browsers."""

    def __init__(self, n_browsers=N_BROWSERS):
        self.browsers = [Browser(i) for i in range(n_browsers)]

    def distributed_retrieve(self, query_tokens, n_peers=5):
        """Query random peers' knowledge stores."""
        peers = random.sample(self.browsers, min(n_peers, len(self.browsers)))
        all_examples = []
        for peer in peers:
            if peer.online:
                examples = peer.knowledge.retrieve(query_tokens)
                all_examples.extend(examples)
        return all_examples

    def evolve(self, train_pairs, generations=GENERATIONS,
               queries_per_gen=QUERIES_PER_GEN):
        """Run evolution: evaluate → select → reproduce → mutate."""

        print(f"  Population: {len(self.browsers)} browsers", flush=True)
        print(f"  Generations: {generations}", flush=True)
        print(f"  Queries/gen: {queries_per_gen}\n", flush=True)

        t0 = time.time()

        for gen in range(generations):
            # Evaluate fitness across random queries
            for browser in self.browsers:
                browser.meme.fitness = 0.0

            for _ in range(queries_per_gen):
                inp, target = random.choice(train_pairs)

                for browser in self.browsers:
                    if browser.online:
                        f = browser.evaluate_fitness(inp, target)
                        browser.meme.fitness += f

                        # Add to knowledge store (learning from queries)
                        browser.knowledge.add(inp, target)

            # Sort by fitness
            self.browsers.sort(key=lambda b: b.meme.fitness, reverse=True)

            # Natural selection
            n = len(self.browsers)
            top_half = self.browsers[:n // 2]
            bottom_half = self.browsers[n // 2:]

            # Top half reproduces, bottom half gets replaced
            new_pop = list(top_half)
            for i, dead_browser in enumerate(bottom_half):
                parent = top_half[i % len(top_half)]
                child_meme = parent.meme.clone()
                dead_browser.meme = child_meme
                # Keep the knowledge store (unique data survives)
                new_pop.append(dead_browser)

            self.browsers = new_pop

            # Age all memes
            for b in self.browsers:
                b.meme.age += 1

            if (gen + 1) % 5 == 0:
                best = max(b.meme.fitness for b in self.browsers)
                avg = np.mean([b.meme.fitness for b in self.browsers])
                worst = min(b.meme.fitness for b in self.browsers)

                # Quick accuracy check
                correct = 0
                total = len(train_pairs)
                best_browser = self.browsers[0]
                for inp, target in train_pairs:
                    pred = best_browser.respond(inp)
                    if len(target) > 0 and pred == target[-1]:
                        correct += 1

                elapsed = time.time() - t0
                print(f"    gen {gen+1}/{generations}  "
                      f"fitness: best={best:.1f} avg={avg:.1f} worst={worst:.1f}  "
                      f"acc={correct}/{total}  "
                      f"elapsed={elapsed:.0f}s", flush=True)

        return self.browsers[0]  # best meme


def run():
    print("=== MEMETIC EVOLUTION ENGINE ===\n", flush=True)

    # Training data: character patterns
    train_pairs = []
    for i in range(0, 26, 2):
        inp = [65 + i, 65 + i + 1]
        out = [65 + i + 2, 65 + i + 3]
        if max(out) < 128:
            train_pairs.append((inp, out))

    engine = MemeticEngine(N_BROWSERS)
    best = engine.evolve(train_pairs, GENERATIONS, QUERIES_PER_GEN)

    # Final eval with best meme
    print("\n=== EVAL (best meme) ===", flush=True)
    correct, total = 0, len(train_pairs)
    for inp, target in train_pairs:
        pred = best.respond(inp)
        target_last = target[-1] if target else -1
        match = pred == target_last
        correct += match
        in_s = ''.join(chr(c) for c in inp)
        tgt_s = chr(target_last) if 32 <= target_last < 128 else '?'
        pred_s = chr(pred) if 32 <= pred < 128 else '?'
        print(f"  {'OK' if match else 'XX'} {in_s} → {tgt_s}  pred:{pred_s}",
              flush=True)

    print(f"\n  Best meme accuracy: {correct}/{total} = {correct/total:.0%}",
          flush=True)
    print(f"  Best meme generation: {best.meme.generation}", flush=True)
    print(f"  Best meme age: {best.meme.age}", flush=True)

    # Collaborative: use ALL browsers' best guesses (majority vote)
    print("\n=== COLLABORATIVE (majority vote across all browsers) ===",
          flush=True)
    correct_collab = 0
    for inp, target in train_pairs:
        votes = {}
        for browser in engine.browsers:
            pred = browser.respond(inp)
            votes[pred] = votes.get(pred, 0) + 1
        winner = max(votes, key=votes.get)
        target_last = target[-1]
        match = winner == target_last
        correct_collab += match
        in_s = ''.join(chr(c) for c in inp)
        tgt_s = chr(target_last)
        pred_s = chr(winner) if 32 <= winner < 128 else '?'
        print(f"  {'OK' if match else 'XX'} {in_s} → {tgt_s}  "
              f"vote:{pred_s} ({votes.get(winner,0)}/{N_BROWSERS})", flush=True)

    print(f"\n  Collaborative accuracy: {correct_collab}/{total} "
          f"= {correct_collab/total:.0%}", flush=True)

    # Resilience: kill 25% of browsers
    print("\n=== RESILIENCE: Kill 5 of 20 browsers ===", flush=True)
    for b in engine.browsers[-5:]:
        b.online = False

    correct_after = 0
    for inp, target in train_pairs:
        votes = {}
        for browser in engine.browsers:
            if browser.online:
                pred = browser.respond(inp)
                votes[pred] = votes.get(pred, 0) + 1
        if votes:
            winner = max(votes, key=votes.get)
            if winner == target[-1]:
                correct_after += 1

    print(f"  Before: {correct_collab}/{total} = {correct_collab/total:.0%}",
          flush=True)
    print(f"  After:  {correct_after}/{total} = {correct_after/total:.0%}",
          flush=True)


if __name__ == "__main__":
    run()
