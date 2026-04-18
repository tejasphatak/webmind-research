#!/usr/bin/env python3
"""
Internet Brain v1 — Learn on the fly, route to specialists
============================================================
- Sentence transformer: routing cortex (understands language)
- N blank neurons: learn from user interactions
- No pretrained knowledge — starts empty
- Train on the fly → ask questions → see it learn

Simulates 10 "users" each teaching different topics, then tests routing.
"""

import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer
from sentence_transformers import SentenceTransformer
import time, random, numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_NEURONS = 10


class RoutingCortex:
    """Sentence transformer that learns which neuron knows what."""

    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2', device=DEVICE)
        # Each neuron has a profile: average embedding of what it's been taught
        self.profiles = {}  # neuron_id → list of embeddings

    def update_profile(self, neuron_id, text):
        emb = self.model.encode([text], convert_to_tensor=True)[0]
        if neuron_id not in self.profiles:
            self.profiles[neuron_id] = []
        self.profiles[neuron_id].append(emb)

    def route(self, query, top_k=3):
        if not self.profiles:
            return list(range(min(top_k, N_NEURONS)))

        query_emb = self.model.encode([query], convert_to_tensor=True)[0]
        scores = []
        for nid, embs in self.profiles.items():
            profile = torch.stack(embs).mean(dim=0)
            sim = F.cosine_similarity(query_emb.unsqueeze(0),
                                      profile.unsqueeze(0)).item()
            scores.append((nid, sim))
        scores.sort(key=lambda x: x[1], reverse=True)
        return [s[0] for s in scores[:top_k]]


class Neuron:
    """A dumb neuron. Blank model. Learns from what it's told."""

    def __init__(self, nid):
        self.id = nid
        config = GPT2Config(vocab_size=50257, n_positions=128,
                           n_embd=256, n_layer=4, n_head=4, n_inner=1024)
        self.model = GPT2LMHeadModel(config).to(DEVICE)  # RANDOM — knows nothing
        self.tok = GPT2Tokenizer.from_pretrained('gpt2')
        self.tok.pad_token = self.tok.eos_token
        self.opt = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.n_learned = 0

    def teach(self, text, n_repeats=30):
        """Learn a fact by training on it."""
        self.model.train()
        for _ in range(n_repeats):
            inp = self.tok(text, return_tensors='pt',
                          truncation=True, max_length=64).to(DEVICE)
            logits = self.model(**inp).logits
            loss = F.cross_entropy(
                logits[:, :-1, :].reshape(-1, logits.size(-1)),
                inp['input_ids'][:, 1:].reshape(-1))
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
        self.n_learned += 1
        return loss.item()

    def ask(self, prompt, max_tokens=20):
        """Generate response."""
        self.model.eval()
        inp = self.tok(prompt, return_tensors='pt',
                      truncation=True, max_length=64).to(DEVICE)
        with torch.no_grad():
            out = self.model.generate(
                inp['input_ids'], max_new_tokens=max_tokens,
                do_sample=False, pad_token_id=self.tok.pad_token_id)
        return self.tok.decode(out[0][inp['input_ids'].size(1):],
                              skip_special_tokens=True)


def run():
    print("=== INTERNET BRAIN v1 ===\n", flush=True)

    cortex = RoutingCortex()
    neurons = [Neuron(i) for i in range(N_NEURONS)]
    print(f"  {N_NEURONS} blank neurons + routing cortex ready", flush=True)

    # Simulate 10 users each teaching different things
    user_topics = {
        0: [  # Geography user
            "Paris is the capital of France",
            "Tokyo is the capital of Japan",
            "Cairo is the capital of Egypt",
        ],
        1: [  # Science user
            "Water freezes at zero degrees Celsius",
            "The sun is a star in our solar system",
            "Gravity pulls objects toward each other",
        ],
        2: [  # Math user
            "Two plus two equals four",
            "Pi is approximately 3.14",
            "The square root of nine is three",
        ],
        3: [  # Tech user
            "Python is a programming language",
            "HTML is used for web pages",
            "Linux is an operating system",
        ],
        4: [  # History user
            "World War 2 ended in 1945",
            "The moon landing was in 1969",
            "The printing press was invented by Gutenberg",
        ],
        5: [  # Food user
            "Pizza comes from Italy",
            "Sushi is from Japan",
            "Chocolate is made from cocoa beans",
        ],
        6: [  # Animals user
            "The blue whale is the largest animal",
            "Cheetahs are the fastest land animals",
            "Dolphins are intelligent mammals",
        ],
        7: [  # Language user
            "The opposite of hot is cold",
            "The past tense of go is went",
            "The plural of child is children",
        ],
        8: [  # Music user
            "Beethoven composed nine symphonies",
            "The guitar has six strings",
            "Jazz originated in New Orleans",
        ],
        9: [  # Sports user
            "Football is the most popular sport worldwide",
            "The Olympics happen every four years",
            "Basketball was invented by James Naismith",
        ],
    }

    # Phase 1: TEACHING — each user teaches their neuron
    print("\n--- PHASE 1: TEACHING ---\n", flush=True)
    t0 = time.time()
    for user_id, facts in user_topics.items():
        neuron = neurons[user_id]
        for fact in facts:
            loss = neuron.teach(fact)
            cortex.update_profile(user_id, fact)
        print(f"  User {user_id} taught neuron {user_id}: "
              f"{len(facts)} facts (loss={loss:.4f})", flush=True)

    teach_time = time.time() - t0
    print(f"\n  Teaching done in {teach_time:.0f}s", flush=True)

    # Phase 2: ASKING — test if routing finds the right neuron
    print("\n--- PHASE 2: ASKING ---\n", flush=True)
    test_queries = [
        ("What is the capital of France?", 0, "Paris"),
        ("At what temperature does water freeze?", 1, "zero"),
        ("What is two plus two?", 2, "four"),
        ("What programming language is Python?", 3, "Python"),
        ("When did World War 2 end?", 4, "1945"),
        ("Where does pizza come from?", 5, "Italy"),
        ("What is the largest animal?", 6, "whale"),
        ("What is the opposite of hot?", 7, "cold"),
        ("How many symphonies did Beethoven compose?", 8, "nine"),
        ("What is the most popular sport?", 9, "football"),
    ]

    correct = 0
    for query, expected_neuron, expected_word in test_queries:
        t0 = time.time()

        # Route
        routed = cortex.route(query, top_k=3)
        route_ms = (time.time() - t0) * 1000

        # Ask best neuron
        best_nid = routed[0]
        response = neurons[best_nid].ask(query)
        total_ms = (time.time() - t0) * 1000

        # Check
        routed_correct = best_nid == expected_neuron
        answer_has_keyword = expected_word.lower() in response.lower()
        match = routed_correct and answer_has_keyword

        correct += match
        status = "OK" if match else "XX"
        route_status = "RIGHT" if routed_correct else f"WRONG(got {best_nid})"

        print(f"  [{status}] route={route_status:12s} [{route_ms:.0f}ms route, "
              f"{total_ms:.0f}ms total]", flush=True)
        print(f"       Q: {query}", flush=True)
        print(f"       A: {response.strip()[:60]}", flush=True)
        print(flush=True)

    print(f"  Accuracy: {correct}/{len(test_queries)} = "
          f"{correct/len(test_queries):.0%}", flush=True)

    # Phase 3: CROSS-TOPIC — ask questions the teaching didn't cover
    print("\n--- PHASE 3: CROSS-TOPIC (never taught) ---\n", flush=True)
    cross_queries = [
        "What country is Tokyo in?",
        "Is the sun a planet?",
        "What is three times three?",
        "What language is HTML?",
        "Who invented the printing press?",
    ]
    for query in cross_queries:
        routed = cortex.route(query, top_k=1)
        response = neurons[routed[0]].ask(query)
        print(f"  route→neuron {routed[0]}  Q: {query}", flush=True)
        print(f"                     A: {response.strip()[:60]}", flush=True)
        print(flush=True)


if __name__ == "__main__":
    run()
