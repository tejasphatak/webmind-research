#!/usr/bin/env python3
"""
Synapse v2 with Sentence Transformer Router
=============================================
- Router: sentence-transformers/all-MiniLM-L6-v2 (~80MB)
  Understands language. Encodes queries into embeddings.
  Finds nearest specialist by cosine similarity.

- Generator: GPT-2 (full 12-layer) per specialist.
  Produces text output.

Router is SHARED (every device has it).
Generator is SPECIALIZED (each device fine-tunes differently).
"""

import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sentence_transformers import SentenceTransformer
import copy, time, json, argparse, numpy as np
from pathlib import Path

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class SBERTRouter:
    """Routes queries to specialists using sentence embeddings."""

    def __init__(self):
        print("  Loading sentence transformer router...", flush=True)
        self.model = SentenceTransformer('all-MiniLM-L6-v2', device=DEVICE)
        self.specialist_profiles = {}  # node_id → embedding of what it knows

    def register_specialist(self, node_id, training_prompts):
        """Build a profile embedding for this specialist from its training data."""
        embeddings = self.model.encode(training_prompts, convert_to_tensor=True)
        profile = embeddings.mean(dim=0)  # average of all training prompts
        self.specialist_profiles[node_id] = profile

    def route(self, query, top_k=3):
        """Find the top-k most relevant specialists for this query."""
        query_emb = self.model.encode([query], convert_to_tensor=True)[0]

        scores = []
        for nid, profile in self.specialist_profiles.items():
            sim = F.cosine_similarity(query_emb.unsqueeze(0),
                                      profile.unsqueeze(0)).item()
            scores.append((nid, sim))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


class SpecialistGenerator:
    """One specialist with a fine-tuned GPT-2 generator."""

    def __init__(self, node_id, model, tokenizer):
        self.id = node_id
        self.model = model.to(DEVICE)
        self.tokenizer = tokenizer
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
        self.training_prompts = []

    def train_on(self, prompt, target):
        self.model.train()
        full = prompt + target
        inp = self.tokenizer(full, return_tensors="pt",
                            truncation=True, max_length=128).to(DEVICE)
        logits = self.model(**inp).logits
        loss = F.cross_entropy(
            logits[:, :-1, :].reshape(-1, logits.size(-1)),
            inp["input_ids"][:, 1:].reshape(-1))
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.training_prompts.append(prompt)
        return loss.item()

    def generate(self, prompt, max_tokens=20):
        self.model.eval()
        inp = self.tokenizer(prompt, return_tensors="pt",
                            truncation=True, max_length=64).to(DEVICE)
        with torch.no_grad():
            out = self.model.generate(
                inp["input_ids"], max_new_tokens=max_tokens,
                do_sample=False, pad_token_id=self.tokenizer.pad_token_id)
        return self.tokenizer.decode(
            out[0][inp["input_ids"].size(1):], skip_special_tokens=True)


def run(output_dir, n_nodes=8):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"=== SYNAPSE v2 + SBERT ROUTER ({n_nodes} nodes) ===\n", flush=True)

    # Router
    router = SBERTRouter()

    # Generators
    print("  Loading base GPT-2...", flush=True)
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    base = GPT2LMHeadModel.from_pretrained("gpt2")
    base.eval()

    nodes = []
    for i in range(n_nodes):
        model = copy.deepcopy(base)
        nodes.append(SpecialistGenerator(i, model, tok))
    del base
    torch.cuda.empty_cache()
    print(f"  {n_nodes} generators ready", flush=True)

    # Topic data
    topics = {
        "geography": [
            ("The capital of France is", " Paris, the largest city in France"),
            ("The capital of Japan is", " Tokyo, a massive metropolitan area"),
            ("The largest continent is", " Asia, covering about 30% of Earth"),
            ("The longest river is the", " Nile, flowing through northeast Africa"),
            ("Mount Everest is located in", " the Himalayas between Nepal and Tibet"),
            ("Australia is both a country and a", " continent in the Southern Hemisphere"),
        ],
        "science": [
            ("Water freezes at", " zero degrees Celsius or 32 Fahrenheit"),
            ("The sun is a", " medium-sized star in our solar system"),
            ("Photosynthesis is the process by which plants", " convert sunlight into energy"),
            ("DNA contains the genetic instructions for", " all living organisms"),
            ("Gravity is the force that", " attracts objects with mass toward each other"),
            ("The speed of light is approximately", " 300,000 kilometers per second"),
        ],
        "math": [
            ("Two plus two equals", " four, a basic arithmetic fact"),
            ("Pi is approximately equal to", " 3.14159, the ratio of circumference to diameter"),
            ("The Pythagorean theorem states that", " a squared plus b squared equals c squared"),
            ("A prime number is divisible only by", " one and itself"),
            ("The square root of 144 is", " twelve"),
            ("In calculus, a derivative measures the", " rate of change of a function"),
        ],
        "tech": [
            ("Python is a popular", " programming language known for its simplicity"),
            ("HTML is used to create", " web pages and structure content on the internet"),
            ("Machine learning is a subset of", " artificial intelligence that learns from data"),
            ("Linux is an open source", " operating system used in servers and development"),
            ("JavaScript runs primarily in", " web browsers to make pages interactive"),
            ("A database stores", " organized collections of data for easy retrieval"),
        ],
        "history": [
            ("World War 2 ended in", " 1945 with the surrender of Germany and Japan"),
            ("The moon landing occurred in", " 1969 when Apollo 11 reached the lunar surface"),
            ("The Renaissance was a period of", " cultural rebirth that began in 14th century Italy"),
            ("The Industrial Revolution transformed", " manufacturing and society starting in Britain"),
            ("Democracy originated in", " ancient Athens, Greece around the 5th century BC"),
            ("The printing press was invented by", " Johannes Gutenberg around 1440"),
        ],
        "food": [
            ("Pizza originated in", " Naples, Italy as a simple flatbread dish"),
            ("Sushi is a traditional dish from", " Japan made with vinegared rice"),
            ("Chocolate is made from", " roasted and ground cacao beans"),
            ("Coffee originated in", " Ethiopia and spread through the Arab world"),
            ("Pasta is a staple food of", " Italian cuisine made from wheat flour"),
            ("Tea has been consumed for thousands of years in", " China and East Asia"),
        ],
        "animals": [
            ("The blue whale is the", " largest animal ever known to have existed"),
            ("Cheetahs are the fastest", " land animals reaching speeds of 70 mph"),
            ("Dolphins are highly intelligent", " marine mammals that communicate with clicks"),
            ("Elephants are the largest", " land animals with remarkable memory"),
            ("Bees are essential for", " pollination of many food crops worldwide"),
            ("Octopuses are known for their", " intelligence and ability to solve problems"),
        ],
        "language": [
            ("The opposite of hot is", " cold, describing low temperature"),
            ("A synonym for happy is", " joyful or content"),
            ("The past tense of run is", " ran"),
            ("An adjective describes a", " noun or pronoun in a sentence"),
            ("A metaphor is a figure of speech that", " compares two unlike things directly"),
            ("The plural of mouse is", " mice, an irregular plural form"),
        ],
    }

    # Train specialists
    topic_names = list(topics.keys())
    assignments = {}
    for i, topic in enumerate(topic_names):
        primary = i * n_nodes // len(topic_names)
        assigned = [primary % n_nodes, (primary + 1) % n_nodes]
        assignments[topic] = assigned

    print("\n  Training specialists...", flush=True)
    for epoch in range(5):
        total_loss, steps = 0, 0
        for topic, node_ids in assignments.items():
            for prompt, target in topics[topic]:
                for nid in node_ids:
                    loss = nodes[nid].train_on(prompt, target)
                    total_loss += loss
                    steps += 1
        print(f"    epoch {epoch+1}/5  loss={total_loss/steps:.4f}", flush=True)

    # Register specialists with router
    print("\n  Registering specialist profiles...", flush=True)
    for topic, node_ids in assignments.items():
        for nid in node_ids:
            router.register_specialist(nid, nodes[nid].training_prompts)

    # === TEST: Interactive queries ===
    test_queries = [
        ("The capital of France is", "geography"),
        ("Water freezes at", "science"),
        ("Two plus two equals", "math"),
        ("Python is a popular", "tech"),
        ("World War 2 ended in", "history"),
        ("Pizza originated in", "food"),
        ("The blue whale is the", "animals"),
        ("The opposite of hot is", "language"),
        # Cross-topic
        ("How does gravity affect the ocean tides", "science+geography"),
        ("What programming language is used for AI", "tech+science"),
    ]

    print(f"\n{'='*60}", flush=True)
    print("RESULTS", flush=True)
    print(f"{'='*60}\n", flush=True)

    correct, total = 0, 0
    for prompt, expected_topic in test_queries:
        t0 = time.time()

        # Route
        route_results = router.route(prompt, top_k=3)
        route_ms = (time.time() - t0) * 1000

        # Generate from best specialist
        best_nid = route_results[0][0]
        best_sim = route_results[0][1]
        response = nodes[best_nid].generate(prompt, max_tokens=15)
        total_ms = (time.time() - t0) * 1000

        # Check if reasonable
        resp_lower = response.strip().lower()
        # Simple check: does the response contain expected content?
        topic_keywords = {
            "geography": ["paris", "tokyo", "asia", "nile", "everest", "australia"],
            "science": ["zero", "star", "light", "dna", "gravity", "photo"],
            "math": ["four", "pi", "3.14", "prime", "twelve", "derivative"],
            "tech": ["programming", "language", "web", "learn", "linux", "data"],
            "history": ["1945", "1969", "renaissance", "revolution", "athens", "gutenberg"],
            "food": ["naples", "italy", "japan", "cacao", "cocoa", "ethiopia", "china"],
            "animals": ["largest", "fastest", "intelligent", "mammal", "pollinat"],
            "language": ["cold", "joyful", "ran", "noun", "compar", "mice"],
        }

        match = False
        for t in expected_topic.split("+"):
            if t in topic_keywords:
                for kw in topic_keywords[t]:
                    if kw in resp_lower:
                        match = True
                        break

        correct += match
        total += 1

        print(f"  [{'OK' if match else 'XX'}] route→node {best_nid} "
              f"(sim={best_sim:.3f}) [{route_ms:.0f}ms route, {total_ms:.0f}ms total]",
              flush=True)
        print(f"       Q: \"{prompt}\"", flush=True)
        print(f"       A: \"{response.strip()[:60]}\"", flush=True)
        print(flush=True)

    print(f"  Accuracy: {correct}/{total} = {correct/total:.0%}", flush=True)

    # Save
    with open(out / "results.json", "w") as f:
        json.dump({"accuracy": correct/total, "n_nodes": n_nodes,
                   "correct": correct, "total": total}, f, indent=2)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="/workspace/results/")
    p.add_argument("--nodes", type=int, default=8)
    run(p.parse_args().output, p.parse_args().nodes)
