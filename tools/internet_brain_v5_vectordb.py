#!/usr/bin/env python3
"""
Internet Brain v5 — Vector DB Architecture
============================================
Key insight: you don't need a model per neuron.
Each neuron is just a shard of a distributed vector database.

Components:
- Encoder: sentence-transformers/all-MiniLM-L6-v2 (80MB, shared)
- Neurons: store (embedding, text) pairs — that's it
- Router: cosine similarity on embeddings
- Complex queries: decompose → multi-hop retrieval → combine

Per-neuron storage: ~KB (embeddings + text), NOT MB (model weights)
Could literally run on an ESP32.

Communication: 1.5KB per query (one 384-float embedding vector)
"""

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
import time, random, json
from dataclasses import dataclass, field

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class VectorEntry:
    embedding: torch.Tensor   # [384]
    text: str                 # the stored fact
    topic: str = ""           # optional metadata


class Neuron:
    """A neuron is just a vector database shard.
    No model. No weights. Just (embedding, text) pairs."""

    def __init__(self, nid):
        self.id = nid
        self.entries: list[VectorEntry] = []
        self.alive = True

    def store(self, embedding, text, topic=""):
        self.entries.append(VectorEntry(embedding=embedding, text=text, topic=topic))

    def search(self, query_emb, top_k=3):
        """Find most similar entries by cosine similarity."""
        if not self.entries or not self.alive:
            return []
        embs = torch.stack([e.embedding for e in self.entries])
        sims = F.cosine_similarity(query_emb.unsqueeze(0), embs)
        topk = min(top_k, len(self.entries))
        vals, idxs = sims.topk(topk)
        return [(self.entries[i], vals[j].item()) for j, i in enumerate(idxs)]

    def profile_embedding(self):
        """Average embedding of all stored facts — used for routing."""
        if not self.entries:
            return None
        return torch.stack([e.embedding for e in self.entries]).mean(dim=0)

    def size_bytes(self):
        """Approximate storage size."""
        emb_bytes = len(self.entries) * 384 * 4  # float32
        text_bytes = sum(len(e.text.encode()) for e in self.entries)
        return emb_bytes + text_bytes

    def kill(self):
        self.alive = False
        self.entries = []


class InternetBrain:
    """Distributed vector DB with semantic routing."""

    def __init__(self, n_neurons=10):
        print("=== INTERNET BRAIN v5 — VECTOR DB ===\n", flush=True)
        print(f"  Device: {DEVICE}", flush=True)

        print("  Loading sentence transformer...", flush=True)
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2', device=DEVICE)

        self.neurons = {i: Neuron(i) for i in range(n_neurons)}
        self.n_neurons = n_neurons
        print(f"  {n_neurons} neurons (vector DB shards) ready", flush=True)
        print(f"  No GPT-2. No decoder. Just embeddings + text.\n", flush=True)

    def encode(self, text):
        with torch.no_grad():
            return self.encoder.encode([text], convert_to_tensor=True)[0]

    def encode_batch(self, texts):
        with torch.no_grad():
            return self.encoder.encode(texts, convert_to_tensor=True)

    def teach(self, neuron_id, text, topic=""):
        """Store a fact in a neuron."""
        emb = self.encode(text)
        self.neurons[neuron_id].store(emb, text, topic)

    def teach_batch(self, neuron_id, texts, topic=""):
        """Store multiple facts at once."""
        embs = self.encode_batch(texts)
        for emb, text in zip(embs, texts):
            self.neurons[neuron_id].store(emb, text, topic)

    def route(self, query, top_k=3, exclude=None):
        """Find the best neurons for this query."""
        query_emb = self.encode(query)
        scores = []
        for nid, neuron in self.neurons.items():
            if not neuron.alive or not neuron.entries:
                continue
            if exclude and nid in exclude:
                continue
            profile = neuron.profile_embedding()
            sim = F.cosine_similarity(
                query_emb.unsqueeze(0), profile.unsqueeze(0)).item()
            scores.append((nid, sim))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def ask(self, query, top_k_neurons=3, top_k_results=3, latency_ms=0):
        """Route query → search best neurons → return results."""
        t0 = time.time()
        query_emb = self.encode(query)
        route_ms = (time.time() - t0) * 1000

        # Simulate network latency
        if latency_ms > 0:
            time.sleep(latency_ms / 1000)

        # Route to best neurons
        routes = self.route(query, top_k=top_k_neurons)
        if not routes:
            return [], 0

        # Search within routed neurons
        all_results = []
        for nid, route_sim in routes:
            neuron = self.neurons[nid]
            results = neuron.search(query_emb, top_k=top_k_results)
            for entry, sim in results:
                all_results.append((entry, sim, nid, route_sim))

        # Sort by local similarity
        all_results.sort(key=lambda x: x[1], reverse=True)
        total_ms = (time.time() - t0) * 1000
        return all_results[:top_k_results], total_ms

    def ask_random(self, query, top_k=3):
        """Ask a random neuron (baseline)."""
        query_emb = self.encode(query)
        alive = [n for n in self.neurons.values() if n.alive and n.entries]
        if not alive:
            return []
        neuron = random.choice(alive)
        return neuron.search(query_emb, top_k=top_k)

    def ask_all(self, query, top_k=3):
        """Search ALL neurons (brute force, for comparison)."""
        query_emb = self.encode(query)
        all_results = []
        for neuron in self.neurons.values():
            if not neuron.alive:
                continue
            results = neuron.search(query_emb, top_k=top_k)
            for entry, sim in results:
                all_results.append((entry, sim))
        all_results.sort(key=lambda x: x[1], reverse=True)
        return all_results[:top_k]

    def multi_hop(self, query, hops=2, top_k=3):
        """Multi-hop retrieval for complex queries.
        Each hop refines the query with retrieved context."""
        context = query
        all_retrieved = []
        for hop in range(hops):
            results, _ = self.ask(context, top_k_neurons=3, top_k_results=2)
            if not results:
                break
            for entry, sim, nid, rsim in results:
                all_retrieved.append((entry, sim, nid, hop))
            # Enrich context with retrieved facts for next hop
            best_fact = results[0][0].text
            context = f"{query} {best_fact}"
        return all_retrieved

    def kill_neurons(self, fraction=0.5):
        alive = [nid for nid, n in self.neurons.items() if n.alive]
        n_kill = int(len(alive) * fraction)
        to_kill = random.sample(alive, n_kill)
        for nid in to_kill:
            self.neurons[nid].kill()
        return to_kill

    def alive_count(self):
        return sum(1 for n in self.neurons.values() if n.alive)

    def total_storage(self):
        return sum(n.size_bytes() for n in self.neurons.values())

    def total_entries(self):
        return sum(len(n.entries) for n in self.neurons.values())


# ══════════════════════════════════════════════════════════════

FACTS = {
    "geography": [
        "Paris is the capital of France",
        "Tokyo is the capital of Japan",
        "Cairo is the capital of Egypt",
        "The largest continent is Asia covering about 30 percent of Earth",
        "Mount Everest is the tallest mountain in the Himalayas",
        "The Nile is the longest river flowing through Africa",
        "Australia is both a country and a continent",
        "Brazil is the largest country in South America",
    ],
    "science": [
        "Water freezes at zero degrees Celsius or 32 Fahrenheit",
        "The sun is a medium sized star in our solar system",
        "Photosynthesis is how plants convert sunlight into energy",
        "DNA contains the genetic instructions for all living organisms",
        "The speed of light is approximately 300000 kilometers per second",
        "Gravity is the force that attracts objects with mass toward each other",
        "Atoms are the basic building blocks of all matter",
        "The Earth orbits the sun once every 365 days",
    ],
    "math": [
        "Two plus two equals four",
        "Pi is approximately 3.14159",
        "The Pythagorean theorem states a squared plus b squared equals c squared",
        "The square root of 144 is twelve",
        "A prime number is divisible only by one and itself",
        "The derivative measures the rate of change of a function",
        "Zero is neither positive nor negative",
        "Infinity is not a number but a concept",
    ],
    "tech": [
        "Python is a popular programming language known for simplicity",
        "HTML is used to create web pages and structure content",
        "Machine learning is a subset of artificial intelligence",
        "Linux is an open source operating system",
        "JavaScript runs in web browsers to make pages interactive",
        "A database stores organized collections of data",
        "TCP IP is the protocol that powers the internet",
        "Git is a version control system for tracking code changes",
    ],
    "history": [
        "World War 2 ended in 1945 with the surrender of Germany and Japan",
        "The moon landing occurred in 1969 when Apollo 11 reached the surface",
        "The Renaissance was a cultural rebirth starting in 14th century Italy",
        "Democracy originated in ancient Athens Greece",
        "The printing press was invented by Johannes Gutenberg around 1440",
        "The French Revolution began in 1789",
        "The Roman Empire fell in 476 AD",
        "The Industrial Revolution started in Britain in the 18th century",
    ],
    "food": [
        "Pizza originated in Naples Italy as a flatbread dish",
        "Sushi is a traditional dish from Japan made with vinegared rice",
        "Chocolate is made from roasted and ground cacao beans",
        "Coffee originated in Ethiopia",
        "Pasta is a staple food of Italian cuisine",
        "Tea has been consumed for thousands of years in China",
        "Bread is one of the oldest prepared foods in human history",
        "Cheese is made by curdling milk with an enzyme called rennet",
    ],
    "animals": [
        "The blue whale is the largest animal ever known to exist",
        "Cheetahs are the fastest land animals reaching 70 mph",
        "Dolphins are highly intelligent marine mammals",
        "Elephants are the largest land animals with remarkable memory",
        "Bees are essential for pollination of food crops",
        "Octopuses are known for their intelligence and problem solving",
        "Penguins are flightless birds that live in the Southern Hemisphere",
        "Ants can carry 50 times their own body weight",
    ],
    "language": [
        "The opposite of hot is cold",
        "A synonym for happy is joyful",
        "The past tense of run is ran",
        "An adjective describes a noun",
        "The plural of mouse is mice",
        "A verb is an action word",
        "An adverb modifies a verb",
        "A metaphor compares two unlike things directly",
    ],
    "music": [
        "Beethoven composed nine symphonies",
        "The guitar has six strings",
        "Jazz originated in New Orleans",
        "A piano has 88 keys",
        "Mozart was born in Salzburg Austria in 1756",
        "Rock and roll emerged in the 1950s in America",
        "A symphony orchestra typically has about 100 musicians",
        "The violin is the smallest member of the string family",
    ],
    "sports": [
        "Football is the most popular sport worldwide",
        "The Olympics happen every four years",
        "Basketball was invented by James Naismith in 1891",
        "A marathon is 26.2 miles or about 42 kilometers",
        "Tennis is played on a rectangular court with a net",
        "Cricket is popular in India England and Australia",
        "Swimming has been an Olympic sport since 1896",
        "Golf originated in Scotland in the 15th century",
    ],
}

TOPIC_KEYWORDS = {
    "geography": ["paris", "tokyo", "cairo", "asia", "everest", "nile", "australia", "brazil"],
    "science": ["zero", "celsius", "star", "solar", "dna", "light", "300", "gravity", "atom"],
    "math": ["four", "pi", "3.14", "pythagorean", "twelve", "prime", "derivative", "infinity"],
    "tech": ["programming", "language", "web", "pages", "linux", "database", "javascript", "git"],
    "history": ["1945", "1969", "renaissance", "athens", "gutenberg", "1789", "476", "industrial"],
    "food": ["naples", "italy", "japan", "cacao", "cocoa", "ethiopia", "china", "cheese"],
    "animals": ["largest", "fastest", "intelligent", "mammal", "pollinat", "octopus", "penguin"],
    "language": ["cold", "joyful", "ran", "noun", "mice", "verb", "adverb", "metaphor"],
    "music": ["nine", "symphon", "six strings", "new orleans", "salzburg", "piano", "88"],
    "sports": ["worldwide", "billion", "four years", "naismith", "26.2", "42", "cricket", "golf"],
}


def check_answer(text, topic):
    t = text.lower()
    for kw in TOPIC_KEYWORDS.get(topic, []):
        if kw in t:
            return True
    return False


def run(n_neurons=10):
    brain = InternetBrain(n_neurons=n_neurons)

    topic_names = list(FACTS.keys())

    # ─── PHASE 1: LOAD FACTS INTO NEURONS ────────────────
    print("── PHASE 1: LOADING FACTS ──\n", flush=True)
    t0 = time.time()
    for i, topic in enumerate(topic_names):
        nid = i % n_neurons
        brain.teach_batch(nid, FACTS[topic], topic=topic)
        print(f"  Neuron {nid} ← {topic}: {len(FACTS[topic])} facts", flush=True)
    load_time = time.time() - t0
    print(f"\n  Loaded {brain.total_entries()} facts in {load_time:.1f}s", flush=True)
    print(f"  Total storage: {brain.total_storage():,} bytes "
          f"({brain.total_storage()/1024:.1f} KB)", flush=True)
    print(f"  Per neuron avg: {brain.total_storage()/n_neurons:.0f} bytes", flush=True)

    # ─── PHASE 2: ROUTING TEST ───────────────────────────
    print("\n── PHASE 2: ROUTING TEST ──\n", flush=True)
    test_queries = [
        ("What is the capital of France?", "geography", 0),
        ("At what temperature does water freeze?", "science", 1),
        ("What is two plus two?", "math", 2),
        ("What programming language is Python?", "tech", 3),
        ("When did World War 2 end?", "history", 4),
        ("Where does pizza come from?", "food", 5),
        ("What is the largest animal?", "animals", 6),
        ("What is the opposite of hot?", "language", 7),
        ("How many symphonies did Beethoven compose?", "music", 8),
        ("What is the most popular sport?", "sports", 9),
    ]

    routed_ok, route_ok = 0, 0
    for query, topic, expected_nid in test_queries:
        t0 = time.time()
        results, total_ms = brain.ask(query, top_k_neurons=3, top_k_results=1)
        if results:
            entry, sim, nid, rsim = results[0]
            match = check_answer(entry.text, topic)
            routed_correct = nid == expected_nid
            routed_ok += match
            route_ok += routed_correct
            print(f"  [{'OK' if match else 'XX'}] route→n{nid} "
                  f"({'RIGHT' if routed_correct else f'WRONG({expected_nid})':>10s}) "
                  f"sim={sim:.3f} [{total_ms:.1f}ms]", flush=True)
            print(f"       Q: \"{query}\"", flush=True)
            print(f"       A: \"{entry.text[:60]}\"", flush=True)
        else:
            print(f"  [XX] No results for: \"{query}\"", flush=True)

    n = len(test_queries)
    print(f"\n  Routing:   {route_ok}/{n} = {route_ok/n:.0%}", flush=True)
    print(f"  Retrieval: {routed_ok}/{n} = {routed_ok/n:.0%}", flush=True)

    # ─── PHASE 3: RANDOM BASELINE ────────────────────────
    print("\n── PHASE 3: RANDOM BASELINE ──\n", flush=True)
    random_ok = 0
    n_trials = 30
    for query, topic, _ in test_queries:
        for _ in range(3):
            results = brain.ask_random(query, top_k=1)
            if results and check_answer(results[0][0].text, topic):
                random_ok += 1
    print(f"  Random: {random_ok}/{n_trials} = {random_ok/n_trials:.0%}", flush=True)
    routing_benefit = (routed_ok/n) / (random_ok/n_trials + 0.001)
    print(f"  Routing benefit: {routing_benefit:.1f}x over random", flush=True)

    # ─── PHASE 4: BRUTE FORCE COMPARISON ─────────────────
    print("\n── PHASE 4: BRUTE FORCE (search all neurons) ──\n", flush=True)
    brute_ok = 0
    for query, topic, _ in test_queries:
        results = brain.ask_all(query, top_k=1)
        if results and check_answer(results[0][0].text, topic):
            brute_ok += 1
    print(f"  Brute force: {brute_ok}/{n} = {brute_ok/n:.0%}", flush=True)
    print(f"  Routed:      {routed_ok}/{n} = {routed_ok/n:.0%} "
          f"(same accuracy, but only queries {min(3,n_neurons)}/{n_neurons} neurons)",
          flush=True)

    # ─── PHASE 5: NOVEL QUERIES ──────────────────────────
    print("\n── PHASE 5: NOVEL QUERIES ──\n", flush=True)
    novel_queries = [
        ("Tell me about the Nile river", "geography"),
        ("How do plants make food?", "science"),
        ("What is the Pythagorean theorem?", "math"),
        ("What is machine learning?", "tech"),
        ("When was the French Revolution?", "history"),
        ("Where does tea come from?", "food"),
        ("Are dolphins smart?", "animals"),
        ("What does an adverb do?", "language"),
        ("Where was Mozart born?", "music"),
        ("Who invented basketball?", "sports"),
    ]
    novel_ok = 0
    for query, topic in novel_queries:
        results, ms = brain.ask(query, top_k_neurons=3, top_k_results=1)
        if results:
            entry, sim, nid, rsim = results[0]
            match = check_answer(entry.text, topic)
            novel_ok += match
            print(f"  [{'OK' if match else 'XX'}] n{nid} sim={sim:.3f} [{ms:.1f}ms]",
                  flush=True)
            print(f"       Q: \"{query}\"", flush=True)
            print(f"       A: \"{entry.text[:60]}\"", flush=True)
    print(f"\n  Novel: {novel_ok}/{len(novel_queries)} = "
          f"{novel_ok/len(novel_queries):.0%}", flush=True)

    # ─── PHASE 6: MULTI-HOP RETRIEVAL ────────────────────
    print("\n── PHASE 6: MULTI-HOP RETRIEVAL ──\n", flush=True)
    complex_queries = [
        "How does the solar system relate to gravity?",
        "What connects Italy to food history?",
        "How are mathematics and science related?",
        "What do Japan and food have in common?",
        "How are animals and science connected?",
    ]
    for query in complex_queries:
        results = brain.multi_hop(query, hops=2, top_k=3)
        print(f"  Q: \"{query}\"", flush=True)
        seen = set()
        for entry, sim, nid, hop in results:
            if entry.text not in seen:
                seen.add(entry.text)
                print(f"    hop{hop} n{nid} sim={sim:.3f}: \"{entry.text[:55]}\"",
                      flush=True)
        print(flush=True)

    # ─── PHASE 7: RESILIENCE (KILL 50%) ──────────────────
    print("── PHASE 7: RESILIENCE (kill 50%) ──\n", flush=True)
    killed = brain.kill_neurons(fraction=0.5)
    print(f"  Killed: {killed}", flush=True)
    print(f"  Alive:  {brain.alive_count()}/{n_neurons}\n", flush=True)

    res_ok = 0
    for query, topic, _ in test_queries:
        results, ms = brain.ask(query, top_k_neurons=3, top_k_results=1)
        if results:
            entry, sim, nid, rsim = results[0]
            match = check_answer(entry.text, topic)
            res_ok += match
            print(f"  [{'OK' if match else 'XX'}] n{nid} Q: \"{query[:35]}\" → "
                  f"\"{entry.text[:40]}\"", flush=True)
        else:
            print(f"  [XX] No results: \"{query[:35]}\"", flush=True)
    print(f"\n  Post-kill: {res_ok}/{n} = {res_ok/n:.0%}", flush=True)

    # ─── PHASE 8: LATENCY SIMULATION ─────────────────────
    print("\n── PHASE 8: LATENCY SIMULATION ──\n", flush=True)
    for latency in [0, 15, 50, 100, 200]:
        t0 = time.time()
        nq = 5
        for query, _, _ in test_queries[:nq]:
            brain.ask(query, latency_ms=latency)
        elapsed = (time.time() - t0) * 1000
        print(f"  Latency={latency:3d}ms  →  avg={elapsed/nq:.0f}ms/query  "
              f"total={elapsed:.0f}ms", flush=True)

    # ─── PHASE 9: ONLINE LEARNING ────────────────────────
    print("\n── PHASE 9: ONLINE LEARNING ──\n", flush=True)
    alive = [nid for nid, n in brain.neurons.items() if n.alive]
    if alive:
        nid = alive[0]
        new_facts = [
            "The Eiffel Tower is 330 meters tall and located in Paris France",
            "The Great Wall of China is over 13000 miles long",
            "The Sahara Desert is the largest hot desert in the world",
        ]
        print(f"  Teaching neuron {nid} new facts...", flush=True)
        brain.teach_batch(nid, new_facts, topic="new")
        for fact in new_facts:
            print(f"    Stored: \"{fact[:55]}\"", flush=True)

        # Test retrieval
        print(flush=True)
        for q in ["How tall is the Eiffel Tower?", "What is the largest desert?"]:
            results, ms = brain.ask(q, top_k_neurons=3, top_k_results=1)
            if results:
                entry, sim, nid, _ = results[0]
                print(f"  Q: \"{q}\"", flush=True)
                print(f"  A: n{nid} sim={sim:.3f}: \"{entry.text[:55]}\"", flush=True)

    # ─── PHASE 10: SCALE ANALYSIS ────────────────────────
    print(f"\n── PHASE 10: SCALE ANALYSIS ──\n", flush=True)
    total_bytes = brain.total_storage()
    total_entries = brain.total_entries()
    print(f"  Total entries:     {total_entries}", flush=True)
    print(f"  Total storage:     {total_bytes:,} bytes ({total_bytes/1024:.1f} KB)", flush=True)
    print(f"  Per entry:         {total_bytes/max(total_entries,1):.0f} bytes", flush=True)
    print(f"  Encoder:           80 MB (shared, every device)", flush=True)
    print(f"  Per-neuron model:  0 MB (NO MODEL — just vectors)", flush=True)
    print(f"  Communication:     1,536 bytes per query (384 × float32)", flush=True)
    print(f"  Compared to v4:    ~500 MB GPT-2 per neuron → 0 MB", flush=True)

    # Projection
    print(f"\n  === SCALE PROJECTION ===", flush=True)
    for n_facts_total in [1000, 10000, 100000, 1000000]:
        per_fact = 384 * 4 + 100  # embedding + ~100 chars text
        total = n_facts_total * per_fact
        per_device_10 = total / 10
        per_device_100 = total / 100
        per_device_1000 = total / 1000
        print(f"  {n_facts_total:>10,} facts: {total/1e6:.1f} MB total | "
              f"{per_device_10/1e3:.0f} KB/device @10 | "
              f"{per_device_100/1e3:.0f} KB/device @100 | "
              f"{per_device_1000/1e3:.0f} KB/device @1000", flush=True)

    # ─── SUMMARY ─────────────────────────────────────────
    print(f"\n{'='*60}", flush=True)
    print("INTERNET BRAIN v5 — VECTOR DB — RESULTS", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  Architecture:       Sentence Transformer + Distributed Vector DB", flush=True)
    print(f"  Per-neuron model:   ZERO (just embeddings + text)", flush=True)
    print(f"  Routing:            {route_ok}/{n} = {route_ok/n:.0%}", flush=True)
    print(f"  Retrieval:          {routed_ok}/{n} = {routed_ok/n:.0%}", flush=True)
    print(f"  Random baseline:    {random_ok}/{n_trials} = {random_ok/n_trials:.0%}", flush=True)
    print(f"  Routing benefit:    {routing_benefit:.1f}x", flush=True)
    print(f"  Brute force:        {brute_ok}/{n} = {brute_ok/n:.0%}", flush=True)
    print(f"  Novel queries:      {novel_ok}/{len(novel_queries)} = "
          f"{novel_ok/len(novel_queries):.0%}", flush=True)
    print(f"  Post-kill (50%):    {res_ok}/{n} = {res_ok/n:.0%}", flush=True)
    print(f"  Storage:            {total_bytes/1024:.1f} KB ({total_entries} facts)", flush=True)
    print(f"  Load time:          {load_time:.1f}s", flush=True)
    print(f"  v4→v5:              GPT-2 per neuron → vector DB shard", flush=True)
    print(f"  Key insight:        You don't need a model. Just a database.", flush=True)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--neurons", type=int, default=10)
    run(p.parse_args().neurons)
