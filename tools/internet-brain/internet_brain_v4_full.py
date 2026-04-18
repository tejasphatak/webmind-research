#!/usr/bin/env python3
"""
Internet Brain v4 — Full System
================================
Combines all proven components:
1. SBERT Router (10/10 perfect routing) — cosine similarity on 384-dim embeddings
2. GPT-2 Specialist Generators (81% accuracy) — fine-tuned per domain
3. Sentence Autoencoder Decoder (100% recall) — embedding → text reconstruction

Architecture:
- Each neuron = independent GPT-2 instance (separate device in production)
- Router = shared sentence transformer (every device has it, 80MB)
- Decoder = shared trained transformer (every device has it, 196MB)
- Communication = 1.5KB per query (one 384-float embedding vector)

Tests:
- Specialist routing accuracy
- Generation quality (specialist vs random vs decoder)
- Resilience (kill neurons, measure degradation)
- Latency simulation (add network delay between neurons)
- Online learning (teach new facts after bootstrap)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import copy, time, random, json, argparse
from pathlib import Path

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMB_DIM = 384
VOCAB_SIZE = 50257
MAX_LEN = 32


# ══════════════════════════════════════════════════════════════
# Component 1: Sentence Autoencoder Decoder
# ══════════════════════════════════════════════════════════════

class TextDecoder(nn.Module):
    """Reverse sentence transformer: embedding → text."""

    def __init__(self, emb_dim=EMB_DIM, vocab_size=VOCAB_SIZE,
                 hidden=512, n_layers=4, max_len=MAX_LEN):
        super().__init__()
        self.max_len = max_len
        self.hidden = hidden
        self.emb_to_hidden = nn.Linear(emb_dim, hidden * max_len)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden, nhead=8, dim_feedforward=hidden*4, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.to_vocab = nn.Linear(hidden, vocab_size)
        self.pos_emb = nn.Embedding(max_len, hidden)

    def forward(self, embedding, target_ids=None):
        B = embedding.size(0)
        memory = self.emb_to_hidden(embedding).view(B, self.max_len, self.hidden)
        S = target_ids.size(1) if target_ids is not None else self.max_len
        pos = torch.arange(S, device=embedding.device)
        tgt = self.pos_emb(pos).unsqueeze(0).expand(B, -1, -1)
        mask = nn.Transformer.generate_square_subsequent_mask(S).to(embedding.device)
        out = self.decoder(tgt, memory[:, :S, :], tgt_mask=mask)
        return self.to_vocab(out)

    def generate(self, embedding, tokenizer, max_tokens=20):
        self.eval()
        B = embedding.size(0)
        tokens = []
        for t in range(max_tokens):
            S = t + 1
            pos = torch.arange(S, device=embedding.device)
            tgt = self.pos_emb(pos).unsqueeze(0).expand(B, -1, -1)
            memory = self.emb_to_hidden(embedding).view(B, self.max_len, self.hidden)
            mask = nn.Transformer.generate_square_subsequent_mask(S).to(embedding.device)
            out = self.decoder(tgt, memory[:, :S, :], tgt_mask=mask)
            logits = self.to_vocab(out[:, -1, :])
            next_token = logits.argmax(dim=-1).item()
            if next_token == tokenizer.eos_token_id:
                break
            tokens.append(next_token)
        return tokenizer.decode(tokens, skip_special_tokens=True)


# ══════════════════════════════════════════════════════════════
# Component 2: SBERT Router
# ══════════════════════════════════════════════════════════════

class Router:
    """Routes queries to the best specialist using sentence embeddings."""

    def __init__(self, encoder):
        self.encoder = encoder
        self.profiles = {}  # neuron_id → list of embeddings

    def update_profile(self, neuron_id, text):
        emb = self.encoder.encode([text], convert_to_tensor=True)[0]
        if neuron_id not in self.profiles:
            self.profiles[neuron_id] = []
        self.profiles[neuron_id].append(emb)

    def route(self, query, top_k=3, exclude=None):
        """Find top-k specialists. Returns [(neuron_id, similarity), ...]"""
        if not self.profiles:
            return []
        query_emb = self.encoder.encode([query], convert_to_tensor=True)[0]
        scores = []
        for nid, embs in self.profiles.items():
            if exclude and nid in exclude:
                continue
            profile = torch.stack(embs).mean(dim=0)
            sim = F.cosine_similarity(
                query_emb.unsqueeze(0), profile.unsqueeze(0)).item()
            scores.append((nid, sim))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


# ══════════════════════════════════════════════════════════════
# Component 3: Specialist Neuron (GPT-2)
# ══════════════════════════════════════════════════════════════

class Neuron:
    """One specialist neuron. Independent GPT-2 instance."""

    def __init__(self, nid, base_model, tokenizer):
        self.id = nid
        self.model = copy.deepcopy(base_model).to(DEVICE)
        self.tokenizer = tokenizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-5)
        self.facts_learned = 0
        self.alive = True

    def train_on(self, prompt, target):
        if not self.alive:
            return float('inf')
        self.model.train()
        full = prompt + target
        inp = self.tokenizer(full, return_tensors='pt',
                            truncation=True, max_length=128).to(DEVICE)
        logits = self.model(**inp).logits
        loss = F.cross_entropy(
            logits[:, :-1, :].reshape(-1, logits.size(-1)),
            inp['input_ids'][:, 1:].reshape(-1))
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.facts_learned += 1
        return loss.item()

    def generate(self, prompt, max_tokens=20):
        if not self.alive:
            return "[NEURON DEAD]"
        self.model.eval()
        inp = self.tokenizer(prompt, return_tensors='pt',
                            truncation=True, max_length=64).to(DEVICE)
        with torch.no_grad():
            out = self.model.generate(
                inp['input_ids'], max_new_tokens=max_tokens,
                do_sample=False, pad_token_id=self.tokenizer.pad_token_id)
        return self.tokenizer.decode(
            out[0][inp['input_ids'].size(1):], skip_special_tokens=True)

    def kill(self):
        self.alive = False
        del self.model
        torch.cuda.empty_cache()

    def param_count(self):
        if not self.alive:
            return 0
        return sum(p.numel() for p in self.model.parameters())


# ══════════════════════════════════════════════════════════════
# Full Internet Brain
# ══════════════════════════════════════════════════════════════

class InternetBrain:
    """The complete system: router + specialists + decoder."""

    def __init__(self, n_neurons=10):
        print("=== INTERNET BRAIN v4 — FULL SYSTEM ===\n", flush=True)
        print(f"  Device: {DEVICE}", flush=True)

        # Shared encoder (every device has this)
        print("  Loading sentence transformer (router)...", flush=True)
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2', device=DEVICE)

        # Shared decoder (every device has this)
        print("  Creating text decoder...", flush=True)
        self.decoder = TextDecoder().to(DEVICE)
        self.dec_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.dec_tokenizer.pad_token = self.dec_tokenizer.eos_token
        self.dec_optimizer = torch.optim.Adam(self.decoder.parameters(), lr=1e-3)

        # Router
        self.router = Router(self.encoder)

        # Specialist neurons (each = independent GPT-2)
        print(f"  Creating {n_neurons} specialist neurons (GPT-2)...", flush=True)
        base = GPT2LMHeadModel.from_pretrained('gpt2')
        base.eval()
        tok = GPT2Tokenizer.from_pretrained('gpt2')
        tok.pad_token = tok.eos_token

        self.neurons = {}
        for i in range(n_neurons):
            self.neurons[i] = Neuron(i, base, tok)
        del base
        torch.cuda.empty_cache()

        self.gen_tokenizer = tok
        self.n_neurons = n_neurons

        # Decoder knowledge for interleaved training
        self.dec_embs = []
        self.dec_ids = []

        n_dec = sum(p.numel() for p in self.decoder.parameters())
        n_neuron = self.neurons[0].param_count()
        print(f"\n  Router:   22M params (80MB, shared)", flush=True)
        print(f"  Decoder:  {n_dec:,} params ({n_dec*4/1e6:.0f}MB, shared)", flush=True)
        print(f"  Neuron:   {n_neuron:,} params ({n_neuron*4/1e6:.0f}MB, x{n_neurons})", flush=True)
        print(f"  Comm:     384 floats = 1.5KB per query\n", flush=True)

    def teach_neuron(self, neuron_id, prompt, target):
        """Teach a specific neuron a fact. Also updates router + decoder."""
        # Train specialist
        loss = self.neurons[neuron_id].train_on(prompt, target)

        # Update router profile
        self.router.update_profile(neuron_id, prompt)

        # Store for decoder training
        full_text = prompt + target
        with torch.no_grad():
            emb = self.encoder.encode([full_text], convert_to_tensor=True).to(DEVICE).clone()
        tokens = self.dec_tokenizer(full_text, return_tensors='pt', truncation=True,
                                    max_length=MAX_LEN, padding='max_length').to(DEVICE)
        self.dec_embs.append(emb)
        self.dec_ids.append(tokens['input_ids'])

        return loss

    def train_decoder_interleaved(self, n_epochs=200, batch_size=8):
        """Train decoder on all accumulated knowledge, interleaved."""
        if not self.dec_embs:
            return
        all_embs = torch.cat(self.dec_embs, dim=0)
        all_ids = torch.cat(self.dec_ids, dim=0)
        n = all_embs.size(0)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.dec_optimizer, T_max=n_epochs, eta_min=1e-5)

        self.decoder.train()
        for epoch in range(n_epochs):
            perm = torch.randperm(n)
            epoch_loss, batches = 0, 0
            for i in range(0, n, batch_size):
                idx = perm[i:i+batch_size]
                logits = self.decoder(all_embs[idx], all_ids[idx])
                loss = F.cross_entropy(
                    logits[:, :-1, :].reshape(-1, VOCAB_SIZE),
                    all_ids[idx][:, 1:].reshape(-1),
                    ignore_index=self.dec_tokenizer.pad_token_id)
                self.dec_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), 1.0)
                self.dec_optimizer.step()
                epoch_loss += loss.item()
                batches += 1
            scheduler.step()
            if (epoch + 1) % 50 == 0:
                print(f"    decoder epoch {epoch+1}/{n_epochs}  "
                      f"loss={epoch_loss/batches:.4f}", flush=True)

    def ask_routed(self, query, top_k=3, latency_ms=0):
        """Route query → best specialist → generate answer."""
        t0 = time.time()

        # Route
        routes = self.router.route(query, top_k=top_k)
        route_ms = (time.time() - t0) * 1000

        if not routes:
            return "[NO SPECIALISTS]", -1, 0, 0

        # Simulate network latency
        if latency_ms > 0:
            time.sleep(latency_ms / 1000)

        # Ask best alive specialist
        for nid, sim in routes:
            if self.neurons[nid].alive:
                response = self.neurons[nid].generate(query, max_tokens=20)
                total_ms = (time.time() - t0) * 1000
                return response, nid, sim, total_ms

        return "[ALL ROUTED NEURONS DEAD]", -1, 0, 0

    def ask_decoder(self, query):
        """Use decoder path: encode query → decode from embedding."""
        with torch.no_grad():
            emb = self.encoder.encode([query], convert_to_tensor=True).to(DEVICE)
        self.decoder.eval()
        with torch.no_grad():
            return self.decoder.generate(emb, self.dec_tokenizer, max_tokens=20)

    def ask_random(self, query):
        """Ask a random alive neuron (baseline)."""
        alive = [n for n in self.neurons.values() if n.alive]
        if not alive:
            return "[ALL DEAD]"
        neuron = random.choice(alive)
        return neuron.generate(query, max_tokens=20)

    def kill_neurons(self, fraction=0.5):
        """Kill a fraction of neurons. Returns list of killed IDs."""
        alive = [nid for nid, n in self.neurons.items() if n.alive]
        n_kill = int(len(alive) * fraction)
        to_kill = random.sample(alive, n_kill)
        for nid in to_kill:
            self.neurons[nid].kill()
        return to_kill

    def alive_count(self):
        return sum(1 for n in self.neurons.values() if n.alive)


# ══════════════════════════════════════════════════════════════
# Topic Data
# ══════════════════════════════════════════════════════════════

TOPICS = {
    "geography": [
        ("The capital of France is", " Paris, the largest city in France"),
        ("The capital of Japan is", " Tokyo, a massive metropolitan area"),
        ("The largest continent is", " Asia, covering about 30% of Earth"),
        ("Mount Everest is located in", " the Himalayas between Nepal and Tibet"),
        ("The longest river is the", " Nile, flowing through northeast Africa"),
    ],
    "science": [
        ("Water freezes at", " zero degrees Celsius or 32 Fahrenheit"),
        ("The sun is a", " medium-sized star in our solar system"),
        ("Photosynthesis is how plants", " convert sunlight into energy"),
        ("DNA contains the genetic instructions for", " all living organisms"),
        ("The speed of light is approximately", " 300,000 kilometers per second"),
    ],
    "math": [
        ("Two plus two equals", " four, a basic arithmetic fact"),
        ("Pi is approximately equal to", " 3.14159, the ratio of circumference to diameter"),
        ("The Pythagorean theorem states that", " a squared plus b squared equals c squared"),
        ("The square root of 144 is", " twelve"),
        ("A prime number is divisible only by", " one and itself"),
    ],
    "tech": [
        ("Python is a popular", " programming language known for its simplicity"),
        ("HTML is used to create", " web pages and structure content on the internet"),
        ("Machine learning is a subset of", " artificial intelligence that learns from data"),
        ("Linux is an open source", " operating system used in servers and development"),
        ("A database stores", " organized collections of data for easy retrieval"),
    ],
    "history": [
        ("World War 2 ended in", " 1945 with the surrender of Germany and Japan"),
        ("The moon landing occurred in", " 1969 when Apollo 11 reached the lunar surface"),
        ("The Renaissance was a period of", " cultural rebirth in 14th century Italy"),
        ("Democracy originated in", " ancient Athens Greece around the 5th century BC"),
        ("The printing press was invented by", " Johannes Gutenberg around 1440"),
    ],
    "food": [
        ("Pizza originated in", " Naples Italy as a simple flatbread dish"),
        ("Sushi is a traditional dish from", " Japan made with vinegared rice"),
        ("Chocolate is made from", " roasted and ground cacao beans"),
        ("Coffee originated in", " Ethiopia and spread through the Arab world"),
        ("Pasta is a staple food of", " Italian cuisine made from wheat flour"),
    ],
    "animals": [
        ("The blue whale is the", " largest animal ever known to have existed"),
        ("Cheetahs are the fastest", " land animals reaching speeds of 70 mph"),
        ("Dolphins are highly intelligent", " marine mammals that communicate with clicks"),
        ("Elephants are the largest", " land animals with remarkable memory"),
        ("Bees are essential for", " pollination of many food crops worldwide"),
    ],
    "language": [
        ("The opposite of hot is", " cold, describing low temperature"),
        ("A synonym for happy is", " joyful or content"),
        ("The past tense of run is", " ran"),
        ("An adjective describes a", " noun or pronoun in a sentence"),
        ("The plural of mouse is", " mice, an irregular plural form"),
    ],
    "music": [
        ("Beethoven composed", " nine symphonies during his lifetime"),
        ("The guitar has", " six strings tuned to specific notes"),
        ("Jazz originated in", " New Orleans in the early 20th century"),
        ("A piano has", " 88 keys spanning over seven octaves"),
        ("Mozart was born in", " Salzburg Austria in 1756"),
    ],
    "sports": [
        ("Football is the most popular", " sport worldwide with billions of fans"),
        ("The Olympics happen every", " four years alternating summer and winter"),
        ("Basketball was invented by", " James Naismith in 1891"),
        ("A marathon is", " 26.2 miles or about 42 kilometers long"),
        ("Tennis is played on", " a rectangular court with a net in the middle"),
    ],
}

TOPIC_KEYWORDS = {
    "geography": ["paris", "tokyo", "asia", "everest", "himalaya", "nile"],
    "science": ["zero", "celsius", "star", "solar", "dna", "light", "300"],
    "math": ["four", "pi", "3.14", "pythagorean", "twelve", "prime"],
    "tech": ["programming", "language", "web", "pages", "linux", "database"],
    "history": ["1945", "1969", "renaissance", "athens", "gutenberg"],
    "food": ["naples", "italy", "japan", "cacao", "cocoa", "ethiopia"],
    "animals": ["largest", "fastest", "intelligent", "mammal", "pollinat"],
    "language": ["cold", "joyful", "ran", "noun", "mice"],
    "music": ["nine", "symphon", "six strings", "new orleans", "salzburg"],
    "sports": ["worldwide", "billion", "four years", "naismith", "26.2", "42"],
}


def check_answer(response, topic):
    """Check if response contains expected keywords for the topic."""
    resp_lower = response.strip().lower()
    for kw in TOPIC_KEYWORDS.get(topic, []):
        if kw in resp_lower:
            return True
    return False


def run(n_neurons=10):
    brain = InternetBrain(n_neurons=n_neurons)

    topic_names = list(TOPICS.keys())
    # Assign topics to neurons (1 neuron per topic)
    assignments = {}
    for i, topic in enumerate(topic_names):
        nid = i % n_neurons
        assignments[topic] = nid

    # ─── PHASE 1: TRAIN SPECIALISTS ───────────────────────
    print("── PHASE 1: TRAINING SPECIALISTS ──\n", flush=True)
    t0 = time.time()
    for epoch in range(5):
        total_loss, steps = 0, 0
        for topic, nid in assignments.items():
            for prompt, target in TOPICS[topic]:
                loss = brain.teach_neuron(nid, prompt, target)
                total_loss += loss
                steps += 1
        print(f"  epoch {epoch+1}/5  loss={total_loss/steps:.4f}", flush=True)
    train_spec_time = time.time() - t0
    print(f"  Specialist training: {train_spec_time:.0f}s\n", flush=True)

    # ─── PHASE 2: TRAIN DECODER ──────────────────────────
    print("── PHASE 2: TRAINING DECODER (interleaved) ──\n", flush=True)
    t0 = time.time()
    brain.train_decoder_interleaved(n_epochs=200, batch_size=8)
    train_dec_time = time.time() - t0
    print(f"  Decoder training: {train_dec_time:.0f}s\n", flush=True)

    # ─── PHASE 3: ROUTING + GENERATION TEST ──────────────
    print("── PHASE 3: ROUTED SPECIALIST TEST ──\n", flush=True)
    test_queries = [
        ("The capital of France is", "geography", 0),
        ("Water freezes at", "science", 1),
        ("Two plus two equals", "math", 2),
        ("Python is a popular", "tech", 3),
        ("World War 2 ended in", "history", 4),
        ("Pizza originated in", "food", 5),
        ("The blue whale is the", "animals", 6),
        ("The opposite of hot is", "language", 7),
        ("Beethoven composed", "music", 8),
        ("Football is the most popular", "sports", 9),
    ]

    routed_correct, routed_total = 0, 0
    route_correct_count = 0
    for query, topic, expected_nid in test_queries:
        response, nid, sim, ms = brain.ask_routed(query, top_k=3)
        match = check_answer(response, topic)
        route_ok = nid == expected_nid
        routed_correct += match
        route_correct_count += route_ok
        routed_total += 1
        print(f"  [{'OK' if match else 'XX'}] route→n{nid} "
              f"({'RIGHT' if route_ok else f'WRONG({expected_nid})':>10s}) "
              f"sim={sim:.3f} [{ms:.0f}ms]", flush=True)
        print(f"       Q: \"{query}\"", flush=True)
        print(f"       A: \"{response.strip()[:60]}\"", flush=True)

    print(f"\n  Routing accuracy:    {route_correct_count}/{routed_total} = "
          f"{route_correct_count/routed_total:.0%}", flush=True)
    print(f"  Generation accuracy: {routed_correct}/{routed_total} = "
          f"{routed_correct/routed_total:.0%}", flush=True)

    # ─── PHASE 4: RANDOM BASELINE ────────────────────────
    print("\n── PHASE 4: RANDOM BASELINE ──\n", flush=True)
    random_correct = 0
    n_trials = 30  # 3 random trials per query
    for query, topic, _ in test_queries:
        for _ in range(3):
            response = brain.ask_random(query)
            if check_answer(response, topic):
                random_correct += 1
    print(f"  Random accuracy: {random_correct}/{n_trials} = "
          f"{random_correct/n_trials:.0%}", flush=True)

    # ─── PHASE 5: DECODER PATH ───────────────────────────
    print("\n── PHASE 5: DECODER PATH ──\n", flush=True)
    dec_correct = 0
    for query, topic, _ in test_queries:
        response = brain.ask_decoder(query)
        match = check_answer(response, topic)
        dec_correct += match
        print(f"  [{'OK' if match else 'XX'}] Q: \"{query}\"", flush=True)
        print(f"       A: \"{response.strip()[:60]}\"", flush=True)
    print(f"\n  Decoder accuracy: {dec_correct}/{routed_total} = "
          f"{dec_correct/routed_total:.0%}", flush=True)

    # ─── PHASE 6: NOVEL QUERIES ──────────────────────────
    print("\n── PHASE 6: NOVEL QUERIES (never trained on) ──\n", flush=True)
    novel_queries = [
        ("What country is Tokyo in?", "geography"),
        ("How fast does light travel?", "science"),
        ("What is the value of pi?", "math"),
        ("What language is used for web pages?", "tech"),
        ("When was the moon landing?", "history"),
        ("Where does coffee come from?", "food"),
        ("What is the fastest land animal?", "animals"),
        ("What is the past tense of run?", "language"),
        ("Where did jazz originate?", "music"),
        ("Who invented basketball?", "sports"),
    ]
    novel_routed, novel_dec = 0, 0
    for query, topic in novel_queries:
        r_resp, nid, sim, ms = brain.ask_routed(query, top_k=3)
        d_resp = brain.ask_decoder(query)
        r_match = check_answer(r_resp, topic)
        d_match = check_answer(d_resp, topic)
        novel_routed += r_match
        novel_dec += d_match
        print(f"  Q: \"{query}\"", flush=True)
        print(f"    Routed  [{'OK' if r_match else 'XX'}] n{nid} sim={sim:.3f}: "
              f"\"{r_resp.strip()[:50]}\"", flush=True)
        print(f"    Decoder [{'OK' if d_match else 'XX'}]: "
              f"\"{d_resp.strip()[:50]}\"", flush=True)

    print(f"\n  Novel routed:  {novel_routed}/{len(novel_queries)} = "
          f"{novel_routed/len(novel_queries):.0%}", flush=True)
    print(f"  Novel decoder: {novel_dec}/{len(novel_queries)} = "
          f"{novel_dec/len(novel_queries):.0%}", flush=True)

    # ─── PHASE 7: RESILIENCE (KILL 50%) ──────────────────
    print("\n── PHASE 7: RESILIENCE (kill 50% of neurons) ──\n", flush=True)
    killed = brain.kill_neurons(fraction=0.5)
    print(f"  Killed neurons: {killed}", flush=True)
    print(f"  Alive: {brain.alive_count()}/{n_neurons}\n", flush=True)

    res_routed, res_dec = 0, 0
    for query, topic, _ in test_queries:
        r_resp, nid, sim, ms = brain.ask_routed(query, top_k=3)
        d_resp = brain.ask_decoder(query)
        r_match = check_answer(r_resp, topic)
        d_match = check_answer(d_resp, topic)
        res_routed += r_match
        res_dec += d_match
        status_r = "OK" if r_match else "XX"
        status_d = "OK" if d_match else "XX"
        nid_str = f"n{nid}" if nid >= 0 else "DEAD"
        print(f"  [{status_r}] Routed→{nid_str:>4s}  [{status_d}] Decoder  "
              f"Q: \"{query[:35]}\"", flush=True)

    print(f"\n  Post-kill routed:  {res_routed}/{routed_total} = "
          f"{res_routed/routed_total:.0%}", flush=True)
    print(f"  Post-kill decoder: {res_dec}/{routed_total} = "
          f"{res_dec/routed_total:.0%} (unchanged — decoder is shared)", flush=True)

    # ─── PHASE 8: LATENCY SIMULATION ─────────────────────
    print("\n── PHASE 8: LATENCY SIMULATION ──\n", flush=True)
    for latency in [0, 15, 50, 100]:
        t0 = time.time()
        n_queries = 5
        for query, _, _ in test_queries[:n_queries]:
            brain.ask_routed(query, latency_ms=latency)
        elapsed = (time.time() - t0) * 1000
        avg = elapsed / n_queries
        print(f"  Latency={latency:3d}ms  →  avg={avg:.0f}ms/query  "
              f"total={elapsed:.0f}ms ({n_queries} queries)", flush=True)

    # ─── PHASE 9: ONLINE LEARNING ────────────────────────
    print("\n── PHASE 9: ONLINE LEARNING (teach new fact) ──\n", flush=True)
    # Pick an alive neuron and teach it something new
    alive_neurons = [nid for nid, n in brain.neurons.items() if n.alive]
    if alive_neurons:
        teach_nid = alive_neurons[0]
        new_facts = [
            ("The Eiffel Tower is in", " Paris France and is 330 meters tall"),
            ("The Great Wall is in", " China and is over 13000 miles long"),
        ]
        print(f"  Teaching neuron {teach_nid} new facts...", flush=True)
        for prompt, target in new_facts:
            for _ in range(30):
                brain.neurons[teach_nid].train_on(prompt, target)
            brain.router.update_profile(teach_nid, prompt)
            print(f"    Taught: \"{prompt}{target[:30]}...\"", flush=True)

        # Test recall
        for prompt, _ in new_facts:
            resp, nid, sim, ms = brain.ask_routed(prompt)
            print(f"    Q: \"{prompt}\"  →  n{nid} sim={sim:.3f}: "
                  f"\"{resp.strip()[:50]}\"", flush=True)

    # ─── SUMMARY ─────────────────────────────────────────
    print(f"\n{'='*60}", flush=True)
    print("INTERNET BRAIN v4 — RESULTS SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  Neurons:            {n_neurons} (GPT-2, independent)", flush=True)
    print(f"  Router:             SBERT all-MiniLM-L6-v2 (shared)", flush=True)
    print(f"  Decoder:            48.9M params (shared)", flush=True)
    print(f"  Communication:      1.5KB per query", flush=True)
    print(f"  ────────────────────────────────────────", flush=True)
    print(f"  Routing accuracy:   {route_correct_count}/{routed_total} = "
          f"{route_correct_count/routed_total:.0%}", flush=True)
    print(f"  Routed generation:  {routed_correct}/{routed_total} = "
          f"{routed_correct/routed_total:.0%}", flush=True)
    print(f"  Random baseline:    {random_correct}/{n_trials} = "
          f"{random_correct/n_trials:.0%}", flush=True)
    print(f"  Decoder path:       {dec_correct}/{routed_total} = "
          f"{dec_correct/routed_total:.0%}", flush=True)
    print(f"  Novel (routed):     {novel_routed}/{len(novel_queries)} = "
          f"{novel_routed/len(novel_queries):.0%}", flush=True)
    print(f"  Novel (decoder):    {novel_dec}/{len(novel_queries)} = "
          f"{novel_dec/len(novel_queries):.0%}", flush=True)
    print(f"  Post-kill routed:   {res_routed}/{routed_total} = "
          f"{res_routed/routed_total:.0%} (50% neurons killed)", flush=True)
    print(f"  Post-kill decoder:  {res_dec}/{routed_total} = "
          f"{res_dec/routed_total:.0%} (decoder unaffected)", flush=True)
    print(f"  Routing benefit:    {(routed_correct/routed_total)/(random_correct/n_trials+0.001):.1f}x over random",
          flush=True)
    print(f"  Train time:         {train_spec_time:.0f}s specialists + "
          f"{train_dec_time:.0f}s decoder", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--neurons", type=int, default=10)
    args = p.parse_args()
    run(n_neurons=args.neurons)
