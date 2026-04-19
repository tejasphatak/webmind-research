#!/usr/bin/env python3
"""
Internet Brain v3 — Interleaved Training Fix
=============================================
v2 had catastrophic forgetting: training facts sequentially → decoder collapsed to last few.
Fix: pre-encode ALL facts, then train in shuffled mini-batches across all facts each epoch.

Architecture unchanged:
- Encoder: sentence-transformers/all-MiniLM-L6-v2 (80MB, shared, frozen)
- Decoder: trained transformer → reconstructs text from 384-dim embeddings
- Communication: 1.5KB per query (one embedding vector)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from transformers import GPT2Tokenizer
import time, random

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMB_DIM = 384
VOCAB_SIZE = 50257
MAX_LEN = 32


class TextDecoder(nn.Module):
    """Reverse sentence transformer: embedding → text."""

    def __init__(self, emb_dim=EMB_DIM, vocab_size=VOCAB_SIZE,
                 hidden=512, n_layers=4, max_len=MAX_LEN):
        super().__init__()
        self.max_len = max_len
        self.emb_dim = emb_dim
        self.hidden = hidden

        self.emb_to_hidden = nn.Linear(emb_dim, hidden * max_len)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden, nhead=8, dim_feedforward=hidden*4,
            batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        self.to_vocab = nn.Linear(hidden, vocab_size)
        self.pos_emb = nn.Embedding(max_len, hidden)

    def forward(self, embedding, target_ids=None):
        B = embedding.size(0)
        memory = self.emb_to_hidden(embedding)
        memory = memory.view(B, self.max_len, self.hidden)

        S = target_ids.size(1) if target_ids is not None else self.max_len
        pos = torch.arange(S, device=embedding.device)
        tgt = self.pos_emb(pos).unsqueeze(0).expand(B, -1, -1)

        mask = nn.Transformer.generate_square_subsequent_mask(S).to(embedding.device)
        out = self.decoder(tgt, memory[:, :S, :], tgt_mask=mask)
        logits = self.to_vocab(out)
        return logits

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


class InternetBrainV3:
    """Encoder + decoder with interleaved batch training."""

    def __init__(self):
        print("Loading sentence transformer (encoder)...", flush=True)
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2', device=DEVICE)

        print("Creating text decoder...", flush=True)
        self.decoder = TextDecoder().to(DEVICE)

        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.optimizer = torch.optim.Adam(self.decoder.parameters(), lr=1e-3)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=200, eta_min=1e-5)

        self.n_params = sum(p.numel() for p in self.decoder.parameters())
        print(f"  Decoder: {self.n_params:,} params ({self.n_params*4/1e6:.0f}MB)",
              flush=True)

        # Knowledge base: pre-encoded facts
        self.knowledge_embs = []  # list of [1, EMB_DIM] tensors
        self.knowledge_ids = []   # list of [1, MAX_LEN] tensors
        self.knowledge_texts = []

    def add_fact(self, text):
        """Pre-encode a fact and store it."""
        with torch.no_grad():
            emb = self.encoder.encode([text], convert_to_tensor=True).to(DEVICE).clone()
        tokens = self.tokenizer(text, return_tensors='pt', truncation=True,
                               max_length=MAX_LEN, padding='max_length').to(DEVICE)
        self.knowledge_embs.append(emb)
        self.knowledge_ids.append(tokens['input_ids'])
        self.knowledge_texts.append(text)

    def train_interleaved(self, n_epochs=200, batch_size=8):
        """Train on ALL facts in shuffled mini-batches each epoch."""
        n_facts = len(self.knowledge_embs)
        if n_facts == 0:
            return

        # Stack all data
        all_embs = torch.cat(self.knowledge_embs, dim=0)   # [N, EMB_DIM]
        all_ids = torch.cat(self.knowledge_ids, dim=0)      # [N, MAX_LEN]

        self.decoder.train()
        for epoch in range(n_epochs):
            # Shuffle
            perm = torch.randperm(n_facts)
            epoch_loss = 0
            n_batches = 0

            for i in range(0, n_facts, batch_size):
                idx = perm[i:i+batch_size]
                emb_batch = all_embs[idx]
                ids_batch = all_ids[idx]

                logits = self.decoder(emb_batch, ids_batch)
                loss = F.cross_entropy(
                    logits[:, :-1, :].reshape(-1, VOCAB_SIZE),
                    ids_batch[:, 1:].reshape(-1),
                    ignore_index=self.tokenizer.pad_token_id)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), 1.0)
                self.optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            self.scheduler.step()

            if (epoch + 1) % 20 == 0:
                avg = epoch_loss / n_batches
                lr = self.optimizer.param_groups[0]['lr']
                print(f"  epoch {epoch+1}/{n_epochs}  loss={avg:.4f}  lr={lr:.6f}",
                      flush=True)

        return epoch_loss / n_batches

    def ask(self, question):
        with torch.no_grad():
            emb = self.encoder.encode([question], convert_to_tensor=True).to(DEVICE)
        self.decoder.eval()
        with torch.no_grad():
            return self.decoder.generate(emb, self.tokenizer, max_tokens=20)


def run():
    print("=== INTERNET BRAIN v3 — INTERLEAVED TRAINING ===\n", flush=True)
    print(f"  Device: {DEVICE}", flush=True)

    brain = InternetBrainV3()

    facts = [
        "Paris is the capital of France",
        "Tokyo is the capital of Japan",
        "Cairo is the capital of Egypt",
        "Water freezes at zero degrees Celsius",
        "The sun is a star in our solar system",
        "Gravity pulls objects toward each other",
        "Two plus two equals four",
        "Pi is approximately 3.14",
        "The square root of nine is three",
        "Python is a programming language",
        "HTML is used for web pages",
        "Linux is an operating system",
        "Pizza comes from Italy",
        "Sushi is from Japan",
        "Chocolate is made from cocoa beans",
        "The blue whale is the largest animal",
        "Cheetahs are the fastest land animals",
        "The opposite of hot is cold",
        "The past tense of go is went",
        "World War 2 ended in 1945",
    ]

    # Phase 1: Pre-encode all facts
    print(f"\nPre-encoding {len(facts)} facts...", flush=True)
    t0 = time.time()
    for fact in facts:
        brain.add_fact(fact)
    print(f"  Encoded in {time.time()-t0:.1f}s\n", flush=True)

    # Phase 2: Interleaved training
    print("Training decoder (interleaved batches)...\n", flush=True)
    t0 = time.time()
    final_loss = brain.train_interleaved(n_epochs=200, batch_size=8)
    train_time = time.time() - t0
    print(f"\n  Training done in {train_time:.0f}s  final_loss={final_loss:.4f}\n",
          flush=True)

    # Phase 3: Recall test
    print("=== RECALL TEST (exact facts) ===\n", flush=True)
    correct = 0
    for fact in facts:
        # Extract query from fact
        if ' is ' in fact:
            query = fact.split(' is ')[0]
        elif ' equals ' in fact:
            query = fact.split(' equals ')[0]
        elif ' freezes ' in fact:
            query = fact.split(' freezes ')[0]
        elif ' pulls ' in fact:
            query = "Gravity"
        elif ' comes ' in fact:
            query = fact.split(' comes ')[0]
        elif ' ended ' in fact:
            query = fact.split(' ended ')[0]
        else:
            query = fact[:20]

        answer = brain.ask(query)
        keywords = fact.lower().split()[-2:]
        match = any(kw in answer.lower() for kw in keywords)
        correct += match
        status = "OK" if match else "XX"
        print(f"  [{status}] Q: {query}", flush=True)
        print(f"       A: {answer.strip()[:60]}", flush=True)

    pct = correct / len(facts)
    print(f"\n  Recall: {correct}/{len(facts)} = {pct:.0%}", flush=True)

    # Phase 4: Novel questions
    print("\n=== NOVEL QUESTIONS ===\n", flush=True)
    novel = [
        "What is the capital of France?",
        "At what temperature does water freeze?",
        "What is two plus two?",
        "Where does pizza come from?",
        "What is the largest animal?",
        "What programming language is Python?",
        "When did World War 2 end?",
        "What is the opposite of hot?",
    ]
    novel_correct = 0
    expected_keywords = [
        "france", "zero", "four", "italy", "whale",
        "programming", "1945", "cold"
    ]
    for q, kw in zip(novel, expected_keywords):
        answer = brain.ask(q)
        match = kw in answer.lower()
        novel_correct += match
        status = "OK" if match else "XX"
        print(f"  [{status}] Q: {q}", flush=True)
        print(f"       A: {answer.strip()[:60]}", flush=True)

    print(f"\n  Novel: {novel_correct}/{len(novel)} = {novel_correct/len(novel):.0%}",
          flush=True)

    # Summary
    print(f"\n{'='*50}", flush=True)
    print(f"SUMMARY", flush=True)
    print(f"{'='*50}", flush=True)
    print(f"  Recall:  {correct}/{len(facts)} = {pct:.0%}", flush=True)
    print(f"  Novel:   {novel_correct}/{len(novel)} = {novel_correct/len(novel):.0%}",
          flush=True)
    print(f"  Device:  {DEVICE}", flush=True)
    print(f"  Params:  {brain.n_params:,}", flush=True)
    print(f"  Train:   {train_time:.0f}s ({200} epochs, interleaved)", flush=True)
    print(f"  v2→v3:   sequential → interleaved batches", flush=True)


if __name__ == "__main__":
    run()
