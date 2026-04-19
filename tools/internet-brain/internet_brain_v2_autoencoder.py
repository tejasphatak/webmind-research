#!/usr/bin/env python3
"""
Internet Brain v2 — Sentence Autoencoder Architecture
======================================================
- Encoder: sentence-transformers (shared, understands language)
- Decoder: trained to reconstruct text from embeddings
- Specialists: produce better embeddings for their domain
- Communication: just 384 floats (1.5KB) per query

Train the decoder on facts → it learns to reverse embeddings to text.
Route questions to specialist neurons → get domain-specific embedding → decode to answer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from transformers import GPT2Tokenizer
import time

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMB_DIM = 384  # MiniLM output dimension
VOCAB_SIZE = 50257
MAX_LEN = 32


class TextDecoder(nn.Module):
    """Reverse sentence transformer: embedding → text.
    Shared across all devices. Trained from usage."""

    def __init__(self, emb_dim=EMB_DIM, vocab_size=VOCAB_SIZE,
                 hidden=512, n_layers=4, max_len=MAX_LEN):
        super().__init__()
        self.max_len = max_len
        self.emb_dim = emb_dim

        # Project embedding to sequence of hidden states
        self.emb_to_hidden = nn.Linear(emb_dim, hidden * max_len)

        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden, nhead=8, dim_feedforward=hidden*4,
            batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        # Output to vocab
        self.to_vocab = nn.Linear(hidden, vocab_size)

        # Positional embedding
        self.pos_emb = nn.Embedding(max_len, hidden)

        self.hidden = hidden

    def forward(self, embedding, target_ids=None):
        """
        embedding: [B, EMB_DIM] — from sentence transformer
        target_ids: [B, S] — target token ids (for teacher forcing)
        Returns: logits [B, S, VOCAB]
        """
        B = embedding.size(0)

        # Project embedding to memory for decoder
        memory = self.emb_to_hidden(embedding)  # [B, hidden*max_len]
        memory = memory.view(B, self.max_len, self.hidden)  # [B, max_len, hidden]

        # Decoder input: positional embeddings
        S = target_ids.size(1) if target_ids is not None else self.max_len
        pos = torch.arange(S, device=embedding.device)
        tgt = self.pos_emb(pos).unsqueeze(0).expand(B, -1, -1)  # [B, S, hidden]

        # Causal mask
        mask = nn.Transformer.generate_square_subsequent_mask(S).to(embedding.device)

        # Decode
        out = self.decoder(tgt, memory[:, :S, :], tgt_mask=mask)
        logits = self.to_vocab(out)
        return logits

    def generate(self, embedding, tokenizer, max_tokens=20):
        """Autoregressive generation from embedding."""
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


class InternetBrainV2:
    """The full system: encoder + decoder + routing."""

    def __init__(self):
        print("Loading sentence transformer (encoder)...", flush=True)
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2', device=DEVICE)

        print("Creating text decoder...", flush=True)
        self.decoder = TextDecoder().to(DEVICE)

        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.optimizer = torch.optim.Adam(self.decoder.parameters(), lr=1e-3)

        # Neuron profiles for routing
        self.profiles = {}
        self.n_params = sum(p.numel() for p in self.decoder.parameters())
        print(f"  Decoder: {self.n_params:,} params ({self.n_params*4/1e6:.0f}MB)",
              flush=True)

    def teach(self, text, n_repeats=50):
        """Teach a fact: encode it, then train decoder to reconstruct it."""
        # Encode
        with torch.no_grad():
            emb = self.encoder.encode([text], convert_to_tensor=True).to(DEVICE).clone()

        # Target tokens
        tokens = self.tokenizer(text, return_tensors='pt', truncation=True,
                               max_length=MAX_LEN, padding='max_length').to(DEVICE)
        target_ids = tokens['input_ids']

        # Train decoder to reconstruct text from embedding
        self.decoder.train()
        for _ in range(n_repeats):
            logits = self.decoder(emb, target_ids)
            loss = F.cross_entropy(
                logits[:, :-1, :].reshape(-1, VOCAB_SIZE),
                target_ids[:, 1:].reshape(-1),
                ignore_index=self.tokenizer.pad_token_id)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), 1.0)
            self.optimizer.step()

        return loss.item()

    def ask(self, question):
        """Encode question → decode answer."""
        with torch.no_grad():
            emb = self.encoder.encode([question], convert_to_tensor=True).to(DEVICE)

        self.decoder.eval()
        with torch.no_grad():
            return self.decoder.generate(emb, self.tokenizer, max_tokens=20)


def run():
    print("=== INTERNET BRAIN v2 — AUTOENCODER ===\n", flush=True)

    brain = InternetBrainV2()

    # Bootstrap: teach it facts
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

    print(f"Teaching {len(facts)} facts...\n", flush=True)
    t0 = time.time()
    for i, fact in enumerate(facts):
        loss = brain.teach(fact, n_repeats=50)
        if (i + 1) % 5 == 0:
            print(f"  {i+1}/{len(facts)}  loss={loss:.4f}", flush=True)

    teach_time = time.time() - t0
    print(f"\n  Teaching done in {teach_time:.0f}s\n", flush=True)

    # Test: can it recall what it learned?
    print("=== RECALL TEST (exact facts) ===\n", flush=True)
    correct = 0
    for fact in facts:
        answer = brain.ask(fact.split(' is ')[0] if ' is ' in fact
                          else fact.split(' equals ')[0] if ' equals ' in fact
                          else fact[:20])
        # Check if key words appear
        keywords = fact.lower().split()[-2:]
        match = any(kw in answer.lower() for kw in keywords)
        correct += match
        status = "OK" if match else "XX"
        print(f"  [{status}] Q: {fact[:40]}", flush=True)
        print(f"       A: {answer.strip()[:60]}", flush=True)

    print(f"\n  Recall: {correct}/{len(facts)} = {correct/len(facts):.0%}",
          flush=True)

    # Test: novel questions
    print("\n=== NOVEL QUESTIONS (never taught exactly) ===\n", flush=True)
    novel = [
        "What is the capital of France?",
        "At what temperature does water freeze?",
        "What is two plus two?",
        "Where does pizza come from?",
        "What is the largest animal?",
    ]
    for q in novel:
        answer = brain.ask(q)
        print(f"  Q: {q}", flush=True)
        print(f"  A: {answer.strip()[:60]}\n", flush=True)


if __name__ == "__main__":
    run()
