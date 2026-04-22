"""
Concept-to-Text Decoder: tiny transformer that turns concept lists into fluent text.

Architecture: 4-layer causal transformer, 256 dim, 4 heads, ~8M params.
Input: concept tokens from brain convergence (unordered keywords).
Output: grammatical sentence expressing those concepts.

Training data: (concept_set, full_sentence) pairs extracted from the brain's
taught sentences. The brain already stores these — we just need to extract
concept keywords and pair them with the original text.

This is the SPEAKING module. The brain is the THINKING module.
Together: think → speak.

Invariant compliance:
  - Inspectable: 8M params, can dump any attention head
  - Editable: retrain on corrected data
  - Traceable: input concepts are from convergence trace
  - The decoder only reformats — it doesn't add knowledge
"""

import math
import os
import json
import time
import re
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# --- Tokenizer (shared with brain vocabulary) ---

class BrainTokenizer:
    """Simple word-level tokenizer using the brain's vocabulary.

    Special tokens: <pad>=0, <bos>=1, <eos>=2, <sep>=3, <unk>=4
    Word tokens start at index 5.
    """

    PAD, BOS, EOS, SEP, UNK = 0, 1, 2, 3, 4
    SPECIAL = ['<pad>', '<bos>', '<eos>', '<sep>', '<unk>']

    def __init__(self, words: list = None, max_vocab: int = 50000):
        self.word2idx = {w: i for i, w in enumerate(self.SPECIAL)}
        self.idx2word = list(self.SPECIAL)
        self.max_vocab = max_vocab

        if words:
            for w in words[:max_vocab - len(self.SPECIAL)]:
                if w not in self.word2idx:
                    idx = len(self.idx2word)
                    self.word2idx[w] = idx
                    self.idx2word.append(w)

    @property
    def vocab_size(self):
        return len(self.idx2word)

    def encode(self, text: str) -> List[int]:
        tokens = re.findall(r'[a-z0-9]+', text.lower())
        return [self.word2idx.get(t, self.UNK) for t in tokens]

    def decode(self, ids: List[int]) -> str:
        words = []
        for idx in ids:
            if idx in (self.PAD, self.BOS, self.SEP):
                continue
            if idx == self.EOS:
                break
            if 0 <= idx < len(self.idx2word):
                words.append(self.idx2word[idx])
        return ' '.join(words)

    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump({'words': self.idx2word}, f)

    @classmethod
    def load(cls, path: str):
        with open(path) as f:
            data = json.load(f)
        tok = cls()
        tok.idx2word = data['words']
        tok.word2idx = {w: i for i, w in enumerate(tok.idx2word)}
        return tok


# --- Model ---

class ConceptDecoder(nn.Module):
    """Tiny causal transformer: concepts → fluent text.

    Encoder-decoder attention where:
      - Encoder input: unordered concept tokens (from brain convergence)
      - Decoder input: autoregressive text generation

    Architecture:
      - Shared embedding (vocab_size × d_model)
      - N decoder layers with self-attention + cross-attention
      - Output projection to vocab
    """

    def __init__(self, vocab_size: int, d_model: int = 256, n_heads: int = 4,
                 n_layers: int = 4, d_ff: int = 512, max_len: int = 128,
                 dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        # Shared embeddings
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)

        # Decoder layers with cross-attention to concept encoder
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)

        # Tie output weights to embedding
        self.output_proj.weight = self.embed.weight

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode_concepts(self, concept_ids: torch.Tensor) -> torch.Tensor:
        """Encode concept tokens (no positional encoding — concepts are unordered)."""
        return self.embed(concept_ids) * math.sqrt(self.d_model)

    def forward(self, concept_ids: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        """
        concept_ids: (batch, concept_len) — unordered concept tokens
        target_ids:  (batch, seq_len) — autoregressive target sequence

        Returns: logits (batch, seq_len, vocab_size)
        """
        # Encode concepts (cross-attention keys/values)
        concept_enc = self.encode_concepts(concept_ids)  # (B, C, D)

        # Decode target sequence
        B, T = target_ids.shape
        positions = torch.arange(T, device=target_ids.device).unsqueeze(0)
        x = self.embed(target_ids) * math.sqrt(self.d_model) + self.pos_embed(positions)
        x = self.dropout(x)

        # Causal mask
        causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()

        # Concept padding mask
        concept_pad_mask = (concept_ids == 0)  # True where padded

        for layer in self.layers:
            x = layer(x, concept_enc, causal_mask, concept_pad_mask)

        x = self.norm(x)
        return self.output_proj(x)

    @torch.no_grad()
    def generate(self, concept_ids: torch.Tensor, max_len: int = 64,
                 temperature: float = 0.7, top_k: int = 50) -> List[int]:
        """Autoregressive generation from concept input."""
        self.eval()
        concept_enc = self.encode_concepts(concept_ids)  # (1, C, D)

        # Start with BOS
        generated = [1]  # BOS
        device = concept_ids.device

        for _ in range(max_len):
            target = torch.tensor([generated], device=device)
            T = target.shape[1]
            positions = torch.arange(T, device=device).unsqueeze(0)
            x = self.embed(target) * math.sqrt(self.d_model) + self.pos_embed(positions)

            causal_mask = torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()
            concept_pad_mask = (concept_ids == 0)

            for layer in self.layers:
                x = layer(x, concept_enc, causal_mask, concept_pad_mask)

            x = self.norm(x)
            logits = self.output_proj(x[:, -1, :])  # last token logits

            # Temperature + top-k sampling
            logits = logits / max(temperature, 1e-8)

            if top_k > 0:
                values, _ = torch.topk(logits, top_k)
                min_val = values[:, -1].unsqueeze(-1)
                logits = torch.where(logits < min_val,
                                     torch.full_like(logits, float('-inf')),
                                     logits)

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()

            if next_token == 2:  # EOS
                break
            generated.append(next_token)

        return generated[1:]  # strip BOS

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class DecoderLayer(nn.Module):
    """Transformer decoder layer: self-attention + cross-attention + FFN."""

    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads,
                                                dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads,
                                                 dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, concept_enc, causal_mask, concept_pad_mask):
        # Self-attention (causal)
        residual = x
        x = self.norm1(x)
        x, _ = self.self_attn(x, x, x, attn_mask=causal_mask)
        x = self.dropout(x) + residual

        # Cross-attention to concepts
        residual = x
        x = self.norm2(x)
        x, _ = self.cross_attn(x, concept_enc, concept_enc,
                                key_padding_mask=concept_pad_mask)
        x = self.dropout(x) + residual

        # FFN
        residual = x
        x = self.norm3(x)
        x = self.ffn(x)
        x = self.dropout(x) + residual

        return x


# --- Training Dataset ---

class ConceptTextDataset(Dataset):
    """Pairs of (concept_keywords, full_sentence) for decoder training.

    Extracts concepts by removing function words from the sentence,
    then pairs concept set with original sentence as target.
    """

    STOP_WORDS = frozenset({
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "must", "of", "in", "to",
        "for", "with", "on", "at", "by", "from", "as", "into", "through",
        "and", "but", "or", "not", "so", "yet", "both", "either", "neither",
        "this", "that", "these", "those", "it", "its", "if", "then",
        "also", "very", "just", "about", "up", "out", "off", "over",
    })

    def __init__(self, tokenizer: BrainTokenizer, data_paths: list,
                 max_concepts: int = 16, max_target_len: int = 64,
                 limit: int = None):
        self.tokenizer = tokenizer
        self.max_concepts = max_concepts
        self.max_target_len = max_target_len
        self.pairs = []

        for path in data_paths:
            if not os.path.exists(path):
                continue
            with open(path) as f:
                for line in f:
                    try:
                        item = json.loads(line.strip())
                    except json.JSONDecodeError:
                        continue

                    # Get text (from 'answer', 'text', or constructed from Q+A)
                    text = item.get('answer', '') or item.get('text', '')
                    if not text or len(text) < 20 or len(text) > 500:
                        continue

                    # Extract concept keywords
                    words = re.findall(r'[a-z0-9]+', text.lower())
                    concepts = [w for w in words if w not in self.STOP_WORDS and len(w) > 2]
                    if len(concepts) < 2:
                        continue

                    self.pairs.append((concepts[:max_concepts], text.lower()))

                    if limit and len(self.pairs) >= limit:
                        break
            if limit and len(self.pairs) >= limit:
                break

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        concepts, text = self.pairs[idx]

        # Encode concepts (no order)
        concept_ids = [self.tokenizer.word2idx.get(c, self.tokenizer.UNK)
                       for c in concepts]

        # Encode target: BOS + text + EOS
        target_ids = [self.tokenizer.BOS] + self.tokenizer.encode(text) + [self.tokenizer.EOS]
        target_ids = target_ids[:self.max_target_len]

        return concept_ids, target_ids

    @staticmethod
    def collate(batch):
        """Pad concepts and targets to batch max length."""
        concepts, targets = zip(*batch)

        max_c = max(len(c) for c in concepts)
        max_t = max(len(t) for t in targets)

        concept_padded = torch.zeros(len(concepts), max_c, dtype=torch.long)
        target_padded = torch.zeros(len(targets), max_t, dtype=torch.long)

        for i, (c, t) in enumerate(zip(concepts, targets)):
            concept_padded[i, :len(c)] = torch.tensor(c)
            target_padded[i, :len(t)] = torch.tensor(t)

        return concept_padded, target_padded


# --- Training ---

def train_decoder(brain_db_path: str = None, data_dir: str = None,
                  epochs: int = 10, batch_size: int = 64, lr: float = 3e-4,
                  limit: int = 100000, save_path: str = None):
    """Train the concept decoder on brain's stored data.

    Args:
        brain_db_path: path to nexus-brain (for vocabulary)
        data_dir: path to data/*.jsonl
        epochs: training epochs
        batch_size: batch size
        lr: learning rate
        limit: max training pairs
        save_path: where to save model
    """
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

    if brain_db_path is None:
        brain_db_path = os.path.expanduser('~/nexus-brain')
    if data_dir is None:
        data_dir = str(Path.home() / 'webmind-research' / 'data')
    if save_path is None:
        save_path = os.path.join(brain_db_path, 'decoder')

    os.makedirs(save_path, exist_ok=True)

    # Load vocabulary from brain
    print("Loading brain vocabulary...")
    import lmdb
    import struct
    env = lmdb.open(os.path.join(brain_db_path, 'brain.lmdb'), max_dbs=16, readonly=True)
    words_db = env.open_db(b'words')
    words = []
    with env.begin(db=words_db) as txn:
        cursor = txn.cursor(db=words_db)
        for key, val in cursor:
            words.append(key.decode('utf-8'))
    env.close()
    print(f"  Vocabulary: {len(words):,} words")

    # Build tokenizer from brain vocabulary
    tokenizer = BrainTokenizer(words, max_vocab=50000)
    print(f"  Tokenizer: {tokenizer.vocab_size:,} tokens")

    # Build dataset from all data files
    data_paths = sorted(Path(data_dir).glob('*.jsonl'))
    print(f"  Data files: {len(data_paths)}")

    dataset = ConceptTextDataset(tokenizer, [str(p) for p in data_paths],
                                  limit=limit)
    print(f"  Training pairs: {len(dataset):,}")

    if len(dataset) == 0:
        print("No training data found!")
        return

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        collate_fn=ConceptTextDataset.collate,
                        num_workers=0, pin_memory=True)

    # Build model
    device = torch.device('cpu')  # CPU-native, no GPU required
    model = ConceptDecoder(
        vocab_size=tokenizer.vocab_size,
        d_model=256, n_heads=4, n_layers=4, d_ff=512,
        max_len=128, dropout=0.1,
    ).to(device)
    print(f"  Model: {model.count_params():,} parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Training loop
    print(f"\nTraining for {epochs} epochs...")
    best_loss = float('inf')

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        n_batches = 0
        t0 = time.time()

        for concept_ids, target_ids in loader:
            concept_ids = concept_ids.to(device)
            target_ids = target_ids.to(device)

            # Teacher forcing: input = target[:-1], labels = target[1:]
            decoder_input = target_ids[:, :-1]
            labels = target_ids[:, 1:]

            logits = model(concept_ids, decoder_input)

            # Cross-entropy loss (ignore padding)
            loss = F.cross_entropy(
                logits.reshape(-1, tokenizer.vocab_size),
                labels.reshape(-1),
                ignore_index=0,  # PAD
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)
        elapsed = time.time() - t0
        print(f"  Epoch {epoch}/{epochs}: loss={avg_loss:.4f} "
              f"({elapsed:.1f}s, {n_batches} batches)")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(save_path, 'model.pt'))
            tokenizer.save(os.path.join(save_path, 'tokenizer.json'))

    # Save config
    config = {
        'vocab_size': tokenizer.vocab_size,
        'd_model': 256, 'n_heads': 4, 'n_layers': 4, 'd_ff': 512,
        'max_len': 128, 'params': model.count_params(),
        'best_loss': best_loss, 'epochs': epochs,
        'training_pairs': len(dataset),
    }
    with open(os.path.join(save_path, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\nDone. Best loss: {best_loss:.4f}")
    print(f"Saved to {save_path}")

    # Quick generation test
    model.eval()
    test_concepts = ["shakespeare", "wrote", "hamlet"]
    concept_ids = torch.tensor([[tokenizer.word2idx.get(c, tokenizer.UNK)
                                  for c in test_concepts]])
    generated = model.generate(concept_ids, max_len=32, temperature=0.7)
    print(f"\nTest: concepts={test_concepts}")
    print(f"  Generated: {tokenizer.decode(generated)}")

    return model, tokenizer


# --- Inference (for brain_csr_adapter integration) ---

class DecoderEngine:
    """Loads trained decoder for inference. Drop-in for brain's answer generation."""

    def __init__(self, model_path: str = None):
        if model_path is None:
            model_path = os.path.expanduser('~/nexus-brain/decoder')

        config_path = os.path.join(model_path, 'config.json')
        model_pt = os.path.join(model_path, 'model.pt')
        tok_path = os.path.join(model_path, 'tokenizer.json')

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"No decoder at {model_path}")

        with open(config_path) as f:
            config = json.load(f)

        self.tokenizer = BrainTokenizer.load(tok_path)
        self.model = ConceptDecoder(
            vocab_size=config['vocab_size'],
            d_model=config.get('d_model', 256),
            n_heads=config.get('n_heads', 4),
            n_layers=config.get('n_layers', 4),
            d_ff=config.get('d_ff', 512),
            max_len=config.get('max_len', 128),
        )
        self.model.load_state_dict(torch.load(model_pt, map_location='cpu',
                                               weights_only=True))
        self.model.eval()
        self._params = config.get('params', 0)

    def generate(self, concepts: List[str], max_len: int = 64,
                 temperature: float = 0.7) -> str:
        """Generate fluent text from concept list."""
        concept_ids = torch.tensor([[
            self.tokenizer.word2idx.get(c.lower(), self.tokenizer.UNK)
            for c in concepts
        ]])
        token_ids = self.model.generate(concept_ids, max_len=max_len,
                                         temperature=temperature)
        return self.tokenizer.decode(token_ids)

    def stats(self) -> dict:
        return {
            'params': self._params,
            'vocab_size': self.tokenizer.vocab_size,
        }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Train concept-to-text decoder")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--limit", type=int, default=100000)
    parser.add_argument("--db-path", default=os.path.expanduser("~/nexus-brain"))
    parser.add_argument("--data-dir", default=str(Path.home() / "webmind-research" / "data"))
    args = parser.parse_args()

    train_decoder(
        brain_db_path=args.db_path,
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        limit=args.limit,
    )
