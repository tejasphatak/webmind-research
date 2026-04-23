"""
Phase-Interference Decoder: Can it learn language?
===================================================

Finding from MNIST: one round of phase interference classifies at 95%
with 12K params. Phase is load-bearing (72% contribution).

Now: autoregressive next-token prediction on TinyStories.
Single interference step. GPT-2 tokenizer. Causal mask.

The question from quantum physics perspective:
  Can wave interference patterns encode sequential dependencies
  (language structure) the way they encode spatial patterns (digit shapes)?

In MNIST, interference learned "all patches attend to center."
In language, it needs to learn "each token attends to relevant context."
That's a harder interference pattern — position-dependent, asymmetric (causal).
"""

import torch
import torch.nn as nn
import torch.optim as optim
import math
import time
import json
import os
from datasets import load_dataset


def p(msg):
    print(msg, flush=True)


# ============================================================
# ARCHITECTURE — Single-step phase interference decoder
# ============================================================

class ComplexNorm(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        mag = torch.abs(x)
        mean_mag = mag.mean(dim=-1, keepdim=True)
        std_mag = mag.std(dim=-1, keepdim=True)
        norm_mag = (mag - mean_mag) / (std_mag + self.eps)
        scale = torch.tanh(norm_mag) / (mag + self.eps)
        return x * scale.to(torch.complex64)


class PhaseDecoder(nn.Module):
    """Single-step phase interference for autoregressive language modeling.

    Architecture (one pass):
      tokens → embedding (magnitude) → positional phase → wavefunction
      → phase rotate Q/K/V → causal interference → superposition
      → complex resonance → normalize → measure (dual-basis readout)
    """
    def __init__(self, vocab_size, embed_dim=256, max_seq_len=128):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        # Token embedding → magnitude
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)

        # Positional phase — multi-frequency sinusoidal
        pos = torch.arange(max_seq_len).unsqueeze(1).float()
        dim = torch.arange(embed_dim).unsqueeze(0).float()
        freq = 1.0 / (10000 ** (dim / embed_dim))
        self.register_buffer('pos_phase', pos * freq * math.pi)

        # Phase rotations (Q/K/V) — the learned unitary operators
        self.q_rot = nn.Parameter(torch.empty(embed_dim).uniform_(-math.pi, math.pi))
        self.k_rot = nn.Parameter(torch.empty(embed_dim).uniform_(-math.pi, math.pi))
        self.v_rot = nn.Parameter(torch.empty(embed_dim).uniform_(-math.pi, math.pi))

        # Resonance matrix (complex feed-forward)
        w_init = torch.empty(embed_dim, embed_dim)
        nn.init.orthogonal_(w_init)
        self.ff_real = nn.Parameter(w_init)
        self.ff_imag = nn.Parameter(torch.empty(embed_dim, embed_dim).uniform_(-0.01, 0.01))

        self.norm = ComplexNorm()

        # Dual-basis readout (measurement)
        self.readout_real = nn.Linear(embed_dim, vocab_size)
        self.readout_imag = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        batch_size, seq_len = x.shape

        # 1. Wavefunction encoding
        mag = torch.tanh(self.token_embedding(x))
        state = mag.to(torch.complex64) * torch.exp(1j * self.pos_phase[:seq_len].unsqueeze(0))

        # 2. Phase rotation
        q = state * torch.exp(1j * self.q_rot)
        k = state * torch.exp(1j * self.k_rot)
        v = state * torch.exp(1j * self.v_rot)

        # 3. Causal interference
        # Q·K† = complex inner product, real part = interference
        interference = torch.matmul(q, k.conj().transpose(-1, -2))
        attn_logits = interference.real / math.sqrt(self.embed_dim)

        # Causal mask — can only attend to past (and self)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
        attn_logits = attn_logits.masked_fill(causal_mask.unsqueeze(0) == 0, float('-inf'))
        attn_weights = torch.softmax(attn_logits * 8.0, dim=-1)

        # Weighted superposition
        attn_out = torch.matmul(attn_weights.to(torch.complex64), v)
        state = state + attn_out

        # 4. Resonance
        ff_weights = torch.complex(self.ff_real, self.ff_imag)
        state = torch.matmul(state, ff_weights)
        state = self.norm(state)

        # 5. Measurement
        logits = self.readout_real(state.real) + self.readout_imag(state.imag)
        return logits

    @torch.no_grad()
    def generate(self, seed, max_new_tokens, temperature=0.8, top_p=0.9):
        max_seq = 128  # model's max sequence length
        for _ in range(max_new_tokens):
            # Truncate input to max_seq (sliding window)
            input_seq = seed[:, -max_seq:]
            logits = self.forward(input_seq)
            next_logits = logits[:, -1, :] / temperature

            # Top-p sampling
            sorted_logits, sorted_idx = torch.sort(next_logits, descending=True)
            cumprobs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            remove = cumprobs > top_p
            remove[..., 1:] = remove[..., :-1].clone()
            remove[..., 0] = False
            sorted_logits[remove] = float('-inf')

            probs = torch.softmax(sorted_logits, dim=-1)
            sampled = torch.multinomial(probs, 1)
            next_token = sorted_idx.gather(-1, sampled)

            seed = torch.cat([seed, next_token], dim=1)
        return seed


# ============================================================
# DATA — TinyStories, char-level (no external tokenizer)
# ============================================================

class CharTokenizer:
    """Minimal char-level tokenizer. No dependencies."""
    def __init__(self):
        # Printable ASCII + newline + tab
        chars = [chr(i) for i in range(32, 127)] + ['\n', '\t']
        self.chars = sorted(set(chars))
        self.char_to_ix = {ch: i for i, ch in enumerate(self.chars)}
        self.ix_to_char = {i: ch for i, ch in enumerate(self.chars)}
        self.vocab_size = len(self.chars)

    def encode(self, text):
        return [self.char_to_ix.get(ch, self.char_to_ix[' ']) for ch in text]

    def decode(self, indices):
        return ''.join(self.ix_to_char.get(i, '?') for i in indices)


def make_batches(tokenizer, seq_len, batch_size, max_batches):
    """Stream TinyStories char-by-char and yield (x, y) batches."""
    dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    batch_x, batch_y = [], []
    count = 0

    for example in dataset:
        tokens = tokenizer.encode(example["text"])
        if len(tokens) <= seq_len:
            continue

        for i in range(0, len(tokens) - seq_len, seq_len):
            chunk = tokens[i:i + seq_len + 1]
            if len(chunk) == seq_len + 1:
                batch_x.append(chunk[:-1])
                batch_y.append(chunk[1:])

                if len(batch_x) == batch_size:
                    yield torch.tensor(batch_x), torch.tensor(batch_y)
                    batch_x, batch_y = [], []
                    count += 1
                    if count >= max_batches:
                        return


# ============================================================
# TRAINING
# ============================================================

def train():
    p("=" * 60)
    p("  Phase-Interference Decoder Test (char-level)")
    p("=" * 60)

    # Config
    seq_len = 128       # chars need longer context
    batch_size = 64     # small vocab = can afford bigger batches
    embed_dim = 128     # small vocab doesn't need 256
    total_steps = 2000
    gen_every = 250

    # Char tokenizer — ~95 vocab, zero dependencies
    tokenizer = CharTokenizer()
    vocab_size = tokenizer.vocab_size
    p(f"\n  Char-level tokenizer: {vocab_size} tokens (printable ASCII)")

    # Model
    model = PhaseDecoder(vocab_size=vocab_size, embed_dim=embed_dim, max_seq_len=seq_len)
    total_params = sum(pp.numel() for pp in model.parameters())
    p(f"  Vocab: {vocab_size}  Embed: {embed_dim}  Seq: {seq_len}")
    p(f"  Params: {total_params:,}")
    p(f"  Steps: {total_steps}  (single interference pass)")

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=5e-3, total_steps=total_steps
    )
    criterion = nn.CrossEntropyLoss()

    # Training loop
    p(f"\nTraining on TinyStories (char-level, streaming)...\n")
    log = []
    model.train()
    t_start = time.time()

    for step, (x, y) in enumerate(make_batches(tokenizer, seq_len, batch_size, total_steps)):
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits.view(-1, vocab_size), y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        loss_val = loss.item()
        log.append({"step": step, "loss": round(loss_val, 4)})

        if step % 50 == 0:
            elapsed = time.time() - t_start
            ppl = math.exp(min(loss_val, 20))  # cap for display
            p(f"  Step {step:4d} | Loss {loss_val:.4f} | PPL {ppl:.1f} | {elapsed:.0f}s")

        if step > 0 and step % gen_every == 0:
            model.eval()
            prompt = "Once upon a time"
            seed = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
            generated = model.generate(seed, max_new_tokens=60)
            text = tokenizer.decode(generated[0].tolist())
            p(f"  >>> {text}")
            model.train()

    total_time = time.time() - t_start
    p(f"\nTraining complete in {total_time:.0f}s")

    # Final generation
    p(f"\n{'='*60}")
    p("  Generation Samples")
    p(f"{'='*60}")
    model.eval()
    prompts = ["Once upon a time", "The dog", "She was happy"]
    for prompt in prompts:
        seed = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
        generated = model.generate(seed, max_new_tokens=80, temperature=0.8)
        text = tokenizer.decode(generated[0].tolist())
        p(f"\n  [{prompt}] → {text}")

    # Phase ablation
    p(f"\n{'='*60}")
    p("  Phase Ablation")
    p(f"{'='*60}")

    test_batches = list(make_batches(tokenizer, seq_len, batch_size, 1))
    if test_batches:
        tx, ty = test_batches[0]
        with torch.no_grad():
            normal_loss = criterion(model(tx).view(-1, vocab_size), ty.view(-1)).item()

            saved = [model.q_rot.data.clone(), model.k_rot.data.clone(), model.v_rot.data.clone()]
            model.q_rot.data.zero_()
            model.k_rot.data.zero_()
            model.v_rot.data.zero_()
            ablated_loss = criterion(model(tx).view(-1, vocab_size), ty.view(-1)).item()
            model.q_rot.data.copy_(saved[0])
            model.k_rot.data.copy_(saved[1])
            model.v_rot.data.copy_(saved[2])

        p(f"  Normal loss:  {normal_loss:.4f} (perplexity {math.exp(normal_loss):.1f})")
        p(f"  Ablated loss: {ablated_loss:.4f} (perplexity {math.exp(ablated_loss):.1f})")
        p(f"  Phase impact: {ablated_loss - normal_loss:+.4f}")

    # Save
    results_path = os.path.join(os.path.dirname(__file__), "phase_decoder_results.json")
    with open(results_path, "w") as f:
        json.dump(log, f, indent=2)
    p(f"\nLog saved to {results_path}")

    model_path = os.path.join(os.path.dirname(__file__), "phase_decoder_model.pth")
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {"vocab_size": vocab_size, "embed_dim": embed_dim, "max_seq_len": seq_len},
        "char_to_ix": tokenizer.char_to_ix,
        "ix_to_char": tokenizer.ix_to_char,
    }, model_path)
    p(f"Model saved to {model_path}")


if __name__ == "__main__":
    train()
