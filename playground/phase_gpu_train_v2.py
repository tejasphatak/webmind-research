"""
Multi-Head Phase Decoder — GPU Training
========================================

Targets:
1. PPL < 3.0 on char-level language (TinyStories)
2. Q&A capability (Dolly dataset — question → answer generation)

Architecture: 8-head phase interference, embed=512, single step.
Gemma 4 uses 8 heads — we match the head count.

Training plan:
  Phase 1: TinyStories (language structure) — 20K steps
  Phase 2: Dolly Q&A (question answering) — 10K steps
  Phase 3: Interleaved (both) — 10K steps

Hardware: RTX 3090 (24GB VRAM)
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
# ARCHITECTURE (same as phase_multihead.py but self-contained for GPU)
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


class PhaseMultiHeadDecoder(nn.Module):
    """8-head phase interference decoder for char-level language + Q&A."""

    def __init__(self, vocab_size=97, embed_dim=512, num_heads=8, max_seq_len=256):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.max_seq_len = max_seq_len

        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)

        # Positional phase
        pos = torch.arange(max_seq_len).unsqueeze(1).float()
        dim = torch.arange(embed_dim).unsqueeze(0).float()
        freq = 1.0 / (10000 ** (dim / embed_dim))
        self.register_buffer('pos_phase', pos * freq * math.pi)

        # Multi-head phase rotations
        self.q_rot = nn.Parameter(torch.empty(num_heads, self.head_dim).uniform_(-math.pi, math.pi))
        self.k_rot = nn.Parameter(torch.empty(num_heads, self.head_dim).uniform_(-math.pi, math.pi))
        self.v_rot = nn.Parameter(torch.empty(num_heads, self.head_dim).uniform_(-math.pi, math.pi))

        # Complex resonance (feed-forward)
        w_init = torch.empty(embed_dim, embed_dim)
        nn.init.orthogonal_(w_init)
        self.ff_real = nn.Parameter(w_init)
        self.ff_imag = nn.Parameter(torch.empty(embed_dim, embed_dim).uniform_(-0.01, 0.01))

        self.norm = ComplexNorm()

        # Dual-basis readout
        self.readout_real = nn.Linear(embed_dim, vocab_size)
        self.readout_imag = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        B, S = x.shape
        device = x.device

        # Wavefunction encoding
        mag = torch.tanh(self.token_embedding(x))
        state = mag.to(torch.complex64) * torch.exp(1j * self.pos_phase[:S].unsqueeze(0))

        # Reshape for multi-head: (B, S, H, d_h)
        state_heads = state.view(B, S, self.num_heads, self.head_dim)

        # Phase rotations per head
        q = state_heads * torch.exp(1j * self.q_rot)
        k = state_heads * torch.exp(1j * self.k_rot)
        v = state_heads * torch.exp(1j * self.v_rot)

        # (B, H, S, d_h) for batched matmul
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # Per-head interference with causal mask
        interference = torch.matmul(q, k.conj().transpose(-1, -2))
        attn_logits = interference.real / math.sqrt(self.head_dim)

        causal_mask = torch.tril(torch.ones(S, S, device=device))
        attn_logits = attn_logits.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0) == 0, float('-inf'))
        attn_weights = torch.softmax(attn_logits * 8.0, dim=-1)

        # Superposition
        attn_out = torch.matmul(attn_weights.to(torch.complex64), v)
        attn_out = attn_out.permute(0, 2, 1, 3).contiguous().view(B, S, self.embed_dim)

        # Residual + resonance
        state = state + attn_out
        ff_weights = torch.complex(self.ff_real, self.ff_imag)
        state = torch.matmul(state, ff_weights)
        state = self.norm(state)

        # Measurement
        return self.readout_real(state.real) + self.readout_imag(state.imag)

    @torch.no_grad()
    def generate(self, seed, max_new_tokens, temperature=0.8, top_p=0.9):
        for _ in range(max_new_tokens):
            input_seq = seed[:, -self.max_seq_len:]
            logits = self.forward(input_seq)
            next_logits = logits[:, -1, :] / temperature

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
# CHAR TOKENIZER
# ============================================================

class CharTokenizer:
    def __init__(self):
        chars = [chr(i) for i in range(32, 127)] + ['\n', '\t']
        self.chars = sorted(set(chars))
        self.char_to_ix = {ch: i for i, ch in enumerate(self.chars)}
        self.ix_to_char = {i: ch for i, ch in enumerate(self.chars)}
        self.vocab_size = len(self.chars)

    def encode(self, text):
        return [self.char_to_ix.get(ch, self.char_to_ix[' ']) for ch in text]

    def decode(self, indices):
        return ''.join(self.ix_to_char.get(i, '?') for i in indices)


# ============================================================
# DATA STREAMS
# ============================================================

def tinystories_stream(tokenizer, seq_len, batch_size):
    """Char-level TinyStories batches."""
    dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    batch_x, batch_y = [], []

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
                    yield torch.tensor(batch_x), torch.tensor(batch_y), "stories"
                    batch_x, batch_y = [], []


def dolly_qa_stream(tokenizer, seq_len, batch_size):
    """Char-level Q&A from Dolly dataset."""
    dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
    batch_x, batch_y = [], []

    for item in dataset:
        instruction = item["instruction"].strip()
        response = item["response"].strip()
        if not instruction or not response:
            continue

        # Format as Q&A
        text = f"Q: {instruction}\nA: {response}\n"
        tokens = tokenizer.encode(text)
        if len(tokens) <= seq_len:
            # Pad short sequences
            tokens = tokens + [tokenizer.char_to_ix[' ']] * (seq_len + 1 - len(tokens))

        for i in range(0, len(tokens) - seq_len, seq_len):
            chunk = tokens[i:i + seq_len + 1]
            if len(chunk) == seq_len + 1:
                batch_x.append(chunk[:-1])
                batch_y.append(chunk[1:])
                if len(batch_x) == batch_size:
                    yield torch.tensor(batch_x), torch.tensor(batch_y), "qa"
                    batch_x, batch_y = [], []


def interleaved_stream(tokenizer, seq_len, batch_size):
    """Alternate between stories and Q&A."""
    stories = tinystories_stream(tokenizer, seq_len, batch_size)
    qa = dolly_qa_stream(tokenizer, seq_len, batch_size)

    use_stories = True
    while True:
        try:
            if use_stories:
                yield next(stories)
            else:
                yield next(qa)
        except StopIteration:
            if use_stories:
                stories = tinystories_stream(tokenizer, seq_len, batch_size)
            else:
                qa = dolly_qa_stream(tokenizer, seq_len, batch_size)
        use_stories = not use_stories


# ============================================================
# TRAINING
# ============================================================

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    p(f"Device: {device}")

    tokenizer = CharTokenizer()
    vocab_size = tokenizer.vocab_size

    # Config — matching Gemma 4's head count
    embed_dim = 512
    num_heads = 8
    seq_len = 256
    batch_size = 64

    model = PhaseMultiHeadDecoder(
        vocab_size=vocab_size, embed_dim=embed_dim,
        num_heads=num_heads, max_seq_len=seq_len
    ).to(device)

    total_params = sum(pp.numel() for pp in model.parameters())
    attn_params = model.q_rot.numel() + model.k_rot.numel() + model.v_rot.numel()
    p(f"Model: {total_params:,} params ({total_params*4/1024/1024:.1f} MB)")
    p(f"  Attention: {attn_params:,} phase rotation params")
    p(f"  Heads: {num_heads}, Head dim: {embed_dim//num_heads}, Embed: {embed_dim}")
    p(f"  Seq len: {seq_len}, Batch: {batch_size}")

    # --- PHASE 1: TinyStories (language) ---
    p(f"\n{'='*60}")
    p(f"  PHASE 1: TinyStories (language structure)")
    p(f"{'='*60}")

    total_steps_p1 = 20000
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=5e-3, total_steps=total_steps_p1)
    criterion = nn.CrossEntropyLoss()

    log = {"phase1": [], "phase2": [], "phase3": []}
    model.train()
    t_start = time.time()

    for step, (x, y, src) in enumerate(tinystories_stream(tokenizer, seq_len, batch_size)):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits.view(-1, vocab_size), y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        lv = loss.item()
        log["phase1"].append({"step": step, "loss": round(lv, 4)})

        if step % 200 == 0:
            ppl = math.exp(min(lv, 20))
            elapsed = time.time() - t_start
            p(f"  Step {step:5d} | Loss {lv:.4f} | PPL {ppl:.1f} | {elapsed:.0f}s")

        if step > 0 and step % 2000 == 0:
            model.eval()
            prompt = "Once upon a time"
            seed = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long).to(device)
            gen = model.generate(seed, max_new_tokens=80)
            p(f"  >>> {tokenizer.decode(gen[0].cpu().tolist())}")
            model.train()

        if step >= total_steps_p1:
            break

    # Save Phase 1
    save_path = "phase_decoder_p1.pth"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {"vocab_size": vocab_size, "embed_dim": embed_dim, "num_heads": num_heads, "max_seq_len": seq_len},
        "char_to_ix": tokenizer.char_to_ix,
        "ix_to_char": tokenizer.ix_to_char,
    }, save_path)
    p(f"\nPhase 1 saved to {save_path}")

    # --- PHASE 2: Dolly Q&A ---
    p(f"\n{'='*60}")
    p(f"  PHASE 2: Dolly Q&A (question answering)")
    p(f"{'='*60}")

    total_steps_p2 = 10000
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=2e-3, total_steps=total_steps_p2)
    t_start = time.time()

    for step, (x, y, src) in enumerate(dolly_qa_stream(tokenizer, seq_len, batch_size)):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits.view(-1, vocab_size), y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        lv = loss.item()
        log["phase2"].append({"step": step, "loss": round(lv, 4)})

        if step % 200 == 0:
            ppl = math.exp(min(lv, 20))
            elapsed = time.time() - t_start
            p(f"  Step {step:5d} | Loss {lv:.4f} | PPL {ppl:.1f} | {elapsed:.0f}s")

        if step > 0 and step % 2000 == 0:
            model.eval()
            prompt = "Q: What is gravity?\nA:"
            seed = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long).to(device)
            gen = model.generate(seed, max_new_tokens=80)
            p(f"  >>> {tokenizer.decode(gen[0].cpu().tolist())}")
            model.train()

        if step >= total_steps_p2:
            break

    # Save Phase 2
    save_path = "phase_decoder_p2.pth"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {"vocab_size": vocab_size, "embed_dim": embed_dim, "num_heads": num_heads, "max_seq_len": seq_len},
        "char_to_ix": tokenizer.char_to_ix,
        "ix_to_char": tokenizer.ix_to_char,
    }, save_path)
    p(f"\nPhase 2 saved to {save_path}")

    # --- PHASE 3: Interleaved ---
    p(f"\n{'='*60}")
    p(f"  PHASE 3: Interleaved (stories + Q&A)")
    p(f"{'='*60}")

    total_steps_p3 = 10000
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, total_steps=total_steps_p3)
    t_start = time.time()

    for step, (x, y, src) in enumerate(interleaved_stream(tokenizer, seq_len, batch_size)):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits.view(-1, vocab_size), y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        lv = loss.item()
        log["phase3"].append({"step": step, "loss": round(lv, 4), "src": src})

        if step % 200 == 0:
            ppl = math.exp(min(lv, 20))
            elapsed = time.time() - t_start
            p(f"  Step {step:5d} | Loss {lv:.4f} | PPL {ppl:.1f} | src={src} | {elapsed:.0f}s")

        if step > 0 and step % 2000 == 0:
            model.eval()
            for prompt in ["Once upon a time", "Q: What is gravity?\nA:"]:
                seed = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long).to(device)
                gen = model.generate(seed, max_new_tokens=80)
                p(f"  >>> {tokenizer.decode(gen[0].cpu().tolist())}")
            model.train()

        if step >= total_steps_p3:
            break

    # Final save
    save_path = "phase_decoder_final.pth"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {"vocab_size": vocab_size, "embed_dim": embed_dim, "num_heads": num_heads, "max_seq_len": seq_len},
        "char_to_ix": tokenizer.char_to_ix,
        "ix_to_char": tokenizer.ix_to_char,
    }, save_path)
    p(f"\nFinal model saved to {save_path}")

    # Phase ablation
    p(f"\n{'='*60}")
    p(f"  Phase Ablation")
    p(f"{'='*60}")
    model.eval()
    # Get a test batch
    test_gen = tinystories_stream(tokenizer, seq_len, batch_size)
    tx, ty, _ = next(test_gen)
    tx, ty = tx.to(device), ty.to(device)

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

    p(f"  Normal:  Loss {normal_loss:.4f} (PPL {math.exp(normal_loss):.1f})")
    p(f"  Ablated: Loss {ablated_loss:.4f} (PPL {math.exp(ablated_loss):.1f})")
    p(f"  Phase impact: {ablated_loss - normal_loss:+.4f}")

    # Final generation
    p(f"\n{'='*60}")
    p(f"  Final Generation Samples")
    p(f"{'='*60}")
    prompts = [
        "Once upon a time, there was a",
        "The little girl was very happy because",
        "Q: What is the meaning of life?\nA:",
        "Q: Who invented the telephone?\nA:",
        "Q: Why is the sky blue?\nA:",
    ]
    for prompt in prompts:
        seed = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long).to(device)
        gen = model.generate(seed, max_new_tokens=100)
        text = tokenizer.decode(gen[0].cpu().tolist())
        p(f"\n  [{prompt[:40]}...]")
        p(f"  {text}")

    # Save full log
    with open("phase_decoder_gpu_log.json", "w") as f:
        json.dump(log, f)
    p(f"\nTraining log saved to phase_decoder_gpu_log.json")
    p(f"Total params: {total_params:,} ({total_params*4/1024/1024:.1f} MB)")


if __name__ == "__main__":
    train()
