"""
Phase-Brain Full Training — ALL Guru Datasets
==============================================

Feeds the phase decoder with everything we have:
- 305K QA pairs (extracted, ready)
- Wikipedia (50K), Natural Questions (50K), StackOverflow (50K)
- HotPotQA (32K), TriviaQA (30K), Dolly (15K), OASST (80K)
- Medicine (18K), Code (100K+), Science, Philosophy, Law, etc.
- TinyStories (streaming, unlimited)

Total: ~1.3M data points + 305K QA pairs

Format: Everything becomes "Q: ... A: ..." for Q&A data,
or raw text for stories/Wikipedia.

Run after Phase 1-2-3 (current GPU training).
Continues from the Phase 3 checkpoint.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import math
import time
import json
import os
import glob
import random
from datasets import load_dataset


def p(msg):
    print(msg, flush=True)


# Reuse the architecture from phase_gpu_train_v2
from phase_gpu_train_v2 import PhaseMultiHeadDecoder, CharTokenizer


# ============================================================
# UNIVERSAL DATA LOADER — handles all Guru dataset formats
# ============================================================

def load_all_datasets(data_dir):
    """Load all JSONL files from data_dir, normalize to Q&A format."""
    all_texts = []

    for path in sorted(glob.glob(os.path.join(data_dir, "*.jsonl"))):
        name = os.path.basename(path).replace(".jsonl", "")
        count = 0

        with open(path) as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                except json.JSONDecodeError:
                    continue

                text = None

                # Format 1: question + answer
                if "question" in item and "answer" in item:
                    q = item["question"].strip()
                    a = str(item["answer"]).strip()
                    if q and a and len(a) > 5:
                        # Strip HTML tags from StackOverflow answers
                        a = a.replace("<p>", "").replace("</p>", "").replace("<br>", " ")
                        text = f"Q: {q}\nA: {a}\n"

                # Format 2: text + answer (Natural Questions style)
                elif "text" in item and "answer" in item:
                    q = item["text"].strip()
                    a = str(item["answer"]).strip()
                    if q and a and len(a) > 3:
                        text = f"Q: {q}\nA: {a}\n"

                # Format 3: text only (conversations, Wikipedia)
                elif "text" in item:
                    t = item["text"].strip()
                    if t and len(t) > 20:
                        text = t + "\n"

                # Format 4: instruction + response (Dolly/Alpaca style)
                elif "instruction" in item and "response" in item:
                    q = item["instruction"].strip()
                    a = item["response"].strip()
                    if q and a:
                        text = f"Q: {q}\nA: {a}\n"

                if text and len(text) > 30:
                    # Truncate very long texts (keep first 500 chars)
                    if len(text) > 500:
                        text = text[:500] + "...\n"
                    all_texts.append(text)
                    count += 1

        if count > 0:
            p(f"  Loaded {count:>7,} from {name}")

    # Also load the pre-extracted QA pairs
    qa_path = os.path.join(os.path.dirname(data_dir), "trained_model", "qa_pairs.jsonl")
    if os.path.exists(qa_path):
        count = 0
        with open(qa_path) as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    q = item.get("question", "").strip()
                    a = item.get("answer", "").strip()
                    if q and a and len(a) > 3:
                        all_texts.append(f"Q: {q}\nA: {a}\n")
                        count += 1
                except json.JSONDecodeError:
                    continue
        p(f"  Loaded {count:>7,} from qa_pairs (pre-extracted)")

    random.shuffle(all_texts)
    p(f"\n  TOTAL: {len(all_texts):,} training examples")
    return all_texts


def batch_generator(texts, tokenizer, seq_len, batch_size):
    """Convert texts to char-level training batches."""
    batch_x, batch_y = [], []

    for text in texts:
        tokens = tokenizer.encode(text)
        if len(tokens) <= seq_len:
            # Pad short sequences
            tokens = tokens + [tokenizer.char_to_ix.get(' ', 0)] * (seq_len + 1 - len(tokens))

        for i in range(0, len(tokens) - seq_len, seq_len):
            chunk = tokens[i:i + seq_len + 1]
            if len(chunk) == seq_len + 1:
                batch_x.append(chunk[:-1])
                batch_y.append(chunk[1:])

                if len(batch_x) == batch_size:
                    yield torch.tensor(batch_x), torch.tensor(batch_y)
                    batch_x, batch_y = [], []


# ============================================================
# TRAINING
# ============================================================

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    p(f"Device: {device}")

    tokenizer = CharTokenizer()
    vocab_size = tokenizer.vocab_size

    # Config
    embed_dim = 512
    num_heads = 8
    seq_len = 256
    batch_size = 64

    # Load model — continue from Phase 3 checkpoint if available
    model = PhaseMultiHeadDecoder(
        vocab_size=vocab_size, embed_dim=embed_dim,
        num_heads=num_heads, max_seq_len=seq_len
    ).to(device)

    checkpoint_path = "phase_decoder_final.pth"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        p(f"Loaded checkpoint from {checkpoint_path}")
    else:
        p("No checkpoint found — training from scratch")

    total_params = sum(pp.numel() for pp in model.parameters())
    p(f"Model: {total_params:,} params")

    # Load ALL datasets
    p(f"\n{'='*60}")
    p(f"  Loading ALL Guru datasets")
    p(f"{'='*60}")

    data_dir = os.path.expanduser("~/webmind-research/data")
    all_texts = load_all_datasets(data_dir)

    # Training
    p(f"\n{'='*60}")
    p(f"  PHASE 4: Full Guru Training")
    p(f"{'='*60}")

    epochs = 3  # 3 passes over all data
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        random.shuffle(all_texts)
        optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)

        # Count batches for scheduler
        est_batches = sum(max(1, len(tokenizer.encode(t)) // seq_len) for t in all_texts[:1000]) * len(all_texts) // 1000 // batch_size
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=1e-3, total_steps=max(est_batches, 1000)
        )

        model.train()
        step = 0
        epoch_loss = 0
        t_start = time.time()

        for x, y in batch_generator(all_texts, tokenizer, seq_len, batch_size):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits.view(-1, vocab_size), y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            try:
                scheduler.step()
            except Exception:
                pass  # scheduler might exhaust steps

            epoch_loss += loss.item()
            step += 1

            if step % 500 == 0:
                avg = epoch_loss / step
                ppl = math.exp(min(avg, 20))
                elapsed = time.time() - t_start
                p(f"  Epoch {epoch} Step {step:6d} | Loss {avg:.4f} | PPL {ppl:.1f} | {elapsed:.0f}s")

            if step % 5000 == 0:
                model.eval()
                prompts = [
                    "Q: What is gravity?\nA:",
                    "Q: Who invented the telephone?\nA:",
                    "Q: What is Python used for?\nA:",
                ]
                for prompt in prompts:
                    seed = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long).to(device)
                    gen = model.generate(seed, max_new_tokens=100)
                    text = tokenizer.decode(gen[0].cpu().tolist())
                    p(f"  >>> {text[:150]}")
                model.train()

        avg = epoch_loss / max(step, 1)
        p(f"\n  Epoch {epoch} complete: {step} steps, avg loss {avg:.4f}, PPL {math.exp(min(avg, 20)):.1f}")

        # Save after each epoch
        save_path = f"phase_guru_epoch{epoch}.pth"
        torch.save({
            "model_state_dict": model.state_dict(),
            "config": {"vocab_size": vocab_size, "embed_dim": embed_dim, "num_heads": num_heads, "max_seq_len": seq_len},
            "char_to_ix": tokenizer.char_to_ix,
            "ix_to_char": tokenizer.ix_to_char,
            "epoch": epoch,
            "avg_loss": avg,
        }, save_path)
        p(f"  Saved {save_path}")

    # Final generation samples
    p(f"\n{'='*60}")
    p(f"  Final Generation Samples")
    p(f"{'='*60}")
    model.eval()
    test_prompts = [
        "Q: What is gravity?\nA:",
        "Q: Who discovered penicillin?\nA:",
        "Q: What is machine learning?\nA:",
        "Q: What causes rain?\nA:",
        "Q: What is the capital of France?\nA:",
        "Q: How does photosynthesis work?\nA:",
        "Q: What is a neural network?\nA:",
        "Q: Explain the theory of relativity.\nA:",
    ]
    for prompt in test_prompts:
        seed = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long).to(device)
        gen = model.generate(seed, max_new_tokens=120)
        text = tokenizer.decode(gen[0].cpu().tolist())
        p(f"\n  {text[:200]}")

    # Phase ablation
    p(f"\n{'='*60}")
    p(f"  Phase Ablation")
    p(f"{'='*60}")
    test_batch = next(batch_generator(all_texts[:100], tokenizer, seq_len, batch_size))
    tx, ty = test_batch[0].to(device), test_batch[1].to(device)
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

    p(f"\nDone. Final model: phase_guru_epoch{epochs-1}.pth")


if __name__ == "__main__":
    train()
