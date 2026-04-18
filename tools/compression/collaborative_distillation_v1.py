#!/usr/bin/env python3
"""
Collaborative Distillation with Diversity Regularization — v1
==============================================================
Distill GPT-2 124M (teacher) into 4 × ~31M students simultaneously.
Each student is a COMPLETE model that works independently.
Together they collaborate by averaging logits.

Loss = α * KD_loss + β * Diversity_loss

KD_loss: each student matches teacher's logit distribution
Diversity_loss: maximize cosine distance between students' hidden states
"""

import os, json, time, math, gc, random, argparse, itertools
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer
from datasets import load_dataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_STUDENTS = 4
TRAIN_STEPS = 3000
LR = 3e-4
BATCH_SIZE = 8
SEQ_LEN = 256
EVAL_TOKENS = 30_000
ALPHA = 1.0    # KD loss weight
BETA = 0.5     # Diversity loss weight
TEMPERATURE = 2.0  # KD temperature


@dataclass
class Result:
    name: str
    ppl: float
    ppl_teacher: float
    gap_pct: float
    n_models: int = 1


def create_student_config():
    """Create a small GPT-2 config (~31M params) as student."""
    return GPT2Config(
        vocab_size=50257,
        n_positions=1024,
        n_embd=384,       # 768/2
        n_layer=6,         # 12/2
        n_head=6,          # 12/2
        n_inner=1536,      # 3072/2
        activation_function='gelu_new',
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
    )


class CollaborativeDistiller(nn.Module):
    """Trains N students from a teacher with diversity regularization."""

    def __init__(self, teacher, n_students=4):
        super().__init__()
        self.teacher = teacher
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

        # Create N student models
        student_config = create_student_config()
        self.students = nn.ModuleList([
            GPT2LMHeadModel(student_config)
            for _ in range(n_students)
        ])
        self.n_students = n_students

        n_params = sum(p.numel() for p in self.students[0].parameters())
        print(f"  Teacher: {sum(p.numel() for p in teacher.parameters()):,} params", flush=True)
        print(f"  Student: {n_params:,} params each × {n_students} = "
              f"{n_params * n_students:,} total", flush=True)

    def forward(self, input_ids):
        """Compute KD loss + diversity loss for all students."""
        # Teacher forward (no grad)
        with torch.no_grad():
            teacher_out = self.teacher(input_ids)
            teacher_logits = teacher_out.logits  # [B, S, V]

        # Student forwards
        student_logits = []
        student_hidden = []
        for student in self.students:
            out = student(input_ids)
            student_logits.append(out.logits)
            # Get last hidden state for diversity loss
            # GPT2 hidden states from transformer output
            hidden = student.transformer(input_ids).last_hidden_state
            student_hidden.append(hidden.mean(dim=1))  # [B, hidden] — mean over seq

        # === KD Loss: each student matches teacher ===
        kd_loss = 0
        teacher_probs = F.softmax(teacher_logits / TEMPERATURE, dim=-1)
        for s_logits in student_logits:
            student_log_probs = F.log_softmax(s_logits / TEMPERATURE, dim=-1)
            kd_loss += F.kl_div(student_log_probs, teacher_probs,
                               reduction='batchmean') * (TEMPERATURE ** 2)
        kd_loss /= self.n_students

        # === Diversity Loss: maximize pairwise cosine distance ===
        diversity_loss = 0
        n_pairs = 0
        for i in range(self.n_students):
            for j in range(i + 1, self.n_students):
                # Cosine similarity between hidden states
                cos_sim = F.cosine_similarity(student_hidden[i],
                                               student_hidden[j], dim=-1)
                diversity_loss += cos_sim.mean()  # want this LOW (diverse)
                n_pairs += 1
        diversity_loss /= max(n_pairs, 1)

        # === Standard LM loss for each student (helps convergence) ===
        lm_loss = 0
        for s_logits in student_logits:
            shift_logits = s_logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            lm_loss += F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1))
        lm_loss /= self.n_students

        total_loss = ALPHA * kd_loss + BETA * diversity_loss + 0.5 * lm_loss

        return total_loss, kd_loss.item(), diversity_loss.item(), lm_loss.item()

    def get_collaborative_logits(self, input_ids, active_students=None):
        """Get averaged logits from active students."""
        if active_students is None:
            active_students = list(range(self.n_students))

        logits_sum = None
        for idx in active_students:
            out = self.students[idx](input_ids)
            if logits_sum is None:
                logits_sum = out.logits.float()
            else:
                logits_sum = logits_sum + out.logits.float()

        return logits_sum / len(active_students)


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------
def eval_ppl_model(model_fn, tokenizer, n_tokens=EVAL_TOKENS):
    """Evaluate perplexity using a model function."""
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    total_loss, total_n = 0, 0
    with torch.no_grad():
        for s in ds:
            if total_n >= n_tokens:
                break
            if len(s["text"].strip()) < 20:
                continue
            inp = tokenizer(s["text"][:500], return_tensors="pt",
                           truncation=True, max_length=SEQ_LEN).to(DEVICE)
            if inp["input_ids"].size(1) < 10:
                continue
            logits = model_fn(inp["input_ids"])
            sl = logits[:, :-1, :].float()
            lab = inp["input_ids"][:, 1:]
            loss = F.cross_entropy(sl.reshape(-1, sl.size(-1)),
                                   lab.reshape(-1), reduction="sum")
            total_loss += loss.item()
            total_n += lab.numel()
    avg = total_loss / max(total_n, 1)
    return math.exp(min(avg, 100)), total_n


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(distiller, tokenizer, steps=TRAIN_STEPS):
    print(f"\n=== COLLABORATIVE DISTILLATION ({steps} steps) ===", flush=True)
    print(f"  α={ALPHA} (KD), β={BETA} (diversity), T={TEMPERATURE}", flush=True)

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts = [s["text"] for s in ds if len(s["text"].strip()) > 50]

    optimizer = torch.optim.AdamW(
        [p for p in distiller.students.parameters()],
        lr=LR)

    distiller.train()
    t0 = time.time()
    total_kd, total_div, total_lm = 0, 0, 0
    idx = 0

    for step in range(steps):
        batch = [texts[idx % len(texts)][:1000] for _ in range(BATCH_SIZE)]
        idx += BATCH_SIZE
        inp = tokenizer(batch, return_tensors="pt", padding=True,
                       truncation=True, max_length=SEQ_LEN).to(DEVICE)

        total_loss, kd, div, lm = distiller(inp["input_ids"])

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(distiller.students.parameters(), 1.0)
        optimizer.step()

        total_kd += kd
        total_div += div
        total_lm += lm

        if (step + 1) % 300 == 0:
            n = step + 1
            print(f"    step {n}/{steps}  kd={total_kd/n:.4f}  "
                  f"div={total_div/n:.4f}  lm={total_lm/n:.4f}  "
                  f"elapsed={time.time()-t0:.1f}s", flush=True)

    t = time.time() - t0
    print(f"  Done. time={t:.1f}s", flush=True)
    return t


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run(output_dir):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = []

    print("Loading GPT-2 124M (teacher)...", flush=True)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    teacher = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
    teacher.eval()

    # ---- Teacher baseline ----
    print("\n=== TEACHER (GPT-2 124M) ===", flush=True)
    ppl_teacher, n = eval_ppl_model(
        lambda ids: teacher(ids).logits, tokenizer)
    print(f"  ppl={ppl_teacher:.4f} ({n} tokens)", flush=True)
    results.append(Result("teacher", ppl_teacher, ppl_teacher, 0.0))
    _save(results, out / "results.json")

    # ---- Build distiller ----
    print("\n=== Building Collaborative Distiller ===", flush=True)
    distiller = CollaborativeDistiller(teacher, N_STUDENTS).to(DEVICE)

    # ---- Pre-training: each student alone ----
    print("\n=== PRE-TRAINING: Each student alone (random init) ===", flush=True)
    for i in range(N_STUDENTS):
        ppl, _ = eval_ppl_model(
            lambda ids, idx=i: distiller.students[idx](ids).logits,
            tokenizer)
        gap = ((ppl - ppl_teacher) / ppl_teacher) * 100
        print(f"  student {i}: ppl={ppl:.2f} (gap={gap:+.1f}%)", flush=True)
        results.append(Result(f"pre_student_{i}", ppl, ppl_teacher, gap))

    # ---- Pre-training: collaborative (averaged logits) ----
    print("\n=== PRE-TRAINING: All 4 students collaborative ===", flush=True)
    ppl_collab_pre, _ = eval_ppl_model(
        lambda ids: distiller.get_collaborative_logits(ids), tokenizer)
    gap = ((ppl_collab_pre - ppl_teacher) / ppl_teacher) * 100
    print(f"  collaborative: ppl={ppl_collab_pre:.2f} (gap={gap:+.1f}%)", flush=True)
    results.append(Result("pre_collaborative", ppl_collab_pre, ppl_teacher, gap, N_STUDENTS))
    _save(results, out / "results.json")

    # ---- Train ----
    train(distiller, tokenizer)

    # ---- Post-training: each student alone ----
    print("\n=== POST-TRAINING: Each student alone ===", flush=True)
    distiller.eval()
    for i in range(N_STUDENTS):
        ppl, _ = eval_ppl_model(
            lambda ids, idx=i: distiller.students[idx](ids).logits,
            tokenizer)
        gap = ((ppl - ppl_teacher) / ppl_teacher) * 100
        print(f"  student {i}: ppl={ppl:.2f} (gap={gap:+.1f}%)", flush=True)
        results.append(Result(f"post_student_{i}", ppl, ppl_teacher, gap))

    # ---- Post-training: collaborative (THE KEY TEST) ----
    print("\n=== POST-TRAINING: All 4 collaborative (KEY TEST) ===", flush=True)
    ppl_collab, _ = eval_ppl_model(
        lambda ids: distiller.get_collaborative_logits(ids), tokenizer)
    gap = ((ppl_collab - ppl_teacher) / ppl_teacher) * 100
    print(f"  collaborative 4/4: ppl={ppl_collab:.4f} (gap={gap:+.2f}%)", flush=True)
    results.append(Result("post_collaborative_4", ppl_collab, ppl_teacher, gap, 4))

    # ---- Resilience: drop each student ----
    print("\n=== POST-TRAINING: Drop each student ===", flush=True)
    for dropped in range(N_STUDENTS):
        active = [i for i in range(N_STUDENTS) if i != dropped]
        ppl, _ = eval_ppl_model(
            lambda ids, a=active: distiller.get_collaborative_logits(ids, a),
            tokenizer)
        gap = ((ppl - ppl_teacher) / ppl_teacher) * 100
        print(f"  drop student {dropped}: ppl={ppl:.2f} (gap={gap:+.1f}%)", flush=True)
        results.append(Result(f"post_drop_{dropped}", ppl, ppl_teacher, gap, 3))

    # ---- Summary ----
    print("\n" + "=" * 65, flush=True)
    print("COLLABORATIVE DISTILLATION RESULTS", flush=True)
    print("=" * 65, flush=True)
    print(f"{'Name':<30s} {'PPL':>8s} {'Gap%':>8s} {'Models':>7s}", flush=True)
    print("-" * 55, flush=True)
    for r in results:
        print(f"{r.name:<30s} {r.ppl:>8.2f} {r.gap_pct:>+7.1f}% "
              f"{r.n_models:>7d}", flush=True)

    _save(results, out / "results.json")
    print(f"\nSaved to {out / 'results.json'}", flush=True)


def _save(results, path):
    with open(path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="/workspace/results/")
    run(p.parse_args().output)
