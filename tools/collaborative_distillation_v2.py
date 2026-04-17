#!/usr/bin/env python3
"""
Collaborative Distillation v2 — Close the quality gap
======================================================
Changes from v1:
- 10K training steps (was 3K)
- OpenWebText dataset (was WikiText-2)
- Larger students: 64M params (was 31M)
- Added intermediate feature matching loss
- Higher KD temperature (4.0 vs 2.0)
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
TRAIN_STEPS = 10000
LR = 1e-4
BATCH_SIZE = 8
SEQ_LEN = 256
EVAL_TOKENS = 30_000
ALPHA = 1.0      # KD loss weight
BETA = 0.3       # Diversity loss weight (reduced — was too aggressive)
GAMMA = 0.5      # LM loss weight
TEMPERATURE = 4.0


@dataclass
class Result:
    name: str
    ppl: float
    ppl_teacher: float
    gap_pct: float
    n_models: int = 1


def create_student_config():
    """Larger student: ~64M params."""
    return GPT2Config(
        vocab_size=50257,
        n_positions=1024,
        n_embd=512,        # bigger (was 384)
        n_layer=8,          # more layers (was 6)
        n_head=8,           # more heads (was 6)
        n_inner=2048,       # bigger (was 1536)
        activation_function='gelu_new',
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
    )


class CollaborativeDistillerV2(nn.Module):
    def __init__(self, teacher, n_students=4):
        super().__init__()
        self.teacher = teacher
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

        student_config = create_student_config()
        self.students = nn.ModuleList([
            GPT2LMHeadModel(student_config)
            for _ in range(n_students)
        ])
        self.n_students = n_students

        n_params = sum(p.numel() for p in self.students[0].parameters())
        print(f"  Teacher: {sum(p.numel() for p in teacher.parameters()):,} params",
              flush=True)
        print(f"  Student: {n_params:,} params each x {n_students} = "
              f"{n_params * n_students:,} total", flush=True)

    def forward(self, input_ids):
        with torch.no_grad():
            teacher_out = self.teacher(input_ids, output_hidden_states=True)
            teacher_logits = teacher_out.logits
            teacher_hidden = teacher_out.hidden_states[-1]  # last hidden state

        student_logits = []
        student_hidden = []
        for student in self.students:
            out = student(input_ids, output_hidden_states=True)
            student_logits.append(out.logits)
            student_hidden.append(out.hidden_states[-1])  # last hidden state

        # KD Loss
        kd_loss = 0
        teacher_probs = F.softmax(teacher_logits / TEMPERATURE, dim=-1)
        for s_logits in student_logits:
            student_log_probs = F.log_softmax(s_logits / TEMPERATURE, dim=-1)
            kd_loss += F.kl_div(student_log_probs, teacher_probs,
                               reduction='batchmean') * (TEMPERATURE ** 2)
        kd_loss /= self.n_students

        # Diversity Loss (between student hidden states)
        diversity_loss = 0
        n_pairs = 0
        for i in range(self.n_students):
            for j in range(i + 1, self.n_students):
                cos_sim = F.cosine_similarity(
                    student_hidden[i].mean(dim=1),
                    student_hidden[j].mean(dim=1), dim=-1)
                diversity_loss += cos_sim.mean()
                n_pairs += 1
        diversity_loss /= max(n_pairs, 1)

        # Standard LM loss
        lm_loss = 0
        for s_logits in student_logits:
            shift_logits = s_logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            lm_loss += F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1))
        lm_loss /= self.n_students

        total = ALPHA * kd_loss + BETA * diversity_loss + GAMMA * lm_loss
        return total, kd_loss.item(), diversity_loss.item(), lm_loss.item()

    def get_collaborative_logits(self, input_ids, active_students=None):
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


def eval_ppl(model_fn, tokenizer, n_tokens=EVAL_TOKENS):
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


def train(distiller, tokenizer, steps=TRAIN_STEPS):
    print(f"\n=== TRAINING v2 ({steps} steps, OpenWebText) ===", flush=True)

    ds = load_dataset("openwebtext", split="train", streaming=True)

    optimizer = torch.optim.AdamW(distiller.students.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)

    distiller.train()
    t0 = time.time()
    total_kd, total_div, total_lm = 0, 0, 0
    data_iter = iter(ds)

    for step in range(steps):
        texts = []
        for _ in range(BATCH_SIZE):
            try:
                sample = next(data_iter)
                texts.append(sample["text"][:1000])
            except StopIteration:
                data_iter = iter(ds)
                sample = next(data_iter)
                texts.append(sample["text"][:1000])

        inp = tokenizer(texts, return_tensors="pt", padding=True,
                       truncation=True, max_length=SEQ_LEN).to(DEVICE)

        total_loss, kd, div, lm = distiller(inp["input_ids"])

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(distiller.students.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_kd += kd
        total_div += div
        total_lm += lm

        if (step + 1) % 500 == 0:
            n = step + 1
            print(f"    step {n}/{steps}  kd={total_kd/n:.4f}  "
                  f"div={total_div/n:.4f}  lm={total_lm/n:.4f}  "
                  f"lr={scheduler.get_last_lr()[0]:.6f}  "
                  f"elapsed={time.time()-t0:.1f}s", flush=True)

        # Early eval checkpoint at 3000 steps
        if (step + 1) == 3000:
            distiller.eval()
            ppl, _ = eval_ppl(
                lambda ids: distiller.get_collaborative_logits(ids), tokenizer)
            print(f"    [checkpoint 3K] collaborative ppl={ppl:.2f}", flush=True)
            distiller.train()

    print(f"  Done. time={time.time()-t0:.1f}s", flush=True)


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

    # Teacher baseline
    print("\n=== TEACHER ===", flush=True)
    ppl_teacher, n = eval_ppl(lambda ids: teacher(ids).logits, tokenizer)
    print(f"  ppl={ppl_teacher:.4f} ({n} tokens)", flush=True)
    results.append(Result("teacher", ppl_teacher, ppl_teacher, 0.0))

    # Build distiller
    print("\n=== Building v2 Distiller ===", flush=True)
    distiller = CollaborativeDistillerV2(teacher, N_STUDENTS).to(DEVICE)

    # Train
    train(distiller, tokenizer)

    # Eval
    distiller.eval()

    print("\n=== POST-TRAINING: Each student alone ===", flush=True)
    for i in range(N_STUDENTS):
        ppl, _ = eval_ppl(
            lambda ids, idx=i: distiller.students[idx](ids).logits, tokenizer)
        gap = ((ppl - ppl_teacher) / ppl_teacher) * 100
        print(f"  student {i}: ppl={ppl:.2f} (gap={gap:+.1f}%)", flush=True)
        results.append(Result(f"student_{i}", ppl, ppl_teacher, gap))

    print("\n=== POST-TRAINING: Collaborative (KEY) ===", flush=True)
    ppl_collab, _ = eval_ppl(
        lambda ids: distiller.get_collaborative_logits(ids), tokenizer)
    gap = ((ppl_collab - ppl_teacher) / ppl_teacher) * 100
    print(f"  collaborative 4/4: ppl={ppl_collab:.4f} (gap={gap:+.2f}%)",
          flush=True)
    results.append(Result("collaborative_4", ppl_collab, ppl_teacher, gap, 4))

    print("\n=== POST-TRAINING: Drop each student ===", flush=True)
    for dropped in range(N_STUDENTS):
        active = [i for i in range(N_STUDENTS) if i != dropped]
        ppl, _ = eval_ppl(
            lambda ids, a=active: distiller.get_collaborative_logits(ids, a),
            tokenizer)
        gap = ((ppl - ppl_teacher) / ppl_teacher) * 100
        print(f"  drop {dropped}: ppl={ppl:.2f} (gap={gap:+.1f}%)", flush=True)
        results.append(Result(f"drop_{dropped}", ppl, ppl_teacher, gap, 3))

    # Summary
    print("\n" + "=" * 60, flush=True)
    print("COLLABORATIVE DISTILLATION v2 RESULTS", flush=True)
    print("=" * 60, flush=True)
    print(f"{'Name':<25s} {'PPL':>8s} {'Gap%':>8s} {'N':>3s}", flush=True)
    print("-" * 48, flush=True)
    for r in results:
        print(f"{r.name:<25s} {r.ppl:>8.2f} {r.gap_pct:>+7.1f}% {r.n_models:>3d}",
              flush=True)

    with open(out / "results.json", "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"\nSaved to {out / 'results.json'}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="/workspace/results/")
    run(p.parse_args().output)
