"""
Single-Step Phase Interference on MNIST
========================================

Finding from convergence analysis: information is present from step 1.
A fresh linear probe on step-1 states gets 82%.
The 4-step loop just rotates the basis — it doesn't add information.

This experiment: train with max_steps=1.
If the readout learns to decode step-1 directly, we should match or beat 87%.
Then scale to full MNIST.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import math
import time
import json
import os
import sys
import numpy as np
from datasets import load_dataset

sys.path.insert(0, os.path.dirname(__file__))
from phase_mnist import PhaseMNIST, p


def load_mnist_patches(max_train, max_test):
    """Vectorized MNIST loader."""
    ds = load_dataset("ylecun/mnist")

    def process_split(split, max_n):
        imgs = np.stack([np.array(split[i]["image"]) for i in range(min(max_n, len(split)))])
        labels = np.array([split[i]["label"] for i in range(min(max_n, len(split)))])
        imgs = imgs.astype(np.float32) / 255.0
        N = imgs.shape[0]
        patches = imgs.reshape(N, 4, 7, 4, 7).transpose(0, 1, 3, 2, 4).reshape(N, 16, 49)
        return torch.tensor(patches), torch.tensor(labels, dtype=torch.long)

    t0 = time.time()
    X_train, y_train = process_split(ds["train"], max_train)
    X_test, y_test = process_split(ds["test"], max_test)
    p(f"  Data: {X_train.shape[0]} train, {X_test.shape[0]} test ({time.time()-t0:.1f}s)")
    return X_train, y_train, X_test, y_test


def train_and_evaluate(max_train, max_test, max_steps, embed_dim, epochs, label):
    """Train a phase model and return results."""
    p(f"\n{'='*60}")
    p(f"  {label}")
    p(f"{'='*60}")

    X_train, y_train, X_test, y_test = load_mnist_patches(max_train, max_test)

    model = PhaseMNIST(
        patch_pixels=49, num_patches=16,
        embed_dim=embed_dim, max_steps=max_steps, tol=1e-3,
    )
    total_params = sum(pp.numel() for pp in model.parameters())
    p(f"  Params: {total_params:,}  |  steps={max_steps}  |  embed={embed_dim}")

    batch_size = 64
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    steps_per_epoch = (X_train.shape[0] + batch_size - 1) // batch_size
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=5e-3,
        steps_per_epoch=steps_per_epoch, epochs=epochs
    )
    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    best_epoch = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        perm = torch.randperm(X_train.shape[0])
        for i in range(0, X_train.shape[0], batch_size):
            idx = perm[i:i+batch_size]
            bx, by = X_train[idx], y_train[idx]

            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / steps_per_epoch
        elapsed = time.time() - t0

        model.eval()
        with torch.no_grad():
            test_logits = model(X_test)
            pred = torch.argmax(test_logits, dim=1)
            acc = (pred == y_test).float().mean().item()
            if acc > best_acc:
                best_acc = acc
                best_epoch = epoch

        if epoch % 5 == 0 or epoch == epochs - 1:
            p(f"  Epoch {epoch:2d} | Loss {avg_loss:.4f} | Acc {acc:.2%} | {elapsed:.1f}s")

    # Phase ablation
    with torch.no_grad():
        saved = [model.q_rot.data.clone(), model.k_rot.data.clone(), model.v_rot.data.clone()]
        model.q_rot.data.zero_()
        model.k_rot.data.zero_()
        model.v_rot.data.zero_()
        ablated_logits = model(X_test)
        ablated_acc = (torch.argmax(ablated_logits, dim=1) == y_test).float().mean().item()
        model.q_rot.data.copy_(saved[0])
        model.k_rot.data.copy_(saved[1])
        model.v_rot.data.copy_(saved[2])

    p(f"\n  RESULT: {best_acc:.2%} (epoch {best_epoch})")
    p(f"  Ablated (no phase): {ablated_acc:.2%}")
    p(f"  Phase contribution: {best_acc - ablated_acc:+.2%}")

    return {
        "label": label,
        "best_acc": best_acc,
        "best_epoch": best_epoch,
        "ablated_acc": ablated_acc,
        "params": total_params,
        "max_steps": max_steps,
    }


def main():
    p("=" * 60)
    p("  Phase Interference — Step Count Comparison")
    p("=" * 60)

    results = []

    # --- Experiment 1: 1 step vs 4 steps on 2K samples ---
    results.append(train_and_evaluate(
        max_train=2000, max_test=500, max_steps=1,
        embed_dim=64, epochs=30,
        label="1-STEP, 2K train, embed=64"
    ))

    results.append(train_and_evaluate(
        max_train=2000, max_test=500, max_steps=2,
        embed_dim=64, epochs=30,
        label="2-STEP, 2K train, embed=64"
    ))

    results.append(train_and_evaluate(
        max_train=2000, max_test=500, max_steps=4,
        embed_dim=64, epochs=30,
        label="4-STEP, 2K train, embed=64 (baseline)"
    ))

    # --- Experiment 2: Scale to 10K with best step count ---
    results.append(train_and_evaluate(
        max_train=10000, max_test=2000, max_steps=1,
        embed_dim=64, epochs=30,
        label="1-STEP, 10K train, embed=64"
    ))

    results.append(train_and_evaluate(
        max_train=10000, max_test=2000, max_steps=2,
        embed_dim=64, epochs=30,
        label="2-STEP, 10K train, embed=64"
    ))

    # --- Summary ---
    p(f"\n{'='*60}")
    p("  SUMMARY")
    p(f"{'='*60}")
    p(f"  {'Config':<35} {'Acc':>6} {'Ablated':>8} {'Phase Δ':>8} {'Params':>8}")
    p(f"  {'-'*35} {'-'*6} {'-'*8} {'-'*8} {'-'*8}")
    for r in results:
        p(f"  {r['label']:<35} {r['best_acc']:5.1%} {r['ablated_acc']:7.1%} "
          f"{r['best_acc']-r['ablated_acc']:+7.1%} {r['params']:>8,}")

    # Save
    results_path = os.path.join(os.path.dirname(__file__), "phase_mnist_step_comparison.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    p(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
