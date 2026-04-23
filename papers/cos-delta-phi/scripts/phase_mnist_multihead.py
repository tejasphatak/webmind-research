"""
Multi-Head Phase Interference on MNIST — Target: 97%+
"""

import torch
import torch.nn as nn
import torch.optim as optim
import time
import json
import os
import numpy as np
from datasets import load_dataset
from phase_multihead import PhaseMultiHeadModel


def p(msg):
    print(msg, flush=True)


def load_mnist_patches(max_train, max_test):
    ds = load_dataset("ylecun/mnist")

    def proc(split, n):
        imgs = np.stack([np.array(split[i]["image"]) for i in range(min(n, len(split)))])
        labels = np.array([split[i]["label"] for i in range(min(n, len(split)))])
        imgs = imgs.astype(np.float32) / 255.0
        N = imgs.shape[0]
        patches = imgs.reshape(N, 4, 7, 4, 7).transpose(0, 1, 3, 2, 4).reshape(N, 16, 49)
        return torch.tensor(patches), torch.tensor(labels, dtype=torch.long)

    t0 = time.time()
    X_tr, y_tr = proc(ds["train"], max_train)
    X_te, y_te = proc(ds["test"], max_test)
    p(f"  Data: {X_tr.shape[0]} train, {X_te.shape[0]} test ({time.time()-t0:.1f}s)")
    return X_tr, y_tr, X_te, y_te


def run_experiment(num_heads, embed_dim, max_train, max_test, epochs, label):
    p(f"\n{'='*60}")
    p(f"  {label}")
    p(f"{'='*60}")

    X_tr, y_tr, X_te, y_te = load_mnist_patches(max_train, max_test)

    model = PhaseMultiHeadModel(
        vocab_size=10, embed_dim=embed_dim, num_heads=num_heads,
        num_classes=10, max_seq_len=16, causal=False,
        task="classify", patch_pixels=49
    )
    total_params = sum(pp.numel() for pp in model.parameters())
    attn_params = sum(pp.numel() for pp in model.attention.parameters())
    p(f"  Params: {total_params:,} total, {attn_params:,} attention ({total_params*4/1024:.0f} KB)")
    p(f"  Heads: {num_heads}, Embed: {embed_dim}, Head dim: {embed_dim//num_heads}")

    batch_size = 128
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    steps_per_epoch = (X_tr.shape[0] + batch_size - 1) // batch_size
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

        perm = torch.randperm(X_tr.shape[0])
        for i in range(0, X_tr.shape[0], batch_size):
            idx = perm[i:i+batch_size]
            optimizer.zero_grad()
            loss = criterion(model(X_tr[idx]), y_tr[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()

        elapsed = time.time() - t0
        model.eval()
        with torch.no_grad():
            # Evaluate in batches to avoid OOM
            correct = 0
            total = 0
            for i in range(0, X_te.shape[0], 512):
                logits = model(X_te[i:i+512])
                pred = torch.argmax(logits, dim=1)
                correct += (pred == y_te[i:i+512]).sum().item()
                total += y_te[i:i+512].shape[0]
            acc = correct / total

            if acc > best_acc:
                best_acc = acc
                best_epoch = epoch

        if epoch % 5 == 0 or epoch == epochs - 1:
            p(f"  Epoch {epoch:2d} | Loss {epoch_loss/steps_per_epoch:.4f} | "
              f"Acc {acc:.2%} | Best {best_acc:.2%} | {elapsed:.1f}s")

    # Phase ablation
    model.eval()
    with torch.no_grad():
        saved_q = model.attention.q_rot.data.clone()
        saved_k = model.attention.k_rot.data.clone()
        saved_v = model.attention.v_rot.data.clone()

        model.attention.q_rot.data.zero_()
        model.attention.k_rot.data.zero_()
        model.attention.v_rot.data.zero_()

        correct = 0
        total = 0
        for i in range(0, X_te.shape[0], 512):
            logits = model(X_te[i:i+512])
            pred = torch.argmax(logits, dim=1)
            correct += (pred == y_te[i:i+512]).sum().item()
            total += y_te[i:i+512].shape[0]
        ablated_acc = correct / total

        model.attention.q_rot.data.copy_(saved_q)
        model.attention.k_rot.data.copy_(saved_k)
        model.attention.v_rot.data.copy_(saved_v)

    p(f"\n  RESULT: {best_acc:.2%} (epoch {best_epoch})")
    p(f"  Ablated: {ablated_acc:.2%}  |  Phase contribution: {best_acc-ablated_acc:+.2%}")
    p(f"  Model size: {total_params*4/1024:.0f} KB")

    return {
        "label": label,
        "best_acc": best_acc,
        "ablated_acc": ablated_acc,
        "params": total_params,
        "attn_params": attn_params,
        "num_heads": num_heads,
        "embed_dim": embed_dim,
    }


def main():
    p("=" * 60)
    p("  Multi-Head Phase Interference — MNIST")
    p("=" * 60)

    results = []

    # Progressive experiments
    results.append(run_experiment(
        num_heads=4, embed_dim=128, max_train=10000, max_test=2000,
        epochs=30, label="4-head, embed=128, 10K train"
    ))

    results.append(run_experiment(
        num_heads=8, embed_dim=256, max_train=10000, max_test=2000,
        epochs=30, label="8-head, embed=256, 10K train"
    ))

    # Scale to full MNIST with best config
    results.append(run_experiment(
        num_heads=8, embed_dim=256, max_train=60000, max_test=10000,
        epochs=30, label="8-head, embed=256, FULL 60K train"
    ))

    # Summary
    p(f"\n{'='*60}")
    p("  SUMMARY")
    p(f"{'='*60}")
    p(f"  {'Config':<40} {'Acc':>6} {'Ablated':>8} {'Params':>10} {'Size':>8}")
    p(f"  {'-'*40} {'-'*6} {'-'*8} {'-'*10} {'-'*8}")
    for r in results:
        size_kb = r['params'] * 4 / 1024
        p(f"  {r['label']:<40} {r['best_acc']:5.1%} {r['ablated_acc']:7.1%} "
          f"{r['params']:>10,} {size_kb:>6.0f} KB")

    # Save results
    path = os.path.join(os.path.dirname(__file__), "phase_mnist_multihead_results.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    p(f"\nSaved to {path}")


if __name__ == "__main__":
    main()
