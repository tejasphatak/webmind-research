"""
Phase Transition Analysis: Why does step 4 suddenly produce 87%?
================================================================

Two hypotheses:

HYPOTHESIS A — "Readout bias"
  The readout layer was trained end-to-end with 4 steps. It learned to
  interpret ONLY step-4 representations. Steps 1-3 might contain the
  information but in a basis the readout can't decode.
  TEST: Fit a fresh linear probe at each step. If step-1 states can
  classify well with their OWN readout, the information was there all along.

HYPOTHESIS B — "Genuine phase transition"
  The interference-resonance cycle needs exactly 4 rotations to separate
  the 10 digit classes in complex space. Like a lens needing a specific
  focal length — at 3 rotations the "image" is blurry, at 4 it snaps
  into focus. Information genuinely doesn't exist until step 4.
  TEST: Fresh linear probes at steps 1-3 will also fail. The state
  space is genuinely unresolved.

HYPOTHESIS C — "Gradual buildup"
  Information accumulates: step 1 separates easy digits (0 vs 1),
  step 2 adds more, etc. But the ORIGINAL readout only learned
  the step-4 encoding, so it can't decode intermediate states.
  TEST: Fresh probes show monotonically increasing accuracy.

Additionally: look at the ATTENTION PATTERNS at each step.
  In quantum interference, constructive/destructive patterns tell us
  which patches are "resonating" with which. If the attention patterns
  at step 4 are qualitatively different (sharper, more structured),
  that's the interference resolving — like a diffraction pattern
  coming into focus.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import math
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from phase_mnist import PhaseMNIST, load_mnist_patches


def p(msg):
    print(msg, flush=True)


def extract_per_step_states(model, x):
    """Run forward, return the complex state and attention weights at each step."""
    mag = torch.tanh(model.patch_proj(x))
    state = mag.to(torch.complex64) * torch.exp(1j * model.pos_phase.unsqueeze(0))
    ff_weights = torch.complex(model.ff_real, model.ff_imag)

    states = []
    attn_maps = []

    for step in range(model.max_steps):
        q = state * torch.exp(1j * model.q_rot)
        k = state * torch.exp(1j * model.k_rot)
        v = state * torch.exp(1j * model.v_rot)

        interference = torch.matmul(q, k.conj().transpose(-1, -2))
        attn_logits = interference.real / math.sqrt(model.embed_dim)
        attn_w = torch.softmax(attn_logits * 8.0, dim=-1)

        attn_out = torch.matmul(attn_w.to(torch.complex64), v)
        state = state + attn_out
        state = torch.matmul(state, ff_weights)
        state = model.norm(state)

        states.append(state.clone())
        attn_maps.append(attn_w.clone())

    return states, attn_maps


def train_linear_probe(features, labels, epochs=200):
    """Train a simple linear classifier on real-valued features."""
    dim = features.shape[1]
    probe = nn.Linear(dim, 10)
    opt = optim.Adam(probe.parameters(), lr=1e-2)
    criterion = nn.CrossEntropyLoss()

    # Split 80/20
    n = features.shape[0]
    split = int(n * 0.8)
    x_tr, x_te = features[:split], features[split:]
    y_tr, y_te = labels[:split], labels[split:]

    for epoch in range(epochs):
        opt.zero_grad()
        loss = criterion(probe(x_tr), y_tr)
        loss.backward()
        opt.step()

    with torch.no_grad():
        pred = torch.argmax(probe(x_te), dim=1)
        acc = (pred == y_te).float().mean().item()
    return acc


def analyze_attention_structure(attn_maps):
    """Analyze the attention patterns at each step.

    Metrics:
    - Entropy: high = uniform attention (no focus), low = sharp focus
    - Max weight: how confident is the strongest connection
    - Diagonal dominance: does each patch attend to itself?
    """
    p(f"\n{'=' * 60}")
    p("3. ATTENTION PATTERN ANALYSIS (interference structure)")
    p("=" * 60)
    p(f"   {'Step':>4}  {'Entropy':>8}  {'Max wt':>8}  {'Diag %':>8}  {'Pattern'}")
    p(f"   {'----':>4}  {'-------':>8}  {'------':>8}  {'------':>8}  {'-------'}")

    for i, attn in enumerate(attn_maps):
        # attn: (batch, seq, seq) — average over batch
        avg_attn = attn.mean(dim=0)  # (16, 16)

        # Entropy per row, then average
        entropy = -(avg_attn * torch.log(avg_attn + 1e-10)).sum(dim=-1).mean().item()
        max_entropy = math.log(avg_attn.shape[-1])  # uniform = max entropy

        # Max attention weight
        max_wt = avg_attn.max().item()

        # Diagonal dominance: how much does each position attend to itself
        diag_pct = avg_attn.diag().mean().item()

        # Characterize
        if entropy > 0.9 * max_entropy:
            pattern = "UNIFORM (no interference)"
        elif diag_pct > 0.3:
            pattern = "SELF-FOCUSED (local)"
        elif max_wt > 0.5:
            pattern = "SHARP (resolved interference)"
        else:
            pattern = "DIFFUSE (partial interference)"

        p(f"   {i+1:4d}  {entropy:8.3f}  {max_wt:8.3f}  {diag_pct:8.3f}  {pattern}")

    # Print the actual attention matrix for step 4 (averaged)
    p(f"\n  Step 4 attention matrix (averaged over batch, rounded):")
    avg_final = attn_maps[-1].mean(dim=0)
    # Show which patches attend to which — 4x4 spatial grid
    p(f"  Patches are in 4x4 grid: rows=spatial rows, cols=spatial cols")
    p(f"  Top-3 attended patches per position:")
    for pos in range(min(4, avg_final.shape[0])):  # Show first 4
        top3 = torch.topk(avg_final[pos], 3)
        row, col = pos // 4, pos % 4
        targets = [(idx.item() // 4, idx.item() % 4, val.item())
                   for idx, val in zip(top3.indices, top3.values)]
        target_str = ", ".join(f"({r},{c})={v:.2f}" for r, c, v in targets)
        p(f"    Patch ({row},{col}) → {target_str}")


def main():
    p("=" * 60)
    p("Phase Transition Analysis")
    p("=" * 60)

    # Load
    path = os.path.join(os.path.dirname(__file__), "phase_mnist_model.pth")
    checkpoint = torch.load(path, weights_only=False)
    model = PhaseMNIST(**checkpoint["config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    _, _, X_test, y_test = load_mnist_patches(max_train=100, max_test=500)

    with torch.no_grad():
        states, attn_maps = extract_per_step_states(model, X_test)

    # === TEST 1: Linear probes at each step ===
    p(f"\n{'=' * 60}")
    p("1. LINEAR PROBE TEST (is information present at each step?)")
    p("=" * 60)
    p("  Training a fresh linear classifier on the pooled state at each step.")
    p("  If step-1 probe scores well → info was there, readout couldn't decode it.")
    p("  If step-1 probe fails too → info genuinely doesn't exist yet.\n")

    probe_accs = []
    for i, state in enumerate(states):
        # Pool: mean over patches, extract magnitude and phase separately
        pooled = state.mean(dim=1)  # (batch, embed_dim) complex

        # Try three feature representations:
        # a) magnitude only
        feat_mag = torch.abs(pooled)
        acc_mag = train_linear_probe(feat_mag, y_test)

        # b) real + imag (what the model's readout uses)
        feat_ri = torch.cat([pooled.real, pooled.imag], dim=-1)
        acc_ri = train_linear_probe(feat_ri, y_test)

        # c) magnitude + phase
        feat_mp = torch.cat([torch.abs(pooled), torch.angle(pooled)], dim=-1)
        acc_mp = train_linear_probe(feat_mp, y_test)

        probe_accs.append({"mag": acc_mag, "real_imag": acc_ri, "mag_phase": acc_mp})
        p(f"  Step {i+1}: magnitude={acc_mag:.1%}  real+imag={acc_ri:.1%}  mag+phase={acc_mp:.1%}")

    # === TEST 2: Class separability ===
    p(f"\n{'=' * 60}")
    p("2. CLASS SEPARABILITY (do digit clusters form?)")
    p("=" * 60)
    p("  Measuring inter-class vs intra-class distance in complex state space.\n")

    for i, state in enumerate(states):
        pooled = state.mean(dim=1)
        # Use magnitude for distance (phase wraps around)
        feat = torch.abs(pooled)

        # Per-class centroids
        centroids = []
        intra_dists = []
        for c in range(10):
            mask = y_test == c
            if mask.sum() == 0:
                continue
            class_feat = feat[mask]
            centroid = class_feat.mean(dim=0)
            centroids.append(centroid)
            intra_dist = torch.norm(class_feat - centroid, dim=1).mean().item()
            intra_dists.append(intra_dist)

        centroids = torch.stack(centroids)
        # Inter-class: average pairwise distance between centroids
        inter_dist = torch.cdist(centroids.unsqueeze(0), centroids.unsqueeze(0)).squeeze()
        # Upper triangle only
        inter_mean = inter_dist[torch.triu(torch.ones_like(inter_dist), diagonal=1).bool()].mean().item()
        intra_mean = np.mean(intra_dists)

        ratio = inter_mean / (intra_mean + 1e-8)
        p(f"  Step {i+1}: inter={inter_mean:.3f}  intra={intra_mean:.3f}  "
          f"ratio={ratio:.2f}  {'SEPARATED' if ratio > 2.0 else 'MIXED'}")

    # === TEST 3: Attention patterns ===
    analyze_attention_structure(attn_maps)

    # === VERDICT ===
    p(f"\n{'=' * 60}")
    p("VERDICT")
    p("=" * 60)

    # Check which hypothesis
    step1_best = max(probe_accs[0].values())
    step4_best = max(probe_accs[-1].values())

    if step1_best > 0.5:
        p("HYPOTHESIS A confirmed: READOUT BIAS")
        p(f"  Step 1 probe achieves {step1_best:.1%} — information IS present early.")
        p("  The trained readout just can't decode it because it learned step-4 basis.")
        p("  The convergence loop transforms the representation, not the information.")
    elif all(max(pa.values()) < 0.2 for pa in probe_accs[:3]) and step4_best > 0.5:
        p("HYPOTHESIS B confirmed: GENUINE PHASE TRANSITION")
        p(f"  Steps 1-3 probes all < 20%, step 4 = {step4_best:.1%}")
        p("  Information genuinely emerges at step 4.")
        p("  The interference needs exactly 4 rotations to resolve the 10-class structure.")
    else:
        monotonic = all(
            max(probe_accs[i].values()) <= max(probe_accs[i+1].values()) + 0.05
            for i in range(len(probe_accs)-1)
        )
        if monotonic:
            p("HYPOTHESIS C confirmed: GRADUAL BUILDUP")
            p("  Information accumulates across steps.")
        else:
            p("MIXED: Information trajectory is non-monotonic.")
            p("  The wavefunction passes through different representation bases at each step.")

        for i, pa in enumerate(probe_accs):
            p(f"  Step {i+1} best probe: {max(pa.values()):.1%}")


if __name__ == "__main__":
    main()
