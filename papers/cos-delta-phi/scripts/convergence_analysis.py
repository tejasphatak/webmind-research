"""
Convergence Loop Analysis
=========================

The MNIST experiment showed Δ ≈ 1.40 at every epoch, never converging early.
This script probes WHY from the quantum physics perspective:

1. Is each iteration doing useful work? (accuracy per step)
2. Is Δ decreasing across steps? (approaching eigenstate or oscillating?)
3. Is the state magnitude growing? (if |ψ| grows, Δ = |ψ_new - ψ_old|/|ψ_new| ≈ 1
   even if the DIRECTION converges — the measurement is broken, not the physics)
4. Does the phase distribution change per step? (is the wavefunction rotating or settled?)
"""

import torch
import torch.nn as nn
import math
import time
import os
import sys

# Import the model class from our experiment
sys.path.insert(0, os.path.dirname(__file__))
from phase_mnist import PhaseMNIST, load_mnist_patches


def p(msg):
    print(msg, flush=True)


def load_trained_model():
    """Load the trained MNIST model."""
    path = os.path.join(os.path.dirname(__file__), "phase_mnist_model.pth")
    checkpoint = torch.load(path, weights_only=False)
    model = PhaseMNIST(**checkpoint["config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    p(f"Loaded model from {path}")
    p(f"  Config: {checkpoint['config']}")
    p(f"  Training result: {checkpoint['results']}")
    return model


def diagnostic_forward(model, x):
    """Run forward pass collecting per-step diagnostics.

    Returns dict with per-step:
    - logits at each step
    - delta (L2 ratio)
    - delta_cosine (directional change)
    - state_magnitude (mean |z|)
    - phase_std (spread of angles)
    """
    batch_size = x.shape[0]

    # Replicate the forward pass with instrumentation
    mag = torch.tanh(model.patch_proj(x))
    state = mag.to(torch.complex64) * torch.exp(1j * model.pos_phase.unsqueeze(0))

    ff_weights = torch.complex(model.ff_real, model.ff_imag)

    diagnostics = []

    for step in range(model.max_steps):
        prev_state = state.clone()

        # Phase rotation
        q = state * torch.exp(1j * model.q_rot)
        k = state * torch.exp(1j * model.k_rot)
        v = state * torch.exp(1j * model.v_rot)

        # Interference
        interference = torch.matmul(q, k.conj().transpose(-1, -2))
        attn_logits = interference.real / math.sqrt(model.embed_dim)
        attn_weights = torch.softmax(attn_logits * 8.0, dim=-1)

        attn_out = torch.matmul(attn_weights.to(torch.complex64), v)
        state = state + attn_out

        # Resonance
        state = torch.matmul(state, ff_weights)
        state = model.norm(state)

        # --- DIAGNOSTICS ---

        # Standard Δ (L2 ratio) — what the model uses
        delta_l2 = (torch.norm(state - prev_state) / (torch.norm(state) + 1e-8)).item()

        # Cosine similarity between states (directional convergence)
        # Flatten to (batch, -1) for cosine
        flat_prev = prev_state.reshape(batch_size, -1)
        flat_curr = state.reshape(batch_size, -1)
        # Use real parts for cosine (complex cosine doesn't have standard definition)
        cos_real = nn.functional.cosine_similarity(
            flat_curr.real, flat_prev.real, dim=-1
        ).mean().item()
        cos_imag = nn.functional.cosine_similarity(
            flat_curr.imag, flat_prev.imag, dim=-1
        ).mean().item()

        # State magnitude (is |ψ| growing?)
        state_mag = torch.abs(state).mean().item()
        prev_mag = torch.abs(prev_state).mean().item()

        # Phase distribution
        phases = torch.angle(state)
        phase_std = phases.std().item()
        phase_mean = phases.mean().item()

        # Logits at this step (measurement)
        pooled = state.mean(dim=1)
        logits = model.readout_real(pooled.real) + model.readout_imag(pooled.imag)

        diagnostics.append({
            "step": step + 1,
            "delta_l2": delta_l2,
            "cosine_real": cos_real,
            "cosine_imag": cos_imag,
            "state_mag": state_mag,
            "prev_mag": prev_mag,
            "mag_growth": state_mag / (prev_mag + 1e-8),
            "phase_std": phase_std,
            "phase_mean": phase_mean,
            "logits": logits.detach(),
        })

    return diagnostics


def main():
    p("=" * 60)
    p("Convergence Loop Analysis")
    p("=" * 60)

    # Load model and data
    model = load_trained_model()
    _, _, X_test, y_test = load_mnist_patches(max_train=100, max_test=500)

    p(f"\nRunning diagnostic forward on {X_test.shape[0]} test samples...\n")

    with torch.no_grad():
        diagnostics = diagnostic_forward(model, X_test)

    # === ANALYSIS 1: Per-step accuracy ===
    p("=" * 60)
    p("1. PER-STEP ACCURACY (does each iteration help?)")
    p("=" * 60)
    for d in diagnostics:
        pred = torch.argmax(d["logits"], dim=1)
        acc = (pred == y_test).float().mean().item()
        p(f"  Step {d['step']}: {acc:.2%}")

    # === ANALYSIS 2: Δ trajectory ===
    p(f"\n{'=' * 60}")
    p("2. Δ TRAJECTORY (approaching eigenstate or oscillating?)")
    p("=" * 60)
    for d in diagnostics:
        p(f"  Step {d['step']}: Δ_L2={d['delta_l2']:.4f}  "
          f"cos_real={d['cosine_real']:.4f}  cos_imag={d['cosine_imag']:.4f}")

    delta_decreasing = all(
        diagnostics[i]["delta_l2"] >= diagnostics[i+1]["delta_l2"]
        for i in range(len(diagnostics)-1)
    )
    p(f"\n  Δ monotonically decreasing? {'YES' if delta_decreasing else 'NO'}")

    if diagnostics[-1]["cosine_real"] > 0.95:
        p("  → Direction is converging (cos > 0.95) even though L2 says otherwise.")
        p("    The wavefunction found its eigenstate direction; the norm just keeps shifting.")
    else:
        p("  → Direction is NOT converging. Each step genuinely changes the state.")

    # === ANALYSIS 3: Magnitude growth ===
    p(f"\n{'=' * 60}")
    p("3. STATE MAGNITUDE EVOLUTION (is |ψ| growing?)")
    p("=" * 60)
    for d in diagnostics:
        p(f"  Step {d['step']}: |ψ|={d['state_mag']:.4f}  "
          f"(prev={d['prev_mag']:.4f}, growth={d['mag_growth']:.4f}x)")

    total_growth = diagnostics[-1]["state_mag"] / (diagnostics[0]["prev_mag"] + 1e-8)
    p(f"\n  Total magnitude growth across loop: {total_growth:.2f}x")
    if total_growth > 2.0:
        p("  → MAGNITUDE IS GROWING. This explains why Δ_L2 ≈ 1:")
        p("    norm(new - old) / norm(new) ≈ 1 when |new| >> |old|.")
        p("    The convergence CHECK is broken, not the convergence itself.")

    # === ANALYSIS 4: Phase evolution ===
    p(f"\n{'=' * 60}")
    p("4. PHASE DISTRIBUTION PER STEP")
    p("=" * 60)
    for d in diagnostics:
        p(f"  Step {d['step']}: phase_std={d['phase_std']:.4f}  "
          f"phase_mean={d['phase_mean']:.4f}")

    # === SUMMARY ===
    p(f"\n{'=' * 60}")
    p("DIAGNOSIS")
    p("=" * 60)

    accs = []
    for d in diagnostics:
        pred = torch.argmax(d["logits"], dim=1)
        accs.append((pred == y_test).float().mean().item())

    acc_improving = accs[-1] > accs[0] + 0.01
    dir_converging = diagnostics[-1]["cosine_real"] > 0.90
    mag_growing = total_growth > 1.5

    if acc_improving and mag_growing:
        p("Each step improves accuracy AND magnitude grows.")
        p("The convergence loop IS doing useful work — it's iterative refinement.")
        if dir_converging:
            p("Direction converges but norm doesn't — fix: use cosine Δ for early stopping.")
        else:
            p("Direction also changes — each step is a genuine computation step, not convergence.")
            p("This is closer to 'unrolled layers' than 'finding an eigenstate'.")
    elif not acc_improving:
        p("Later steps don't improve accuracy — they're wasted compute.")
        p("Reduce max_steps to save time.")

    p(f"\nAccuracy trajectory: {' → '.join(f'{a:.1%}' for a in accs)}")


if __name__ == "__main__":
    main()
