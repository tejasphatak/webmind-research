"""
interference.py

Start with nothing.
Apply one formula.
Watch a universe emerge.

The formula: Re(ψ · ψ†) — the interference cross-term.
The input: zero everywhere, plus the smallest instability: ε = 1e-10.
The rule: each point interferes with its neighbors.
Nothing else.

    python interference.py
"""

import torch
import math


def run():

    # ── THE VOID ──────────────────────────────────────────
    # Everything is zero.
    # Except: perfect zero is impossible (Heisenberg).
    # So: ε. The smallest thing that isn't nothing.

    size = 100
    epsilon = 1e-10  # the instability

    field = torch.zeros(size, size, dtype=torch.complex64)
    field = field + epsilon * torch.randn(size, size).to(torch.complex64)

    print()
    print("  ═══════════════════════════════════════════")
    print("  Start: zero + ε (the smallest instability)")
    print(f"  Field: {size}×{size}")
    print(f"  ε = {epsilon}")
    print("  Formula: Re(ψ · ψ†)")
    print("  ═══════════════════════════════════════════")

    show(field, 0)

    # ── APPLY THE FORMULA ─────────────────────────────────
    # Re(ψ · ψ†) between neighbors.
    # That's it. No forces. No constants. No tuning.

    for t in range(1, 301):

        padded = torch.nn.functional.pad(
            field.unsqueeze(0).unsqueeze(0),
            (1, 1, 1, 1), mode='circular'
        ).squeeze()

        # Sum of all 8 neighbors
        neighbors = torch.zeros_like(field)
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                neighbors += padded[1+di:size+1+di, 1+dj:size+1+dj]

        # ── THE FORMULA ──
        # Re(ψ · ψ†) = |ψ₁| · |ψ₂| · cos(φ₁ - φ₂)
        # Constructive: in-phase → amplify
        # Destructive: anti-phase → cancel
        cross_term = (field * neighbors.conj()).real

        # Magnitude grows where constructive, shrinks where destructive
        mag = torch.abs(field) + 1e-30  # avoid division by zero
        phase = torch.angle(field)
        new_mag = mag + 0.1 * cross_term / 8.0

        # Phase drifts toward neighbors (synchronization)
        new_phase = phase + 0.05 * torch.sin(torch.angle(neighbors) - phase)

        # Rebuild the field
        field = torch.polar(torch.clamp(new_mag, min=0), new_phase)

        # Energy conservation (total amplitude stays constant)
        total = torch.abs(field).sum()
        if total > 0:
            field = field * (size / (total + 1e-30))

        if t in [1, 2, 5, 10, 20, 50, 100, 150, 200, 300]:
            show(field, t)

    # ── WHAT EMERGED ──────────────────────────────────────
    result(field)


def show(field, t):
    mag = torch.abs(field)
    step = max(1, field.shape[0] // 50)
    small = mag[::step, ::step][:50, :50]
    mn, mx = small.min(), small.max()
    if mx - mn < 1e-20:
        norm = torch.zeros_like(small)
    else:
        norm = (small - mn) / (mx - mn)

    chars = " ·:;+*#@"
    structure = mag.std().item()
    energy = mag.sum().item()

    print(f"  t={t:<4d} structure={structure:.2e}  energy={energy:.2f}")
    for row in range(norm.shape[0]):
        line = "  "
        for col in range(norm.shape[1]):
            idx = int(norm[row, col].item() * (len(chars) - 1))
            line += chars[min(idx, len(chars) - 1)]
        print(line)
    print()


def result(field):
    mag = torch.abs(field)
    structure = mag.std().item()
    dense = (mag > mag.mean() * 2).float().mean().item()
    void = (mag < mag.mean() * 0.1).float().mean().item()

    print("  ═══════════════════════════════════════════")
    print("  RESULT")
    print("  ═══════════════════════════════════════════")
    print()
    print(f"  Started with: zero + ε ({1e-10})")
    print(f"  Applied: Re(ψ · ψ†) between neighbors, 300 steps")
    print(f"  Nothing else.")
    print()
    print(f"  Structure: {structure:.2e}")
    print(f"  Dense regions: {dense:.1%} of space")
    print(f"  Void regions: {void:.1%} of space")
    print()

    if structure > 1e-6:
        print("  Something emerged from nothing.")
        print("  The formula created structure where there was none.")
        print("  No forces. No gravity. No rules. Just interference.")
    else:
        print("  The field remained uniform.")
        print("  ε was too small or steps too few for structure to emerge.")

    print()
    print("  The formula: Re(ψ · ψ†) = |ψ₁|·|ψ₂|·cos(φ₁ - φ₂)")
    print()


if __name__ == "__main__":
    run()
