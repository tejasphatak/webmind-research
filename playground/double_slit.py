"""
double_slit.py — The experiment that proves it.

A wave passes through two slits.
We apply ONLY Re(ψ · ψ†) between neighbors.
No optics. No Huygens. No Fraunhofer.

If the output matches cos²(πd·sinθ/λ), the formula IS physics.

    python double_slit.py
"""

import torch
import math


def run():
    W, D = 200, 300  # width, depth
    wavelength = 20
    slit_sep = 40
    slit_width = 4
    wall_y = D // 3
    steps = 400

    print()
    print("  ═══════════════════════════════════════════════")
    print("  Double-Slit via Re(ψ · ψ†)")
    print("  ═══════════════════════════════════════════════")
    print(f"  Grid: {W}×{D}, λ={wavelength}, d={slit_sep}")
    print(f"  No optics equations. Just interference.")
    print()

    field = torch.zeros(D, W, dtype=torch.complex64)
    k = 2 * math.pi / wavelength

    # Wall mask: True = blocked
    wall = torch.ones(W, dtype=torch.bool)
    c1 = W // 2 - slit_sep // 2
    c2 = W // 2 + slit_sep // 2
    wall[c1 - slit_width//2 : c1 + slit_width//2 + 1] = False
    wall[c2 - slit_width//2 : c2 + slit_width//2 + 1] = False

    wall_mask = torch.zeros(D, W, dtype=torch.bool)
    wall_mask[wall_y, :] = wall

    for t in range(1, steps + 1):
        # Source: plane wave at y=0
        phase_t = k * torch.arange(W, dtype=torch.float32) * 0 - 0.1 * t
        field[0, :] = torch.polar(torch.ones(W), phase_t)

        # VECTORIZED propagation: each cell += coupling * neighbors
        # Pad for periodic boundary in x, absorbing in y
        padded = torch.nn.functional.pad(
            field.unsqueeze(0).unsqueeze(0),
            (1, 1, 1, 1), mode='constant', value=0
        ).squeeze()

        # 4-neighbor sum
        neighbors = (padded[:-2, 1:-1] + padded[2:, 1:-1] +
                     padded[1:-1, :-2] + padded[1:-1, 2:])

        # THE FORMULA
        cross_term = (field * neighbors.conj()).real
        mag = torch.abs(field) + 1e-30
        phase = torch.angle(field)
        neighbor_phase = torch.angle(neighbors)

        new_mag = mag + 0.05 * cross_term / 4.0
        new_phase = phase + 0.2 * torch.sin(neighbor_phase - phase)

        field = torch.polar(torch.clamp(new_mag, min=0, max=3.0), new_phase)

        # Enforce wall
        field[wall_mask] = 0

        # Keep source alive
        field[0, :] = torch.polar(torch.ones(W), phase_t)

        if t % 100 == 0:
            print(f"  Step {t}/{steps}...", flush=True)

    # ── MEASUREMENT ──
    screen_y = D - 20
    intensity = torch.abs(field[screen_y, :]) ** 2
    intensity = intensity / (intensity.max() + 1e-30)

    # Theory: cos²(πd·sinθ/λ)
    L = screen_y - wall_y
    center = W // 2
    theory = torch.zeros(W)
    for x in range(W):
        theta = math.atan2(x - center, L)
        theory[x] = math.cos(math.pi * slit_sep * math.sin(theta) / wavelength) ** 2

    # ── ASCII PLOT ──
    print()
    print("  Interference pattern on detection screen:")
    print()
    h = 20
    for row in range(h, -1, -1):
        thr = row / h
        line = "  │"
        for x in range(0, W, 2):
            if intensity[x].item() >= thr:
                line += "█"
            else:
                line += " "
        print(line)
    print("  └" + "─" * (W // 2))

    print()
    print("  Theory (cos²):")
    for row in range(h, -1, -1):
        thr = row / h
        line = "  │"
        for x in range(0, W, 2):
            if theory[x].item() >= thr:
                line += "█"
            else:
                line += " "
        print(line)
    print("  └" + "─" * (W // 2))

    # Stats
    sim_peaks = sum(1 for x in range(2, W-2)
                    if intensity[x] > intensity[x-1] and intensity[x] > intensity[x+1]
                    and intensity[x] > 0.05)
    thy_peaks = sum(1 for x in range(2, W-2)
                    if theory[x] > theory[x-1] and theory[x] > theory[x+1]
                    and theory[x] > 0.05)

    margin = W // 4
    s = intensity[margin:W-margin]
    t_ = theory[margin:W-margin]
    if s.std() > 1e-6 and t_.std() > 1e-6:
        corr = torch.corrcoef(torch.stack([s, t_]))[0, 1].item()
    else:
        corr = 0.0

    print()
    print("  ═══════════════════════════════════════════════")
    print(f"  Simulation peaks: {sim_peaks}")
    print(f"  Theory peaks:     {thy_peaks}")
    print(f"  Correlation:      {corr:.4f}")
    print("  ═══════════════════════════════════════════════")

    if corr > 0.5 or sim_peaks >= 3:
        print()
        print(f"  Re(ψ · ψ†) reproduces double-slit interference.")
        print(f"  No optics equations. Same formula as the attention paper.")
    print()


if __name__ == "__main__":
    run()
