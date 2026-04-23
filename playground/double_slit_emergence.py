"""
double_slit_emergence.py

Can the double-slit interference pattern EMERGE from two local operations
applied to a complex field, without implementing any equation of physics?

Operations (same as interference.py):
  1. INTERFERENCE: Re(ψ · ψ†) between neighbors → updates magnitude
     (constructive amplifies, destructive cancels)
  2. PROPAGATION: sin(θ_neighbors - θ_self) → updates phase
     (phase synchronizes toward neighbors = wave spreading)

These are the same two operations from interference.py.
No Schrödinger equation. No wave equation. No Huygens principle.
Just local rules on complex numbers.

If an interference pattern appears on the screen behind two slits,
then these two operations PRODUCE quantum mechanical behavior
without being told to.

    python double_slit_emergence.py
"""

import torch
import math


def p(msg=""):
    print(msg, flush=True)


def run():
    W, D = 200, 400
    wavelength = 24.0
    slit_sep = 40
    slit_width = 6
    wall_y = 100
    steps = 1500

    k = 2 * math.pi / wavelength

    p()
    p("  ═══════════════════════════════════════════════")
    p("  Double Slit — Emergence Test")
    p("  ═══════════════════════════════════════════════")
    p(f"  Grid: {W}×{D}, λ={wavelength}, d={slit_sep}")
    p(f"  Op 1: Re(ψ · ψ†) — interference")
    p(f"  Op 2: sin(Δθ) — phase propagation")
    p(f"  NO equations of physics. Just local rules.")
    p()

    # Field — NOT zero. Vacuum has fluctuations (ε everywhere).
    # Propagation = interference with the vacuum.
    # THE VACUUM: ε everywhere. Not zero. Zero doesn't exist.
    # lim → 0 but never arrives. This IS the physics.
    epsilon = 1e-4
    psi = torch.polar(
        epsilon * torch.ones(D, W),
        2 * math.pi * torch.rand(D, W)  # random phase = vacuum fluctuations
    )

    # Wall: True = blocked
    wall_mask = torch.zeros(D, W, dtype=torch.bool)
    c1 = W // 2 - slit_sep // 2
    c2 = W // 2 + slit_sep // 2
    for x in range(W):
        is_slit1 = abs(x - c1) <= slit_width // 2
        is_slit2 = abs(x - c2) <= slit_width // 2
        if not (is_slit1 or is_slit2):
            wall_mask[wall_y, x] = True

    # Screen accumulator
    screen_y = D - 50
    screen_accum = torch.zeros(W)
    accum_count = 0

    # Coupling strengths — weaker to prevent runaway
    alpha_mag = 0.02    # magnitude coupling (conservative)
    alpha_phase = 0.15  # phase coupling

    for t in range(1, steps + 1):
        # --- SOURCE: continuous plane wave at y = 0..5 ---
        source_phase = -k * t * 0.3
        for sy in range(5):
            psi[sy, :] = torch.polar(
                torch.ones(W),
                torch.full((W,), source_phase + k * sy)
            )

        # --- COMPUTE NEIGHBOR SUM (vectorized) ---
        padded = torch.nn.functional.pad(
            psi.unsqueeze(0).unsqueeze(0),
            (1, 1, 1, 1), mode='constant', value=0
        ).squeeze()

        # 4-neighbor sum (Von Neumann neighborhood)
        n_sum = (padded[:-2, 1:-1] + padded[2:, 1:-1] +
                 padded[1:-1, :-2] + padded[1:-1, 2:])

        # --- OPERATION 1: INTERFERENCE ---
        # Re(ψ · ψ_neighbors†) = Σ |ψ|·|ψ_n|·cos(Δφ)
        cross_term = (psi * n_sum.conj()).real

        # --- OPERATION 2: PROPAGATION ---
        # Phase moves toward neighbor average phase
        phase_self = torch.angle(psi)
        phase_neighbors = torch.angle(n_sum)
        phase_diff = torch.sin(phase_neighbors - phase_self)

        # --- UPDATE ---
        mag = torch.abs(psi) + 1e-30
        new_mag = mag + alpha_mag * cross_term / 4.0
        new_phase = phase_self + alpha_phase * phase_diff

        psi = torch.polar(torch.clamp(new_mag, min=0, max=3.0), new_phase)

        # --- ENERGY CONSERVATION ---
        # Total amplitude stays constant (universe conserves energy)
        total_before = torch.abs(psi).sum()

        # --- ENFORCE WALL ---
        psi[wall_mask] = 0

        # --- DAMPING AT EDGES ---
        damp = 15
        for edge in range(damp):
            f = edge / damp
            psi[-1-edge, :] *= f
            psi[:, edge] *= f
            psi[:, -1-edge] *= f

        # --- RE-INJECT SOURCE ---
        for sy in range(5):
            psi[sy, :] = torch.polar(
                torch.ones(W),
                torch.full((W,), source_phase + k * sy)
            )

        # --- NORMALIZE (energy conservation, excluding source) ---
        total_after = torch.abs(psi).sum()
        if total_after > total_before * 1.01:  # only if growing
            psi[6:, :] = psi[6:, :] * (total_before / (total_after + 1e-30))

        # --- ACCUMULATE SCREEN ---
        if t > 300:
            screen_accum += torch.abs(psi[screen_y, :]) ** 2
            accum_count += 1

        if t % 200 == 0:
            energy = torch.abs(psi[screen_y, :]).sum().item()
            p(f"  Step {t:4d} | Screen energy: {energy:.4f}")

    # --- RESULTS ---
    if accum_count > 0:
        screen = screen_accum / accum_count
    else:
        screen = torch.abs(psi[screen_y, :]) ** 2

    screen_norm = screen / (screen.max() + 1e-30)

    # Theory: cos²(πd·sinθ/λ)
    L = float(screen_y - wall_y)
    center = W // 2
    theory = torch.zeros(W)
    for x in range(W):
        theta = math.atan2(x - center, L)
        theory[x] = math.cos(math.pi * slit_sep * math.sin(theta) / wavelength) ** 2

    # --- PLOT ---
    p()
    p("  Simulation (Re(ψ·ψ†) + sin(Δθ) only):")
    h = 20
    for row in range(h, -1, -1):
        thr = row / h
        line = "  │"
        for x in range(0, W, 2):
            line += "█" if screen_norm[x].item() >= thr else " "
        print(line)
    p("  └" + "─" * (W // 2))

    p()
    p("  Schrödinger prediction (cos²(πd·sinθ/λ)):")
    for row in range(h, -1, -1):
        thr = row / h
        line = "  │"
        for x in range(0, W, 2):
            line += "█" if theory[x].item() >= thr else " "
        print(line)
    p("  └" + "─" * (W // 2))

    # --- METRICS ---
    sim_peaks = [x for x in range(2, W-2)
                 if screen_norm[x] > screen_norm[x-1]
                 and screen_norm[x] > screen_norm[x+1]
                 and screen_norm[x] > 0.05]

    thy_peaks = [x for x in range(2, W-2)
                 if theory[x] > theory[x-1]
                 and theory[x] > theory[x+1]
                 and theory[x] > 0.05]

    margin = W // 4
    s = screen_norm[margin:W-margin]
    t_ = theory[margin:W-margin]
    corr = 0.0
    if s.std() > 1e-6 and t_.std() > 1e-6:
        corr = torch.corrcoef(torch.stack([s, t_]))[0, 1].item()

    theory_spacing = wavelength * L / slit_sep

    p()
    p("  ═══════════════════════════════════════════════")
    p("  RESULT")
    p("  ═══════════════════════════════════════════════")
    p(f"  Simulation peaks:          {len(sim_peaks)}")
    p(f"  Theory peaks:              {len(thy_peaks)}")
    p(f"  Correlation with theory:   {corr:.4f}")
    p(f"  Theoretical fringe spacing (λL/d): {theory_spacing:.1f}")
    p()

    if len(sim_peaks) >= 3:
        spacings = [sim_peaks[i+1] - sim_peaks[i] for i in range(len(sim_peaks)-1)]
        avg = sum(spacings) / len(spacings)
        p(f"  Measured fringe spacing:    {avg:.1f}")
        spacing_error = abs(avg - theory_spacing) / theory_spacing * 100
        p(f"  Spacing error vs theory:   {spacing_error:.1f}%")

    p()
    if len(sim_peaks) >= 3 and corr > 0.3:
        p("  DOUBLE-SLIT PATTERN EMERGED.")
        p("  Two local operations produced quantum interference.")
        p("  No Schrödinger equation was used. It emerged.")
    elif len(sim_peaks) >= 2:
        p("  PARTIAL INTERFERENCE DETECTED.")
        p(f"  {len(sim_peaks)} fringes visible. Pattern is forming.")
    else:
        p("  PATTERN NOT RESOLVED.")
        p("  Reported honestly. May need parameter adjustment.")
    p()


if __name__ == "__main__":
    run()
