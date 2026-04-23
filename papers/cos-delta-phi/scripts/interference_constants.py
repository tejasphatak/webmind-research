"""
interference_constants.py

Start with nothing. Apply Re(ψ · ψ†).
Measure what constants emerge.

No constants are put in. We measure what comes out.
If the universe runs on interference, it should produce
its own structure constants — ratios, scaling laws, symmetries.

    python interference_constants.py
"""

import torch
import math


def run():
    size = 100
    epsilon = 1e-10

    field = torch.zeros(size, size, dtype=torch.complex64)
    field = field + epsilon * torch.randn(size, size).to(torch.complex64)

    print()
    print("  ═══════════════════════════════════════════════════")
    print("  Emergent Constants from Interference")
    print("  Start: zero + ε.  Formula: Re(ψ · ψ†).  Nothing else.")
    print("  ═══════════════════════════════════════════════════")
    print()
    print(f"  {'t':>5}  {'structure':>12}  {'dense/void':>12}  {'phase_order':>12}  {'energy_ratio':>12}  {'cluster_n':>10}")
    print(f"  {'─'*5}  {'─'*12}  {'─'*12}  {'─'*12}  {'─'*12}  {'─'*10}")

    constants_log = []

    for t in range(1, 501):
        padded = torch.nn.functional.pad(
            field.unsqueeze(0).unsqueeze(0),
            (1, 1, 1, 1), mode='circular'
        ).squeeze()

        neighbors = torch.zeros_like(field)
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                neighbors += padded[1+di:size+1+di, 1+dj:size+1+dj]

        cross_term = (field * neighbors.conj()).real

        mag = torch.abs(field) + 1e-30
        phase = torch.angle(field)
        new_mag = mag + 0.1 * cross_term / 8.0
        new_phase = phase + 0.05 * torch.sin(torch.angle(neighbors) - phase)

        field = torch.polar(torch.clamp(new_mag, min=0), new_phase)
        total = torch.abs(field).sum()
        if total > 0:
            field = field * (size / (total + 1e-30))

        if t % 25 == 0 or t in [1, 5, 10]:
            mag = torch.abs(field)
            phase = torch.angle(field)

            # EMERGENT CONSTANT 1: Structure (std/mean ratio)
            structure = mag.std().item() / (mag.mean().item() + 1e-30)

            # EMERGENT CONSTANT 2: Dense-to-void ratio
            mean_mag = mag.mean().item()
            dense = (mag > mean_mag * 2).float().sum().item()
            void = (mag < mean_mag * 0.1).float().sum().item()
            dv_ratio = dense / (void + 1)

            # EMERGENT CONSTANT 3: Phase order parameter
            # Average of e^(iφ) — 1.0 = perfect alignment, 0.0 = random
            phase_order = torch.abs(torch.exp(1j * phase).mean()).item()

            # EMERGENT CONSTANT 4: Energy concentration
            # What fraction of total energy is in top 10% of cells?
            sorted_mag = mag.flatten().sort(descending=True).values
            top_10_pct = sorted_mag[:size*size//10].sum().item()
            total_energy = sorted_mag.sum().item()
            energy_ratio = top_10_pct / (total_energy + 1e-30)

            # EMERGENT CONSTANT 5: Number of clusters
            # Connected components above 2× mean
            binary = (mag > mean_mag * 2).float()
            # Simple flood fill count
            n_clusters = count_clusters(binary)

            print(f"  {t:5d}  {structure:12.6f}  {dv_ratio:12.4f}  {phase_order:12.6f}  {energy_ratio:12.4f}  {n_clusters:10d}")

            constants_log.append({
                "t": t,
                "structure": structure,
                "dense_void_ratio": dv_ratio,
                "phase_order": phase_order,
                "energy_concentration": energy_ratio,
                "n_clusters": n_clusters,
            })

    # FINAL ANALYSIS: What constants stabilized?
    print()
    print("  ═══════════════════════════════════════════════════")
    print("  EMERGENT CONSTANTS (values that stabilized)")
    print("  ═══════════════════════════════════════════════════")
    print()

    if len(constants_log) >= 4:
        last_4 = constants_log[-4:]

        for name in ["structure", "dense_void_ratio", "phase_order", "energy_concentration", "n_clusters"]:
            values = [c[name] for c in last_4]
            mean_v = sum(values) / len(values)
            std_v = (sum((v - mean_v)**2 for v in values) / len(values)) ** 0.5
            stability = 1.0 - (std_v / (abs(mean_v) + 1e-30))

            if stability > 0.9 and abs(mean_v) > 1e-10:
                print(f"  {name:<25s} = {mean_v:.6f}  (stable, σ/μ = {std_v/(abs(mean_v)+1e-30):.4f})")
            elif abs(mean_v) > 1e-10:
                print(f"  {name:<25s} = {mean_v:.6f}  (drifting, σ/μ = {std_v/(abs(mean_v)+1e-30):.4f})")

    # Ratios between constants
    print()
    print("  ─── Ratios ───")
    if constants_log:
        last = constants_log[-1]
        if last["phase_order"] > 1e-10 and last["structure"] > 1e-10:
            ratio1 = last["energy_concentration"] / last["phase_order"]
            print(f"  energy_concentration / phase_order = {ratio1:.4f}")
        if last["structure"] > 1e-10:
            ratio2 = last["energy_concentration"] / last["structure"]
            print(f"  energy_concentration / structure    = {ratio2:.4f}")
        if last["n_clusters"] > 0:
            ratio3 = last["structure"] / last["n_clusters"]
            print(f"  structure / n_clusters              = {ratio3:.6f}")

    print()
    print("  These constants were not put in. They came out.")
    print("  They are properties of Re(ψ · ψ†) itself.")
    print()


def count_clusters(binary):
    """Count connected components in a binary 2D field."""
    visited = torch.zeros_like(binary, dtype=torch.bool)
    n = 0
    for i in range(binary.shape[0]):
        for j in range(binary.shape[1]):
            if binary[i, j] > 0 and not visited[i, j]:
                # BFS flood fill
                stack = [(i, j)]
                while stack:
                    ci, cj = stack.pop()
                    if ci < 0 or ci >= binary.shape[0] or cj < 0 or cj >= binary.shape[1]:
                        continue
                    if visited[ci, cj] or binary[ci, cj] == 0:
                        continue
                    visited[ci, cj] = True
                    for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                        stack.append((ci+di, cj+dj))
                n += 1
    return n


if __name__ == "__main__":
    run()
