"""
Universe Emergence Simulator
=============================

The question: can structure emerge from uniformity through interference alone?

Physics says yes — the cosmic microwave background shows that the entire
observable universe started as a nearly uniform quantum field. Tiny quantum
fluctuations (1 part in 100,000) interfered with each other. Constructive
interference amplified some regions → matter clumped → galaxies formed.
Destructive interference cancelled others → cosmic voids.

This simulation:
1. Start with a UNIFORM 2D field of complex wavefunctions (the "Big Bang" — everything the same)
2. Add tiny quantum fluctuations (random phase perturbations)
3. Apply LOCAL phase interference (each cell interferes with neighbors)
4. Watch structure emerge from nothing

The rules are the SAME as the phase attention model:
- State = complex wavefunction z = r·e^(iθ)
- Interaction = interference (constructive/destructive)
- Evolution = iterate

No forces. No gravity equations. No particle physics.
Just interference. Let's see what emerges.
"""

import torch
import math
import time
import os


def p(msg):
    print(msg, flush=True)


def create_universe(size=64, initial_perturbation=0.001):
    """Create the initial state — nearly uniform, with quantum fluctuations.

    Like the universe at t=0: everything is the same energy,
    with tiny random perturbations (1 part in 1000).
    """
    # Uniform magnitude (same energy everywhere)
    magnitude = torch.ones(size, size)

    # Uniform phase + tiny perturbations (quantum fluctuations)
    phase = torch.zeros(size, size) + initial_perturbation * torch.randn(size, size)

    # Complex wavefunction
    state = torch.polar(magnitude, phase)
    return state


def local_interference(state, interaction_strength=0.1):
    """Each cell interferes with its 8 neighbors.

    This is the SAME mechanism as phase attention:
    Q·K† between neighbors = constructive/destructive interference.
    But instead of learned rotations, physics uses the natural phase differences.
    """
    size = state.shape[0]

    # Compute interference from all 8 neighbors
    # Pad with periodic boundaries (the universe wraps — like a torus)
    padded = torch.nn.functional.pad(
        state.unsqueeze(0).unsqueeze(0),
        (1, 1, 1, 1), mode='circular'
    ).squeeze()

    # Sum of neighbor wavefunctions (interference)
    neighbor_sum = torch.zeros_like(state, dtype=torch.complex64)
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            if di == 0 and dj == 0:
                continue
            neighbor_sum += padded[1+di:size+1+di, 1+dj:size+1+dj]

    # Interference: the REAL PART of (self * neighbor†) determines attraction/repulsion
    # Constructive (in-phase) → amplify. Destructive (anti-phase) → cancel.
    interference = (state * neighbor_sum.conj()).real

    # Update: state evolves based on interference
    # Positive interference → magnitude grows (matter clumps)
    # Negative interference → magnitude shrinks (voids form)
    magnitude = torch.abs(state)
    phase = torch.angle(state)

    # Magnitude update: grow where constructive, shrink where destructive
    new_magnitude = magnitude + interaction_strength * interference / 8.0

    # Phase update: rotate toward neighbors (synchronization)
    phase_diff = torch.angle(neighbor_sum) - phase
    new_phase = phase + interaction_strength * 0.5 * torch.sin(phase_diff)

    # Normalize to conserve total energy (like the universe)
    new_state = torch.polar(torch.abs(new_magnitude), new_phase)
    total_energy = torch.abs(new_state).sum()
    original_energy = torch.abs(state).sum()
    new_state = new_state * (original_energy / (total_energy + 1e-10))

    return new_state


def measure_structure(state):
    """Quantify how much structure has emerged.

    Uniformity = no structure (entropy is maximal).
    Structure = some regions dense, others sparse (entropy decreases).
    """
    magnitude = torch.abs(state)

    # Structure metric: standard deviation of magnitude
    # Uniform → std ≈ 0. Structured → std >> 0.
    structure = magnitude.std().item()

    # Clustering: how many "dense" regions exist
    mean_mag = magnitude.mean()
    dense = (magnitude > mean_mag * 1.5).float().mean().item()  # fraction above 1.5× mean
    sparse = (magnitude < mean_mag * 0.5).float().mean().item()  # fraction below 0.5× mean

    # Phase coherence: do nearby cells have similar phase?
    phase = torch.angle(state)
    phase_grad = ((phase[1:, :] - phase[:-1, :]).abs().mean() +
                  (phase[:, 1:] - phase[:, :-1]).abs().mean()) / 2
    coherence = 1.0 / (phase_grad.item() + 1e-6)

    return {
        "structure": structure,
        "dense_fraction": dense,
        "sparse_fraction": sparse,
        "phase_coherence": coherence,
    }


def visualize_ascii(state, size=40):
    """Render the universe as ASCII art based on magnitude."""
    magnitude = torch.abs(state)

    # Downsample if needed
    if state.shape[0] > size:
        step = state.shape[0] // size
        magnitude = magnitude[::step, ::step][:size, :size]

    # Normalize
    mag_min = magnitude.min()
    mag_max = magnitude.max()
    normalized = (magnitude - mag_min) / (mag_max - mag_min + 1e-10)

    # ASCII density map
    chars = " ·:;+*#@"
    lines = []
    for row in range(normalized.shape[0]):
        line = ""
        for col in range(normalized.shape[1]):
            idx = int(normalized[row, col].item() * (len(chars) - 1))
            line += chars[min(idx, len(chars) - 1)]
        lines.append(line)
    return "\n".join(lines)


def simulate():
    p("=" * 60)
    p("  Universe Emergence Simulator")
    p("  Structure from interference alone")
    p("=" * 60)

    # Parameters
    universe_size = 64
    perturbation = 0.01   # quantum fluctuation strength (1%)
    interaction = 0.15     # how strongly neighbors interfere
    time_steps = 200

    p(f"\n  Grid: {universe_size}×{universe_size} cells")
    p(f"  Initial perturbation: {perturbation} (quantum fluctuations)")
    p(f"  Interaction strength: {interaction}")
    p(f"  Time steps: {time_steps}")

    # Create universe
    state = create_universe(universe_size, perturbation)
    initial_metrics = measure_structure(state)

    p(f"\n  t=0 (Big Bang): structure={initial_metrics['structure']:.6f}")
    p(f"  (Nearly uniform — no structure yet)\n")
    p(visualize_ascii(state))

    # Evolve
    p(f"\n{'='*60}")
    p(f"  Evolving through interference...")
    p(f"{'='*60}\n")

    snapshots = [(0, initial_metrics, state.clone())]

    for t in range(1, time_steps + 1):
        state = local_interference(state, interaction)

        if t % 20 == 0 or t == 1 or t == time_steps:
            metrics = measure_structure(state)
            snapshots.append((t, metrics, state.clone()))

            p(f"  t={t:3d} | structure={metrics['structure']:.6f} | "
              f"dense={metrics['dense_fraction']:.1%} | "
              f"sparse={metrics['sparse_fraction']:.1%} | "
              f"coherence={metrics['phase_coherence']:.1f}")

            if t in [1, 20, 60, 100, time_steps]:
                p(f"\n  --- Universe at t={t} ---")
                p(visualize_ascii(state))
                p("")

    # Final analysis
    final_metrics = measure_structure(state)
    p(f"\n{'='*60}")
    p(f"  RESULTS")
    p(f"{'='*60}")
    p(f"\n  Structure growth: {initial_metrics['structure']:.6f} → {final_metrics['structure']:.6f}")
    p(f"  ({final_metrics['structure']/max(initial_metrics['structure'], 1e-10):.1f}× amplification)")
    p(f"\n  Dense regions: {final_metrics['dense_fraction']:.1%} of space")
    p(f"  Sparse regions (voids): {final_metrics['sparse_fraction']:.1%} of space")
    p(f"  Phase coherence: {final_metrics['phase_coherence']:.1f}")

    if final_metrics['structure'] > initial_metrics['structure'] * 10:
        p(f"\n  → STRUCTURE EMERGED FROM UNIFORMITY.")
        p(f"    Tiny perturbations + interference = cosmic structure.")
        p(f"    No forces needed. No gravity. Just waves interfering.")
        p(f"    The dense regions are 'galaxies'. The sparse regions are 'voids'.")
        p(f"    This is how physics says it actually happened.")
    else:
        p(f"\n  → Structure did not significantly emerge.")
        p(f"    Try increasing perturbation or interaction strength.")

    # Save
    results = {
        "initial": initial_metrics,
        "final": final_metrics,
        "params": {
            "size": universe_size,
            "perturbation": perturbation,
            "interaction": interaction,
            "steps": time_steps,
        }
    }
    import json
    path = os.path.join(os.path.dirname(__file__), "universe_sim_results.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    p(f"\n  Results saved to {path}")


if __name__ == "__main__":
    simulate()
