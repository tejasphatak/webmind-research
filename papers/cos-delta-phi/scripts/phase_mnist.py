"""
Phase-Interference Vision Transformer on MNIST
================================================

Architecture framed through quantum physics, NOT neural network conventions:

1. WAVEFUNCTION ENCODING
   Each image patch becomes a complex state z = r·e^(iθ)
   - r (magnitude) = activation strength of the patch features
   - θ (phase) = positional identity — WHERE this patch lives in the image
   This is analogous to a quantum state |ψ⟩ = α|0⟩ + β|1⟩ where
   the complex coefficients encode both amplitude and phase.

2. PHASE ROTATION (Q/K/V)
   Instead of learned linear projections W_q, W_k, W_v, we apply
   UNITARY ROTATIONS: multiply by e^(iθ_q), e^(iθ_k), e^(iθ_v).
   In quantum mechanics, these are single-qubit rotations on the Bloch sphere.
   The rotation angles are learned — the system discovers which rotations
   make interference patterns most informative.

3. WAVE INTERFERENCE (Attention)
   Q·K† (conjugate transpose) computes the complex inner product.
   The REAL PART of this = cos(θ_q - θ_k) · |q|·|k|
   This IS constructive/destructive interference:
   - In-phase signals (θ_q ≈ θ_k) → large positive (constructive)
   - Anti-phase signals (θ_q ≈ θ_k + π) → large negative (destructive)
   - Quadrature (θ_q ≈ θ_k + π/2) → cancellation
   Softmax over the real part = observation probability distribution.

4. RESONANCE (Feed-Forward)
   Complex matrix multiply with W = W_real + i·W_imag.
   Orthogonal initialization → approximately unitary → preserves norm.
   This is a ROTATION in the full complex Hilbert space,
   not a squash-and-project like ReLU(Wx+b).

5. COMPLEX NORMALIZATION
   Normalize magnitude while PRESERVING PHASE.
   tanh(normalized_magnitude) · (original_z / |original_z|)
   Phase is the information carrier — we must not destroy it.

6. MEASUREMENT (Readout)
   Project real and imaginary parts separately to class logits.
   Analogous to measuring in the computational basis AND the Hadamard basis,
   then combining — extracting maximal information from the quantum state.

Experiment: MNIST (28x28), 16 patches of 7x7, embed_dim=64, convergence loop.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import math
import time
import json
import os
import numpy as np
from datasets import load_dataset


# ============================================================
# ARCHITECTURE
# ============================================================

class ComplexNorm(nn.Module):
    """Normalize magnitude, preserve phase exactly.

    Physics: In a quantum system, normalization ensures unit probability
    (|ψ|² = 1). Here we normalize magnitude statistics across the
    embedding dimension while keeping the phase angle untouched.
    The phase carries positional and relational information — destroying
    it would be like randomizing the spin of every particle.
    """
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        mag = torch.abs(x)
        mean_mag = mag.mean(dim=-1, keepdim=True)
        std_mag = mag.std(dim=-1, keepdim=True)
        norm_mag = (mag - mean_mag) / (std_mag + self.eps)
        # Scale original complex number: preserves phase, replaces magnitude
        scale = torch.tanh(norm_mag) / (mag + self.eps)
        return x * scale.to(torch.complex64)


class PhaseMNIST(nn.Module):
    """Phase-interference classifier for MNIST.

    Config:
        patch_pixels: pixels per patch (7x7 = 49 for MNIST)
        num_patches:  patches per image (4x4 grid = 16)
        embed_dim:    complex embedding dimension
        max_steps:    max convergence iterations (eigenstate search budget)
        tol:          convergence threshold (when to stop iterating)
    """
    def __init__(self, patch_pixels=49, num_patches=16, embed_dim=64,
                 num_classes=10, max_steps=4, tol=1e-3):
        super().__init__()
        self.patch_pixels = patch_pixels
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        self.max_steps = max_steps
        self.tol = tol

        # --- Encoding ---
        # Project raw pixel patch into complex magnitude space
        self.patch_proj = nn.Linear(patch_pixels, embed_dim)

        # Positional phase: multi-frequency sinusoidal
        # Each patch position gets a unique phase signature across dimensions,
        # like a unique "spin configuration" that identifies where it is
        pos = torch.arange(num_patches).unsqueeze(1).float()
        dim = torch.arange(embed_dim).unsqueeze(0).float()
        freq = 1.0 / (10000 ** (dim / embed_dim))
        self.register_buffer('pos_phase', pos * freq * math.pi)

        # --- Phase rotations (Q/K/V) ---
        # These are the learned unitary operators
        # Initialized uniform in [-π, π] — full Bloch sphere coverage
        self.q_rot = nn.Parameter(torch.empty(embed_dim).uniform_(-math.pi, math.pi))
        self.k_rot = nn.Parameter(torch.empty(embed_dim).uniform_(-math.pi, math.pi))
        self.v_rot = nn.Parameter(torch.empty(embed_dim).uniform_(-math.pi, math.pi))

        # --- Resonance matrix (complex feed-forward) ---
        # Orthogonal init → approximately unitary → norm-preserving rotation
        w_init = torch.empty(embed_dim, embed_dim)
        nn.init.orthogonal_(w_init)
        self.ff_real = nn.Parameter(w_init)
        self.ff_imag = nn.Parameter(torch.empty(embed_dim, embed_dim).uniform_(-0.01, 0.01))

        self.norm = ComplexNorm()

        # --- Measurement ---
        # Dual-basis readout: project real and imaginary parts separately
        self.readout_real = nn.Linear(embed_dim, num_classes)
        self.readout_imag = nn.Linear(embed_dim, num_classes)

    def forward(self, x, return_diagnostics=False):
        """
        x: (batch, num_patches, patch_pixels) — raw pixel patches
        returns: (batch, num_classes) logits
        """
        batch_size = x.shape[0]

        # 1. WAVEFUNCTION ENCODING
        # magnitude = tanh(linear(pixels)) — bounded activation
        # phase = positional encoding — unique per patch position
        mag = torch.tanh(self.patch_proj(x))
        state = mag.to(torch.complex64) * torch.exp(1j * self.pos_phase.unsqueeze(0))

        ff_weights = torch.complex(self.ff_real, self.ff_imag)

        convergence_steps = self.max_steps  # track actual steps used

        for step in range(self.max_steps):
            prev_state = state

            # 2. PHASE ROTATION — apply learned unitary operators
            q = state * torch.exp(1j * self.q_rot)
            k = state * torch.exp(1j * self.k_rot)
            v = state * torch.exp(1j * self.v_rot)

            # 3. WAVE INTERFERENCE
            # Q · K† = complex inner product
            # .real = Re(⟨q|k⟩) = constructive/destructive interference
            interference = torch.matmul(q, k.conj().transpose(-1, -2))
            attn_logits = interference.real / math.sqrt(self.embed_dim)
            attn_weights = torch.softmax(attn_logits * 8.0, dim=-1)

            # Weighted superposition of value states
            attn_out = torch.matmul(attn_weights.to(torch.complex64), v)
            state = state + attn_out  # residual = superposition accumulation

            # 4. RESONANCE — complex rotation in full Hilbert space
            state = torch.matmul(state, ff_weights)
            state = self.norm(state)

            # 5. CONVERGENCE CHECK — has the system found an eigenstate?
            diff = torch.norm(state - prev_state) / (torch.norm(state) + 1e-8)
            if diff.item() < self.tol:
                convergence_steps = step + 1
                break

        # 6. MEASUREMENT
        # Pool across patches (average over spatial positions)
        # Then observe in two bases: real and imaginary
        pooled = state.mean(dim=1)  # (batch, embed_dim) complex
        logits = self.readout_real(pooled.real) + self.readout_imag(pooled.imag)

        if return_diagnostics:
            return logits, convergence_steps, diff.item()
        return logits


# ============================================================
# DATA — MNIST via HuggingFace, patched into 4x4 grid of 7x7
# ============================================================

def load_mnist_patches(max_train=10000, max_test=2000):
    """Load MNIST and slice 28x28 images into 16 patches of 7x7 pixels.

    Vectorized — no Python loops over images. Loads as numpy, patches via reshape.
    """
    print("Loading MNIST from HuggingFace...", flush=True)
    ds = load_dataset("ylecun/mnist")

    def process_split(split, max_n):
        # Convert PIL images to numpy array in one shot
        imgs = np.stack([np.array(split[i]["image"]) for i in range(min(max_n, len(split)))])
        labels = np.array([split[i]["label"] for i in range(min(max_n, len(split)))])

        # imgs: (N, 28, 28) uint8 → float32 normalized
        imgs = imgs.astype(np.float32) / 255.0

        # Vectorized patching: reshape 28x28 → 4x7 x 4x7 → 16 patches of 49
        N = imgs.shape[0]
        # (N, 4, 7, 4, 7) then transpose to (N, 4, 4, 7, 7) then flatten patches
        patches = imgs.reshape(N, 4, 7, 4, 7).transpose(0, 1, 3, 2, 4).reshape(N, 16, 49)

        return torch.tensor(patches), torch.tensor(labels, dtype=torch.long)

    t0 = time.time()
    X_train, y_train = process_split(ds["train"], max_train)
    X_test, y_test = process_split(ds["test"], max_test)
    print(f"  Data loaded in {time.time()-t0:.1f}s", flush=True)
    print(f"  Train: {X_train.shape} -> {y_train.shape}")
    print(f"  Test:  {X_test.shape} -> {y_test.shape}")
    return X_train, y_train, X_test, y_test


# ============================================================
# TRAINING — with full diagnostics
# ============================================================

def p(msg):
    """Print with flush — critical for monitoring background training."""
    print(msg, flush=True)


def train():
    p("=" * 60)
    p("Phase-Interference MNIST Experiment")
    p("=" * 60)

    # --- Data (small first — prove it learns, then scale) ---
    X_train, y_train, X_test, y_test = load_mnist_patches(max_train=2000, max_test=500)

    # --- Model ---
    model = PhaseMNIST(
        patch_pixels=49,
        num_patches=16,
        embed_dim=64,
        max_steps=4,
        tol=1e-3,
    )
    total_params = sum(p_.numel() for p_ in model.parameters())
    p(f"\nModel: {total_params:,} parameters")
    p(f"  embed_dim=64, max_steps=4, 16 patches of 7x7")

    # --- Training config ---
    epochs = 30
    batch_size = 64
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    steps_per_epoch = (X_train.shape[0] + batch_size - 1) // batch_size
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=5e-3,
        steps_per_epoch=steps_per_epoch, epochs=epochs
    )
    criterion = nn.CrossEntropyLoss()

    # --- Logging ---
    log = {
        "config": {
            "embed_dim": 64, "max_steps": 4, "tol": 1e-3,
            "epochs": epochs, "batch_size": batch_size,
            "train_samples": X_train.shape[0],
            "test_samples": X_test.shape[0],
            "total_params": total_params,
        },
        "epochs": []
    }

    p(f"\nTraining for {epochs} epochs, {steps_per_epoch} steps/epoch\n")

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

        # --- Evaluation with diagnostics ---
        model.eval()
        with torch.no_grad():
            # Test accuracy
            test_logits, conv_steps, final_diff = model(X_test, return_diagnostics=True)
            pred = torch.argmax(test_logits, dim=1)
            acc = (pred == y_test).float().mean().item()

            # Phase angle statistics
            q_angles = model.q_rot.detach()
            k_angles = model.k_rot.detach()
            v_angles = model.v_rot.detach()

        epoch_log = {
            "epoch": epoch,
            "train_loss": round(avg_loss, 4),
            "test_acc": round(acc, 4),
            "convergence_steps": conv_steps,
            "convergence_diff": round(final_diff, 6),
            "q_rot_std": round(q_angles.std().item(), 4),
            "k_rot_std": round(k_angles.std().item(), 4),
            "v_rot_std": round(v_angles.std().item(), 4),
            "epoch_seconds": round(elapsed, 1),
        }
        log["epochs"].append(epoch_log)

        # Print every epoch for small runs
        p(f"Epoch {epoch:2d} | Loss {avg_loss:.4f} | Acc {acc:.2%} | "
          f"Conv {conv_steps}/{model.max_steps} (Δ={final_diff:.4f}) | "
          f"{elapsed:.1f}s")

    # --- Final diagnostics ---
    p("\n" + "=" * 60)
    p("FINAL RESULTS")
    p("=" * 60)
    best_epoch = max(log["epochs"], key=lambda e: e["test_acc"])
    p(f"Best accuracy: {best_epoch['test_acc']:.2%} (epoch {best_epoch['epoch']})")
    p(f"Final convergence: {log['epochs'][-1]['convergence_steps']}/{model.max_steps} steps")

    # Phase angle analysis
    p(f"\nPhase Rotation Analysis (learned angles):")
    p(f"  Q rotation std: {q_angles.std():.4f} (range [{q_angles.min():.2f}, {q_angles.max():.2f}])")
    p(f"  K rotation std: {k_angles.std():.4f} (range [{k_angles.min():.2f}, {k_angles.max():.2f}])")
    p(f"  V rotation std: {v_angles.std():.4f} (range [{v_angles.min():.2f}, {v_angles.max():.2f}])")

    # Phase ablation: zero all phases, re-evaluate
    p(f"\nPhase Ablation Test (set all θ=0, measure accuracy drop):")
    with torch.no_grad():
        saved_q = model.q_rot.data.clone()
        saved_k = model.k_rot.data.clone()
        saved_v = model.v_rot.data.clone()

        model.q_rot.data.zero_()
        model.k_rot.data.zero_()
        model.v_rot.data.zero_()

        ablated_logits = model(X_test)
        ablated_pred = torch.argmax(ablated_logits, dim=1)
        ablated_acc = (ablated_pred == y_test).float().mean().item()

        model.q_rot.data.copy_(saved_q)
        model.k_rot.data.copy_(saved_k)
        model.v_rot.data.copy_(saved_v)

    p(f"  With phase:    {best_epoch['test_acc']:.2%}")
    p(f"  Without phase: {ablated_acc:.2%}")
    p(f"  Phase contribution: {best_epoch['test_acc'] - ablated_acc:+.2%}")
    if ablated_acc < best_epoch['test_acc'] - 0.01:
        p(f"  → PHASE IS LOAD-BEARING. The complex plane is doing real work.")
    else:
        p(f"  → Phase may be decorative. Magnitude alone carries the signal.")

    # Save
    log["final"] = {
        "best_acc": best_epoch["test_acc"],
        "ablated_acc": round(ablated_acc, 4),
        "phase_contribution": round(best_epoch["test_acc"] - ablated_acc, 4),
    }

    results_path = os.path.join(os.path.dirname(__file__), "phase_mnist_results.json")
    with open(results_path, "w") as f:
        json.dump(log, f, indent=2)
    p(f"\nFull log saved to {results_path}")

    model_path = os.path.join(os.path.dirname(__file__), "phase_mnist_model.pth")
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {
            "patch_pixels": 49, "num_patches": 16, "embed_dim": 64,
            "max_steps": 4, "tol": 1e-3,
        },
        "results": log["final"],
    }, model_path)
    p(f"Model saved to {model_path}")


if __name__ == "__main__":
    train()
