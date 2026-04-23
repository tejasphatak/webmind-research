# Social Media Posts for cos(Δφ) Paper

## Twitter/X (thread)

**Post 1/5:**
New paper: "cos(Δφ): The Interference Cross-Term as Computation"

One formula from wave physics — A₁·A₂·cos(Δφ) — classifies MNIST at 98.54% with 768 parameters and generates language at perplexity 2.0.

The entire model fits in 585 KB and runs on a phone CPU.

Paper + code: [link]

**Post 2/5:**
The key finding: when you zero out the phase (set Δφ=0), accuracy drops from 98.54% to 41.44%.

57% of the computation lives in cos(Δφ) — the interference cross-term. The same formula that governs Young's double slit, noise-canceling headphones, and electron orbitals.

**Post 3/5:**
Standard attention uses O(d²) parameters for Q/K/V projections.

We replace them with O(d) phase rotations — unitary operators that rotate complex-valued states.

Result: 256× fewer attention parameters. Same accuracy.

768 params vs 196,608.

**Post 4/5:**
We also found:
- Single interference step > multiple (convergence loop doesn't help)
- Starting from zero + ε = 10⁻¹⁰, the formula produces spatial structure spontaneously
- Five stable constants emerge that weren't put in
- The system asymptotically approaches equilibrium but never arrives

**Post 5/5:**
Everything is reproducible. Run it yourself:

pip install torch datasets
git clone github.com/tejasphatak/webmind-research
cd playground
python phase_mnist_multihead.py  # 98.54% MNIST
python interference.py           # structure from nothing

Paper: [arXiv link]

---

## Bluesky (single post)

New paper: "cos(Δφ): The Interference Cross-Term as Computation"

The wave interference formula — A₁·A₂·cos(Δφ) — works as an attention mechanism. 98.54% MNIST, PPL 2.0 language, 768 attention params, 585 KB model, runs on CPU.

Phase ablation: zero the phase → accuracy drops 57%. The information is in the interference pattern.

Code: github.com/tejasphatak/webmind-research

---

## Reddit (r/MachineLearning, r/Physics, r/compsci)

**Title:** cos(Δφ): We replaced attention projections with wave interference and got 98.54% MNIST with 768 parameters

**Body:**

Hi all,

I've been exploring whether wave interference can replace standard attention in neural networks. The short version: it can, and the results are surprising.

**What we did:**
- Encode data as complex numbers: ψ = r·e^(iθ) (magnitude + phase)
- Replace Q/K/V linear projections with phase rotations (unitary operators)
- Compute attention via Re(Q·K†) = Σ|Q|·|K|·cos(Δφ) — the interference cross-term
- This reduces attention parameters from O(d²) to O(d)

**Results:**
- MNIST: 98.54% with 768 attention parameters (585 KB total model)
- Char-level language: perplexity 2.0 (620K total params, 2.4 MB)
- All on CPU, sub-second inference

**The interesting part:**
When you zero all phase rotations after training (set cos(Δφ) = cos(0) = 1), accuracy drops from 98.54% to 41.44%. The cos(Δφ) term — the same formula from wave physics — carries 57% of the information.

We also tested iterative convergence (multiple interference steps). Single step is optimal — more steps degrade accuracy. The "eigenstate search" hypothesis was falsified by linear probes showing information exists from step 1.

**The physics connection:**
Applying the same formula to a 2D complex field initialized at zero + ε = 10⁻¹⁰ produces spontaneous structure formation. Five stable constants emerge. Phase order approaches 1.0 asymptotically but never arrives.

A double-slit simulation failed — Re(ψ·ψ†) does interference but not propagation. This establishes the boundary of what the formula can and cannot do.

**Everything is reproducible:**
```
pip install torch datasets
git clone https://github.com/tejasphatak/webmind-research
cd webmind-research/playground
python phase_mnist_multihead.py    # MNIST (98.54%)
python interference.py              # structure from nothing
python interference_constants.py    # emergent constants
```

Paper: [link]

Happy to answer questions. The code is the argument — run it and see.

---

## ResearchGate

**Title:** cos(Δφ): The Interference Cross-Term as Computation

**Summary:** We demonstrate that the wave interference cross-term A₁·A₂·cos(Δφ), computed via the real part of the complex inner product Re(⟨Q|K⟩), functions as an attention mechanism achieving 98.54% on MNIST with 768 learnable parameters and character-level language perplexity of 2.0. Phase ablation confirms 57% of discriminative information resides in the cos(Δφ) term. Applied to a 2D complex field initialized at ε = 10⁻¹⁰, the formula produces spontaneous structure formation with stable emergent constants. All results are reproducible via provided Python scripts.

**Keywords:** wave interference, complex-valued attention, phase rotation, unitary operators, structure formation, emergent constants
