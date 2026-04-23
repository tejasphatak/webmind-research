# Social Posts — Machine Learning Focused

DOI: 10.17605/OSF.IO/7X8AS
GitHub: github.com/tejasphatak/webmind-research/releases/tag/cos-delta-phi-v1

---

## Reddit r/MachineLearning

**Title:** [R] Phase rotations replace Q/K/V projections: 98.54% MNIST with 768 attention parameters (585 KB model)

**Body:**

We replaced standard attention's Q/K/V linear projections (O(d²) params) with phase rotations on complex-valued states (O(d) params). The attention map comes from Re(Q·K†) = Σ|Q|·|K|·cos(Δθ), which is the wave interference cross-term.

**Results:**

| | Params | Accuracy/PPL |
|---|---|---|
| 1-head, d=64, MNIST | 192 attn | 95.15% |
| 8-head, d=256, MNIST | 768 attn | 98.54% |
| 8-head, d=512, char language | 1536 attn | PPL 2.0 |

That's 256× fewer attention parameters than standard MHA (768 vs 196,608 for d=256).

**Key finding — phase ablation:**
Zero all rotation angles after training → accuracy drops from 98.54% to 41.44%. The cos(Δθ) term carries 57% of the discriminative signal. Standard attention uses Q·K^T which is Re(Q·K†) with zero phase — it discards this term entirely.

**Other findings:**
- Single interference step is optimal. More steps degrade accuracy (92.6% > 91.4% > 86.4% for 1/2/4 steps). The "convergence loop" hypothesis was falsified by linear probes showing info exists from step 1.
- The same formula produces spatial structure when applied to a 2D complex field initialized at ε = 10⁻¹⁰.

**Limitations (I'll save you the trouble):**
- MNIST and char-level PPL are toy benchmarks
- FF layer still has d² params — total model reduction is moderate
- No parameter-matched standard transformer baseline on identical data
- Complex arithmetic is slower per-op than real on CPU
- No multi-layer experiments

**But:** 768 attention params achieving 98.54% is real. The ablation is real. The code is public.

```
pip install torch datasets
git clone https://github.com/tejasphatak/webmind-research
cd webmind-research/playground
python phase_mnist_multihead.py
```

Paper: https://doi.org/10.17605/OSF.IO/7X8AS

---

## Reddit r/LocalLLaMA

**Title:** 2.4 MB language model (PPL 2.0) using wave interference instead of standard attention — runs on CPU

**Body:**

Built a char-level language model that uses phase rotations + interference instead of Q/K/V projections.

- 620K parameters, 2.4 MB
- PPL 2.0 on mixed corpus (Wikipedia, StackOverflow, Q&A, stories, code, medicine)
- 768 attention parameters total (the rest is embedding + FF)
- Trained on RTX 3090 in ~2 hours
- Inference runs on CPU

Generation sample: "Once upon a time, there was a little girl named Lily. She loved to play with her toys and the li..."

It's char-level so don't expect GPT-quality. But PPL 2.0 with 2.4 MB is interesting for on-device applications.

Combined it with a retrieval KB (MiniLM + cosine search). Teach it a fact → instantly available. No retraining. The 2.4 MB model is the "thinking" module, the KB is the knowledge. Total system: ~85 MB including MiniLM.

Code: github.com/tejasphatak/webmind-research/tree/master/playground

---

## Twitter/X (ML thread)

**1/3:**
Replaced attention Q/K/V projections with phase rotations on complex-valued states.

768 attention params → 98.54% MNIST
Standard MHA would need 196,608 params

256× reduction. Phase ablation: zero the rotations → 41.44%. The cos(Δθ) term carries 57%.

Code: github.com/tejasphatak/webmind-research

**2/3:**
Char-level language model: PPL 2.0, 2.4 MB, runs on CPU.

The attention mechanism is Re(Q·K†) = Σ|Q|·|K|·cos(Δθ) — literally the wave interference formula. Standard attention is this with phase set to zero.

**3/3:**
Single interference step > 4 steps (92.6% vs 86.4%). The "convergence loop" hypothesis is falsified.

Linear probes show 82% accuracy at step 1 — info exists immediately. Extra steps just rotate the representation basis.

DOI: 10.17605/OSF.IO/7X8AS

---

## Bluesky (ML)

Phase rotations replace attention projections: 768 params → 98.54% MNIST. 256× fewer than standard MHA.

Phase ablation: zero the rotations → 41.44%. cos(Δθ) carries 57% of the signal.

2.4 MB language model, PPL 2.0, runs on CPU.

DOI: 10.17605/OSF.IO/7X8AS
