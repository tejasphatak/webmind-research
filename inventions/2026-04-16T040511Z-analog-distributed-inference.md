---
name: Analog Compute-in-Memory Tiles for Distributed Volunteer Inference
description: Speculative research direction — pluggable analog NN tiles (memristor/PCM crossbars) as the compute substrate for Synapse-style decentralized inference networks. Timestamped for priority; not a current research commitment.
type: project
status: SPECULATIVE — TIMESTAMPED FOR PRIORITY, NOT UNDER ACTIVE DEVELOPMENT
---

**Inventor:** Tejas Phatak (Webmind Research)
**Recorded:** 2026-04-16
**Status:** Speculative research direction. Timestamped for priority claim. Under the ROADMAP kill list for 2026-2027 but retained for future pursuit.

---

## The Core Idea

Build **pluggable analog neural-network tiles** — discrete hardware units where the neurons are programmable analog components (memristors, phase-change cells, floating-gate flash, or other variable-conductance devices), and matrix-vector multiplication happens via Kirchhoff's laws rather than digital arithmetic. These tiles then participate in a Synapse-style decentralized inference mesh, with shards mapped across heterogeneous analog tiles contributed by volunteers.

The intuition: digital GPUs are energy-wasteful for inference (most energy goes to moving bits, not computing). Analog compute-in-memory (CIM) is orders of magnitude more energy-efficient per MAC. If every user had a small analog accelerator as a USB stick or phone co-processor, a global inference mesh of these could run large models at ~0.1x the energy cost of datacenter inference.

---

## Prior Art (acknowledging)

This is **not a new idea** at the device level. Compute-in-memory with crossbar arrays has been actively researched since ~2015:

- Burr et al., *Nature* 2015 — large-scale NN with non-volatile memory crossbars
- Ambrogio et al., *Nature* 2018 — analog training with PCM
- Mythic AI (2014-2022) — commercial analog AI chips using flash CIM (company failed, tech was real)
- IBM Research — ongoing PCM and RRAM neural accelerator work
- Hung et al., *Nature Electronics* 2021 — 4-Mbit CIM macro
- Encharge AI, Rain AI, Syntiant — current startups in the space

**What is claimed here is not the analog tile itself**, but the **specific intersection with volunteer-based decentralized inference** (Synapse/webmind.sh model).

---

## The Novel Angle

Existing analog CIM research assumes:
- Datacenter deployment (centralized)
- Or phone/edge deployment (single-user)

What's missing:
- **Distributed analog inference** — a mesh of heterogeneous volunteer analog devices, each with different tile configurations, different noise profiles, different bit-accuracies, coordinated over a network.

Specific open problems at this intersection:

1. **Analog-noise-aware shard placement.** Each volunteer tile has a characteristic noise profile (drift, non-linearity, bit-equivalent accuracy). How do you map transformer layers to tiles such that cumulative error stays within tolerance? This is a graph-partitioning problem with device-physics constraints. Extension of our Carrier-Payload work.

2. **Activation transport across analog tile boundaries.** Between tiles (different devices), activations must be digitized (ADC), transmitted, re-converted (DAC). Carrier-Payload compression is even MORE valuable here because ADC/DAC energy is often the dominant cost.

3. **Continuous calibration protocol.** Memristors drift over time and temperature. A decentralized network of drifting tiles needs a protocol to periodically re-measure each tile's actual behavior and re-program weights. This is related to Byzantine verification (Artifact 2).

4. **Federated analog-aware training.** If the network eventually self-trains (Nexus trajectory), the gradients must account for which tile they're going to be deployed on. Training with analog-noise models is an open problem.

5. **Analog-specific sharding topology.** Analog tiles have natural granularity (e.g., 256×256 crossbars). Transformers don't map cleanly to these sizes. Architecture search over analog-native transformers.

---

## Why This Is Parked (Not Active)

Per ROADMAP (2026-04-16), this project is explicitly **on the kill list** for the 18-month EB1-A portfolio window. Reasons:

1. **Timeline mismatch.** Hardware research has 5-10 year cycles. EB1-A filing target is Q1 2028.
2. **Fab access.** Novel hardware requires $10-100M+ in fab access. Not realistic at CU Boulder / independent level.
3. **Prior-art overwhelm.** 10 years of analog CIM literature must be absorbed before contributing. 6 months of reading before writing.
4. **Narrative coherence.** Current papers (Carrier-Payload, WebGPU Parity, Distributed Speculative Decoding) form a tight story around distributed software inference. Adding analog hardware dilutes this.
5. **Not mature for software-only contribution.** Even simulation-based analog work requires realistic device models; these exist but are proprietary to groups with hardware.

---

## What IS Actionable (If We Ever Return)

### Path A — Simulation-based architecture study (6 months, publishable)
Use NeuroSim or DNN+NeuroSim (open-source, from Georgia Tech) to model an analog tile fleet. Answer:
- Optimal tile size for transformer attention/FFN mapping
- Noise budget across cascaded tiles
- Carrier-Payload compression integrated with ADC/DAC quantization

### Path B — Analog-noise-robust training (3-month paper)
Train transformers in PyTorch with injected analog-style noise at each linear layer. Show robustness training improves deployment accuracy on analog hardware by X%. Open-source framework.

### Path C — Position paper (6-week effort)
"Decentralized Analog Inference: A Research Agenda." Position piece laying out the 5 open problems above. Appears in a workshop or as a white paper. Doesn't need experiments. Stakes intellectual claim.

### Path D — Partnership with hardware group
Partner with an academic lab that has analog hardware (e.g., Wei Lu at Michigan, Stanley Williams-alumni groups). Contribute software/systems expertise in exchange for measurement access. Legitimate path but requires relationships.

---

## Relation to Other Webmind Research

- **Extends Carrier-Payload (P1):** compression is MORE valuable at analog tile boundaries than digital shard boundaries (ADC/DAC energy cost).
- **Extends Distributed Speculative Decoding (P3):** the draft model could run on low-bit-accuracy analog tiles, verification on high-accuracy digital nodes. Hybrid architecture.
- **Extends Nexus (self-learning trajectory):** analog tiles have a physics-grounded reason why continual learning matters — devices drift and must re-learn.
- **Extends Threat Model:** analog devices have new Byzantine modes (physical tampering, device-level fingerprinting) that don't exist in software.

---

## Priority Claim

This document records the specific idea of **"pluggable analog neural-network tiles coordinated via a decentralized volunteer inference protocol"** as of 2026-04-16. The individual constituents (analog CIM, decentralized inference) are pre-existing; the intersection and specific protocol design is recorded here for future development by the author.

No active experimental or engineering work is claimed as of this date. This is an intellectual placeholder.

---

## Honest Limits

- I am not a device physicist or chip designer.
- I do not have fab access.
- This document was written based on literature review, not original hardware research.
- The "novelty" claimed is at the intersection of known fields, not in device innovation.
- If someone else publishes this exact intersection first, I will cede priority gracefully.

---

## Next Actions (if ever pursued)

1. Read Burr 2015, Ambrogio 2018, Hung 2021 end-to-end
2. Install NeuroSim, reproduce one published baseline
3. Draft Path C position paper as 6-week side project
4. Reach out to an analog AI research group for collaboration

Until then, **parked**.
