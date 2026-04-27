# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A reasoning-only model that feels like an LLM but stores zero knowledge in parameters. T5-small handles language (encode/decode), a small iterative reasoning module (~7M params) handles thinking, and all knowledge comes from FAISS DB + internet search.

## Architecture

```
Query → T5 Encoder (frozen) → query embedding
     → FAISS / Web search → top-K passages
     → T5 Encoder (frozen) → passage embeddings
     → Iterative Reasoning Loop (2 shared blocks, looped until convergence)
         - Convergence Gate: halt when representation stabilizes
         - Search Decision Head: re-retrieve if evidence insufficient
     → T5 Decoder (frozen) → generated answer
         - Sufficiency check: decoder can request more evidence mid-generation
```

The reasoning loop uses shared weights looped N times (not N separate layers). Easy queries converge in 1 iteration, multi-hop reasoning takes 3-6. Same convergence pattern as Synapse/SAQT.

## Commands

```bash
# Training (each phase separately)
python train.py --phase 1   # Single-hop QA (SQuAD)
python train.py --phase 2   # Multi-hop reasoning (HotPotQA)
python train.py --phase 3   # Long-form generation (ELI5)
python train.py --phase 4   # Internet search integration

# Inference
python infer.py              # FAISS retrieval
python infer.py --web        # Web search only
python infer.py --hybrid     # FAISS + web fallback
python infer.py --phase 2    # Load phase 2 checkpoint

# GPU training (RunPod)
python run_gpu.py            # Create A100 pod, run all phases
python run_gpu.py --phase 1  # Run specific phase
```

## File structure

- `model.py` — LMRAG: T5 + IterativeReasoningLayer + ConvergenceGate + SufficiencyClassifier
- `train.py` — Phase-aware training with composite losses and curriculum
- `infer.py` — Interactive Q&A with convergence trace display
- `config.py` — Model and training configs per phase
- `retriever.py` — FAISS, Web (DuckDuckGo), and Hybrid retrievers
- `datasets.py` — SQuAD, HotPotQA, ELI5 loaders + KB builder
- `losses.py` — Ponder cost, search decision, sufficiency losses
- `run_gpu.py` — RunPod A100 launcher + remote training orchestrator

## Relationship to other work

Extends Synapse/SAQT (parent repo + github.com/tejasphatak/Synapse). SAQT does retrieval + convergence loop but returns stored answers verbatim. This project adds a generative decoder to synthesize novel answers, and trains the reasoning loop to handle multi-hop queries.

## Design principles

- The database IS the knowledge. The model only understands language.
- Reasoning = convergence loop. Same block, variable depth.
- No hardcoding — behaviors and thresholds come from data.
- Honest benchmarking — report in-distribution and held-out scores separately.
