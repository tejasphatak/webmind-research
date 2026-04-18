# Webmind Research — Context for Gemini

## Quick context

You are looking at a research project that proposes a new AI architecture: **self-evolving retrieval**. It is not an LLM. It does not generate text. It understands queries with a small encoder (22M params), searches a growing knowledge base, and learns from every interaction.

## The claim

Traditional LLMs use 175B+ parameters to memorize facts and generate text. Most of those parameters store facts that belong in a database. This system uses:
- **22M parameters** for language understanding (sentence transformer)
- **A database** for knowledge (300K+ verified Q&A pairs, grows autonomously)
- **A convergence loop** for reasoning (fixed-point iteration in embedding space)

Result: 0.7% → 25.3% exact match on held-out QA benchmarks through self-evolution. Zero human intervention.

## Your role

You have been used as a reviewer and research advisor for this project. Previous sessions asked you to:
- Validate the architecture against prior art
- Review the paper for scientific rigor
- Identify gaps (missing baselines, overclaimed novelty, etc.)
- Suggest the right venue (workshop/demo paper, not main conference)

Key feedback you gave:
1. "Third architecture class" framing is overclaimed — needs baselines
2. The 71.3% number is memorization (same questions tested after learning) — report the 25.3% held-out number honestly
3. Don't call it "RLHF" — it's knowledge base augmentation
4. Best venue: demo paper at EMNLP/ACL, or workshop paper
5. The HotPotQA 0→72% generalization result is the strongest finding

## Repo structure

```
papers/                     ← Research papers
  self-evolving-retrieval-2026-04-18.md  ← MAIN PAPER
  review-self-evolving-retrieval.md      ← Your review (4/10)
benchmarks/                 ← All benchmark JSON + summaries
mindmap/                    ← How everything connects (raw ideas)
inventions/                 ← Timestamped invention disclosures
findings/                   ← Experiment results
tools/                      ← Research tools and prototypes
trained_model/              ← KB data and embeddings
```

## Architecture

```
Query → Encoder (MiniLM, 22M params, 384 dims)
     → FAISS search → top-K candidates
     → Bi-embedding re-rank (encode answer, compare with query)
     → If weak → web search (Wikipedia + DuckDuckGo, parallel)
     → Learn answer → KB grows
     → Convergence loop (iterate until answer embedding stabilizes)
```

## What needs your help

1. **Paper review:** Is the revised paper (with honest framing + human analogy) closer to publishable?
2. **Missing references:** NELL, PAQ, REALM, poly-encoders, KILT — are these the right comparisons?
3. **Experimental design:** Is the held-out evaluation (offset=500, different questions) methodologically sound?
4. **Architecture critique:** Is the convergence loop (fixed-point iteration) genuinely novel in retrieval?
5. **Scale questions:** Will this architecture hold at 1M+ pairs? 10M? What breaks?

## Live demo

https://webmind.sh — browser-native, 214MB download, works offline after first load.

## Code

https://github.com/tejasphatak/Synapse (docs/ directory for browser engine)
