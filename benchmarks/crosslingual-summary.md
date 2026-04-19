# Cross-Lingual Retrieval Benchmark — Baseline

**Date:** 2026-04-18T16:40:30.728Z
**Endpoint:** https://chat.webmind.sh (tested via localhost:3002)
**Current model:** all-MiniLM-L6-v2 (English-only)
**Target model:** paraphrase-multilingual-MiniLM-L12-v2

## Summary

The current English-only encoder (MiniLM-L6-v2) is expected to fail on non-English queries.
This baseline measures that failure rate so we can quantify improvement after deploying
paraphrase-multilingual-MiniLM-L12-v2.

## Overall Results

| Metric | Value |
|--------|-------|
| Total pairs | 47 |
| Exact match rate (same KB entry) | 0.0% |
| Same top fact rate | 0.0% |
| High answer overlap (>50%) | 0.0% |
| Avg answer similarity | 0.014 |
| Avg EN confidence | 1.278 |
| Avg XL confidence | 0.625 |
| Confidence delta (EN - XL) | +0.653 |

## By Language

| Language | Pairs | Exact Match | Same Fact | Avg Similarity | Conf EN | Conf XL | Delta |
|----------|-------|-------------|-----------|----------------|---------|---------|-------|
| Hindi | 12 | 0.0% | 0.0% | 0.026 | 1.220 | 0.601 | +0.619 |
| Marathi | 15 | 0.0% | 0.0% | 0.007 | 1.375 | 0.571 | +0.804 |
| Spanish | 10 | 0.0% | 0.0% | 0.016 | 1.240 | 0.737 | +0.503 |
| French | 10 | 0.0% | 0.0% | 0.008 | 1.240 | 0.622 | +0.618 |

## Detailed Results

| ID | Lang | EN Question | Match? | Answer Sim | EN Conf | XL Conf |
|----|------|-------------|--------|------------|---------|--------|
| 1 | hi | What is Python? | DIFF | 0.000 | 4.073 | 0.525 |
| 2 | hi | How does a neural network work? | ERROR | - | - | - |
| 3 | hi | What is machine learning? | ERROR | - | - | - |
| 4 | hi | What is a database? | ERROR | - | - | - |
| 5 | hi | How to sort an array? | DIFF | 0.000 | 0.915 | 0.529 |
| 6 | hi | What is an API? | DIFF | 0.018 | 1.000 | 0.587 |
| 7 | hi | What is a linked list? | DIFF | 0.000 | 1.000 | 0.585 |
| 8 | hi | What is recursion? | DIFF | 0.057 | 1.000 | 0.649 |
| 9 | hi | What is HTML? | DIFF | 0.000 | 1.000 | 0.623 |
| 10 | hi | What is cloud computing? | DIFF | 0.085 | 0.861 | 0.678 |
| 11 | hi | What is an operating system? | DIFF | 0.057 | 0.981 | 0.618 |
| 12 | hi | What is encryption? | DIFF | 0.050 | 0.853 | 0.564 |
| 13 | hi | What is a compiler? | DIFF | 0.043 | 0.975 | 0.587 |
| 14 | hi | What is JavaScript? | DIFF | 0.000 | 1.000 | 0.614 |
| 15 | hi | What is a binary tree? | DIFF | 0.000 | 0.984 | 0.658 |
| 16 | mr | What is Python? | DIFF | 0.024 | 4.073 | 0.524 |
| 17 | mr | How does a neural network work? | DIFF | 0.000 | 0.871 | 0.538 |
| 18 | mr | What is machine learning? | DIFF | 0.040 | 1.000 | 0.601 |
| 19 | mr | What is a database? | DIFF | 0.000 | 0.789 | 0.645 |
| 20 | mr | How to sort an array? | DIFF | 0.000 | 0.915 | 0.536 |
| 21 | mr | What is an algorithm? | DIFF | 0.019 | 4.881 | 0.635 |
| 22 | mr | What is a function in programming? | DIFF | 0.000 | 0.788 | 0.580 |
| 23 | mr | What is artificial intelligence? | DIFF | 0.021 | 1.000 | 0.532 |
| 24 | mr | What is a variable? | DIFF | 0.000 | 0.979 | 0.553 |
| 25 | mr | What is the internet? | DIFF | 0.000 | 0.959 | 0.620 |
| 26 | mr | What is object-oriented programming? | DIFF | 0.000 | 0.944 | 0.571 |
| 27 | mr | What is a stack data structure? | DIFF | 0.000 | 0.958 | 0.617 |
| 28 | mr | What is SQL? | DIFF | 0.000 | 0.756 | 0.583 |
| 29 | mr | What is version control? | DIFF | 0.000 | 0.714 | 0.544 |
| 30 | mr | What is a web server? | DIFF | 0.000 | 1.000 | 0.492 |
| 31 | es | What is Python? | DIFF | 0.000 | 4.073 | 0.720 |
| 32 | es | How does a neural network work? | DIFF | 0.000 | 0.871 | 0.768 |
| 33 | es | What is machine learning? | DIFF | 0.038 | 1.000 | 0.799 |
| 34 | es | What is a database? | DIFF | 0.060 | 0.789 | 0.685 |
| 35 | es | What is an operating system? | DIFF | 0.015 | 0.981 | 0.673 |
| 36 | es | What is encryption? | DIFF | 0.010 | 0.853 | 0.700 |
| 37 | es | What is cloud computing? | DIFF | 0.000 | 0.861 | 0.700 |
| 38 | es | What is artificial intelligence? | DIFF | 0.000 | 1.000 | 1.000 |
| 39 | es | What is a compiler? | DIFF | 0.039 | 0.975 | 0.671 |
| 40 | es | What is an API? | DIFF | 0.000 | 1.000 | 0.656 |
| 41 | fr | What is Python? | DIFF | 0.013 | 4.073 | 0.749 |
| 42 | fr | How does a neural network work? | DIFF | 0.000 | 0.871 | 0.566 |
| 43 | fr | What is machine learning? | DIFF | 0.000 | 1.000 | 0.633 |
| 44 | fr | What is a database? | DIFF | 0.000 | 0.789 | 0.538 |
| 45 | fr | What is an operating system? | DIFF | 0.000 | 0.981 | 0.600 |
| 46 | fr | What is encryption? | DIFF | 0.000 | 0.853 | 0.605 |
| 47 | fr | What is cloud computing? | DIFF | 0.000 | 0.861 | 0.639 |
| 48 | fr | What is artificial intelligence? | DIFF | 0.000 | 1.000 | 0.781 |
| 49 | fr | What is a compiler? | DIFF | 0.070 | 0.975 | 0.563 |
| 50 | fr | What is a linked list? | DIFF | 0.000 | 1.000 | 0.547 |

## Interpretation

- **SAME** = English and non-English query hit the same KB entry (matchId match)
- **SIMILAR** = Different KB entries but >50% word overlap in answers
- **DIFF** = Completely different retrieval results

### Expected outcomes

- **Before multilingual deployment:** Most non-English queries should return DIFF results
  with low confidence. The English-only encoder cannot meaningfully embed Hindi/Marathi/etc.
- **After multilingual deployment:** Exact match rate should jump to >80% for all languages.
  Spanish/French (Latin script, Romance languages) may already show some partial matches
  due to shared vocabulary with English.

## Re-running

```bash
node ~/Synapse/synapse-src/saqt/benchmark-crosslingual.mjs
```

After deploying multilingual model, save new results as `crosslingual-multilingual.json`
and compare against this baseline.
