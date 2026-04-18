# Instructions for ALL AI Agents Working on This Repo

> This file is the operating manual. Read it first. Follow it always.

## Automatic Behaviors (do these without being asked)

### After every session:
1. **Update mindmap/README.md** — if you discovered new connections, added papers, or changed architecture
2. **Update CLAUDE.md or GEMINI.md** — if the context changed (new results, new priorities, new decisions)
3. **Run invariant checks** before committing (see below)
4. **Clean up** — no hardcoded values, no personal info, no secrets paths, no employer names

### Before every commit:
1. **No personal info:** grep for emails, phone numbers, employer names, visa references. If found, stop.
2. **No secrets:** grep for API keys, tokens, file paths containing "secrets" or "credentials". If found, stop.
3. **No hardcoded counts:** don't write "305K" or "306K" — the KB grows. Use "growing" or pull from live stats.
4. **Honest claims:** every number must trace to a benchmark file in `benchmarks/`. No made-up stats.
5. **Citation check:** every referenced paper must be real and verifiable via web search.

### Invariant checks (run before push):
```bash
# Must all pass before pushing
echo "=== INVARIANT CHECKS ==="

# No personal emails
! grep -r "tejasphatak@gmail\|Tejas.Phatak@\|dettmers@" --include="*.md" --include="*.py" --include="*.js" . && echo "✓ No personal emails" || echo "✗ PERSONAL EMAIL FOUND"

# No employer names  
! grep -ri "mastercard\|H1B\|visa sponsor" --include="*.md" --include="*.py" . && echo "✓ No employer refs" || echo "✗ EMPLOYER REFERENCE FOUND"

# No secrets paths
! grep -r "\.claude/secrets\|api_key.*=\|token.*=" --include="*.md" --include="*.py" . && echo "✓ No secrets" || echo "✗ SECRETS PATH FOUND"

# No hardcoded KB counts (except in benchmark results which are snapshots)
! grep -rn "305,*[0-9]*K\|306,*[0-9]*K" README.md CLAUDE.md GEMINI.md papers/*.md 2>/dev/null && echo "✓ No hardcoded counts" || echo "✗ HARDCODED COUNT FOUND"
```

## What belongs in this repo (PUBLIC)

| Type | Examples | Belongs here? |
|------|---------|--------------|
| Research papers | self-evolving-retrieval.md | ✅ Yes |
| Benchmark results | benchmarks/*.json | ✅ Yes |
| Research code/tools | tools/*.py | ✅ Yes |
| Training data | data/*.jsonl | ✅ Yes |
| Model files | trained_model/* | ✅ Yes |
| Inventions | inventions/*.md | ✅ Yes |
| Experiment logs | findings/*.json, *.log | ✅ Yes |
| Mindmap | mindmap/README.md | ✅ Yes |

## What does NOT belong (PRIVATE)

| Type | Why | Where it goes |
|------|-----|--------------|
| Outreach emails | Strategy, not research | ~/webmind-private/outreach/ |
| Agent coordination | Internal ops | Not in any repo |
| Infrastructure details | Security | Not in any repo |
| Draft comms to people | Privacy | ~/webmind-private/ |
| Visa/immigration docs | Personal legal | ~/webmind-private/legal/ |
| Secrets/API keys | Security | ~/.claude/secrets/ |

## Architecture (the mental model)

```
UNDERSTAND (encoder, 22M params)
    → just turns text into meaning vectors
    → does NOT generate text
    → can be swapped for any sentence transformer

THINK (convergence loop)
    → search → check answer → search again → converge
    → fixed-point iteration in embedding space
    → bi-embedding re-ranking (Q-sim + A-sim)

REMEMBER (growing database)  
    → Q&A pairs with embeddings + weights
    → learns from web search when KB can't answer
    → feedback adjusts weights (boost/penalize)
    → delta sync via watermarks (like git pull)

DISTRIBUTE (Synapse mesh — future)
    → shard KB across devices
    → broadcast queries via WebRTC
    → carrier-payload compression for transfer

PROTECT (ethics through data)
    → high-weight ethics pairs teach boundaries
    → PII sanitization on all learned content
    → source agreement validation for web results
```

## Current priorities (update this when priorities change)

1. Add baselines (BM25, DPR) to benchmark comparison
2. Fix ethics gate (48% adversarial pass rate → need >90%)
3. Deploy multilingual encoder (ONNX for reproducibility)
4. Start 30-day self-evolution experiment
5. Browser optimization (Voy/USearch ANN search)

## How to run benchmarks

```bash
cd ~/Synapse

# Structural tests (53 tests, no API key needed)
node synapse-src/saqt/test-deploy.mjs

# QA benchmarks (NQ, TriviaQA, HotPotQA — includes RLHF learn-back)
node synapse-src/saqt/benchmark.mjs --samples 50 --concurrency 1

# LLM-judged quality tests (uses claude --print)
node synapse-src/saqt/test-quality.mjs
```

## How to update the mindmap

When you discover a new connection, add a result, or change the architecture:

1. Read `mindmap/README.md`
2. Add your finding to the appropriate section
3. Update the mermaid diagrams if the architecture changed
4. Move proven things from "Speculative" to "Proven"
5. Add new speculative connections if you see them
6. Commit with a clear message

## Naming conventions

- Papers: `papers/[topic]-YYYY-MM-DD.md`
- Findings: `findings/YYYY-MM-DD-[description].md`
- Inventions: `inventions/YYYYMMDDTHHMMSSZ-[name].md`
- Benchmarks: auto-generated with timestamp
- Tools: `tools/[name]_v[N].py`

## The one rule

**The database is the model. Everything else is plumbing.**

If you're about to add code logic that decides behavior — stop. Can it be a KB pair instead? Can the data teach the behavior? If yes, add data, not code.
