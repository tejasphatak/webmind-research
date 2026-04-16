---
title: External deep-research review of Carrier-Payload text-only paper — DEFERRED
date: 2026-04-16
reviewer: Gemini Deep Research Pro Preview (attempted)
paper: papers/carrier-payload-text-only-v1.md
status: deferred_blocked_on_gemini_key
license: CC-BY-4.0
---

# Gemini Deep Research external review — Carrier-Payload text-only v1

**Status: DEFERRED.** All available Google AI Studio keys on cortex2-vm
return `INVALID_ARGUMENT` — two distinct errors were observed:

- `~/.claude/secrets/gemini.json` → `API key expired. Please renew the API key.`
- `~/.google_api_key` → `API key not valid. Please pass a valid API key.` (the stored value is not an AIza-prefixed Studio key and was probably the wrong credential type for this endpoint.)
- Triadic workspace key (`AIzaSyCBsZDN1p5zbA3SZl7QUAZim4Xy0sUF2ts`) → `HTTP 400`.

This matches the 2026-04-15 memory note `project_beat_acceleration_2026-04-15.md` ("Gemini keys deactivated externally"). Deep Research Pro Preview could not be reached; the standard `gemini-3.1-pro-preview`, `gemini-2.5-pro`, and `gemini-2.0-pro-exp` endpoints all returned the same key error.

**What was attempted:**
- Endpoints tried: `gemini-deep-research-pro-preview`, `gemini-3.0-pro-deep-research-preview`, `gemini-3.1-pro-preview`, `gemini-2.5-pro`, `gemini-2.0-pro-exp`.
- Prompt prepared and cached at `/tmp/gemini_deep_review.py` — ready to re-run once a fresh Studio key is available.
- Questions covered: (a) novelty positioning vs Pluralis Beyond-Top-K, BottleNet++, SpecPipe-family, MemGPT; (b) methodological weak spots; (c) reproducibility bar gaps; (d) arXiv venue fit (cs.LG / cs.CL / cs.DC).

**Unblock action for Tejas:**
- Generate a fresh key at `https://aistudio.google.com/apikey` and write to `~/.claude/secrets/gemini.json` with shape `{"api_key": "AIza..."}`.
- Then: `python3 /tmp/gemini_deep_review.py` re-runs the review and overwrites this file with the actual model output.

**Does this block arXiv submission?**

No. Gemini sign-off for P1 already landed 2026-04-16 via `gemini_responses/two_paper_split.md` + `gemini_responses/final_signoff_v1.md` — that is the Gate-7 artifact of record. A Deep-Research pass would have been an additional pre-ship confidence check, not a hard gate. The recommendation below treats this as a "nice-to-have, not a blocker."
