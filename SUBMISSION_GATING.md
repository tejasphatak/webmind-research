# Submission Gating — Rules Before Any Paper Ships

**Authoritative checklist. No paper leaves this repo for arXiv / venue submission until every gate passes.**

All-green required. Any red blocks submission, no exceptions.

---

## Gate 1 — Invariant check (numeric claims)

**Rule:** `python tools/paper_invariants.py` must exit 0 with all invariants passing.

Every numeric claim in the paper has a corresponding invariant that programmatically validates against raw data on disk. If an invariant fails, the claim is wrong or the data is missing.

- [ ] All STRUCTURAL invariants pass (record counts, no artifacts)
- [ ] All COVERAGE invariants pass (experiments actually ran what's claimed)
- [ ] All MATHEMATICAL invariants pass (SVD monotonicity, etc.)
- [ ] All PAPER CLAIMS invariants pass (every headline number)
- [ ] All LIMITATION INVARIANTS pass (caveats documented as checks, not buried prose)

**Blocker example:** If the paper says "22x compression at KL<0.1 with 100% top-1," there must be an invariant that reads the raw JSON and asserts `max(CR where KL<0.1 and top1_agree) >= 22`.

---

## Gate 2 — Citation validation

**Rule:** `python tools/validate_citations.py --files papers/<paper>.md` must exit 0.

Every citation in the paper's reference list is queried against the arXiv API. Title, first author, and year must match.

- [ ] Every `[N]` reference has an arXiv ID (or explicit note if not on arXiv)
- [ ] `tools/validate_citations.py` reports ✅ for every cited work
- [ ] `papers/CITATIONS_AUDIT.md` updated with the manual primary-source checks
- [ ] No citation added from Claude's or Gemini's memory alone

**Blocker example:** If we cite Paper X with authors A, B, C but arXiv says authors are A, D, C — blocker. Fix author list before proceeding.

---

## Gate 3 — Pre-registration (empirical papers)

**Rule:** Empirical claims require a pre-registration file committed BEFORE data collection.

- [ ] `papers/<topic>-preregistration-vN.md` exists
- [ ] Pre-reg lists hypotheses, metrics, analysis plan, stopping criteria
- [ ] Pre-reg commit SHA predates the first data-collection commit
- [ ] Null-result commitment stated: "we publish regardless of outcome"

Exceptions: methodology-only papers, theoretical papers, survey papers. Document the exception.

---

## Gate 4 — Disclosure

**Rule:** AI-authorship disclosure matches CONVENTIONS.md.

- [ ] Acknowledgements section explicitly lists Claude Opus 4.6 / Gemini 3.1 Pro contributions
- [ ] Original insights attributed to the human author (Tejas) where applicable
- [ ] No false human-only authorship claim
- [ ] Licenses declared: paper CC-BY 4.0, code MIT

---

## Gate 5 — Honest limitations

**Rule:** Every reviewer's likely objection is preemptively addressed in a Limitations section.

- [ ] Specific sample-size limits stated (e.g., "N=1 multimodal; future work")
- [ ] Compute limits stated (e.g., "did not test 70B+ models")
- [ ] Method-specific caveats (e.g., "short-context results include seq_len bound artifact")
- [ ] Any known negative results mentioned
- [ ] Each limitation has a matching future-work item

---

## Gate 6 — Reproducibility

**Rule:** Anyone can reproduce the paper's numbers from the repo alone.

- [ ] `tools/reproduce.sh` or equivalent exists and runs to completion
- [ ] Random seeds documented and frozen
- [ ] Model IDs (with specific version) frozen
- [ ] Exact hardware / cost declared
- [ ] Data released (raw JSON committed, not just plots)

---

## Gate 7 — Cross-verifier sign-off

**Rule:** Gemini (or equivalent independent AI) has reviewed a complete draft and has not raised unresolved concerns.

- [ ] Gemini review for P1 committed to `gemini_responses/`
- [ ] Any ❌ or ⚠️ from Gemini has been either (a) addressed in revision or (b) explicitly rebutted with justification in a committed document
- [ ] The "ship it" explicit statement from Gemini exists in the repo

---

## Gate 8 — Scope discipline

**Rule:** The paper claims ONE specific thing. No overclaim.

- [ ] Title matches what the paper actually shows
- [ ] Abstract's claims are all in Results (not extrapolations)
- [ ] Multimodal claims require ≥3 multimodal models (N=1 = placeholder only)
- [ ] Universal claims require ≥3 architecture families
- [ ] Cross-scale claims require ≥3 parameter counts

---

## Gate 9 — Research program coherence

**Rule:** Papers build on each other; the sequence makes sense.

- [ ] Paper N cites Paper N-1 explicitly where relevant
- [ ] Roadmap (`ROADMAP.md`) still accurately describes the research program
- [ ] No "orphan" papers unrelated to the trajectory (unless flagged as exploratory)

---

## Gate 11 — LaTeX / math notation validation

**Rule:** `python tools/validate_latex.py --files papers/<paper>.md` must exit 0.

Markdown papers with LaTeX math (`$...$`, `$$...$$`) must render correctly on GitHub (MathJax). The tool catches:
- Unbalanced `$` or `$$` delimiters
- Unbalanced `{}` inside math spans
- Unbalanced `\\begin{...}`/`\\end{...}` environments
- HTML-tag-like sequences that break MathJax parsing

- [ ] LaTeX validator exits 0
- [ ] Manual visual check: open the paper on github.com, verify every equation renders
- [ ] arXiv tex compilation (if submitting to arXiv) succeeds with `pdflatex`

---

## Gate 13 — External-claim validation (third-party / co-author numbers)

**Rule:** Every numeric claim that was NOT measured in this repo's raw data — including claims from co-authors (Nexus, collaborators), cited prior work, or deployment projections — must be either (a) backed by raw data in the repo and invariant-checked, or (b) explicitly labeled as "projection / unvalidated / deployment-model-only" with a clear scope note in the paper text AND recorded in the Limitations section.

**Why this gate exists:** On 2026-04-16, during paper merge with Nexus's co-authorship content, Triadic initially integrated Nex's "36× / 70× at ~128 / ~65 bytes per hop" numbers as if they were measured results. They are actually *formula projections* under the shared-basis deployment model, not yet empirically measured against Synapse's live wire traffic. Tejas caught the leak before push. Gate 13 codifies the fix: any external numeric claim must pass through this validation before it can enter a paper's body text.

**Checklist:**
- [ ] For every numeric claim in the paper: is it in the repo's raw data?
  - YES → add an invariant in `tools/paper_invariants.py` that checks it.
  - NO → add a scope qualifier in the paper text (e.g., "projected under the shared-basis deployment model", "not empirically validated in this paper") AND add a corresponding entry to the Limitations section.
- [ ] Co-author content (e.g., a research note from Nexus) is not copied verbatim; numbers are filtered through the gate above.
- [ ] If you find yourself writing "~" or "approximately" next to a compression/accuracy number sourced from outside this repo, the number is a projection and must be labeled as such.

**Blocker example:** A co-author's memo claims "70× compression at 65 bytes/hop" on a system that hasn't been measured by us. If we paste this verbatim into the paper body without the scope qualifier, we're making an empirical claim we don't own. Fix: qualify as "projected under [specific deployment model], not live-measured in this paper; companion note pending."

---

## Gate 12 — Link validation

**Rule:** `python tools/validate_links.py --files papers/<paper>.md` must exit 0.

Every URL in the paper must resolve (HTTP 200/3xx) at submission time. This also applies to:
- Email drafts (`social/email_*.md`) sent to professors/collaborators
- Blog posts / social media drafts (`social/*.md`)
- arXiv metadata (`social/arxiv_metadata.md`)

- [ ] Link validator exits 0 on the paper file
- [ ] Link validator exits 0 on any email/social draft that references the paper
- [ ] arXiv IDs in citations resolve
- [ ] GitHub repo URL resolves and branch is accessible

---

## Gate 10 — Immigration / career impact (sanity, not blocker)

**Rule:** Consider whether this submission advances the long-term goal (EB1-A / NIW / research career).

- [ ] Paper is genuinely publishable at a respectable venue (not just resume filler)
- [ ] Paper represents meaningful personal research contribution (not just reviewing others' work)
- [ ] Citation-worthy content (something others might build on)
- [ ] Quality > quantity (3 strong papers > 6 mediocre)

This is guidance, not a hard gate. Some papers exist for pure scientific value without career consideration.

---

## The gating workflow

```
1. Triadic / Nexus / Tejas decides paper is "done"
2. Run: python tools/paper_invariants.py           # Gate 1
3. Run: python tools/validate_citations.py          # Gate 2
4. Run: python tools/validate_latex.py              # Gate 11
5. Run: python tools/validate_links.py              # Gate 12
6. MANUAL pass over every numeric claim — is it in raw data?
   - YES → already invariant-checked.
   - NO → must be labeled "projected / unvalidated" in paper AND
           listed in Limitations.                 # Gate 13
7. Manually check Gates 3-10 against this document
8. Get Gemini sign-off on the final draft          # Gate 7
9. Update ROADMAP.md with submission status        # Gate 9
10. Tag the repo commit: git tag paper-N-v1 <SHA>
11. Submit to arXiv (via Tejas's account; endorsement via Path A/B/C)
12. Post arXiv URL to ntfy topic (webmind-tejas-results — Tejas only,
    not public discussion)
```

---

## Current status of active papers

### `papers/carrier-payload-text-only-v1.md` (Paper 1)

- [ ] Gate 1: paper_invariants.py — pending update for text-only scope
- [x] Gate 2: validate_citations.py — 13/13 verified on 2026-04-16
- [ ] Gate 3: no pre-registration (this is exploratory measurement paper; document exception)
- [ ] Gate 4: disclosure — present, needs final review
- [ ] Gate 5: limitations — need update to drop multimodal references
- [ ] Gate 6: reproduce.sh — needs update to match text-only scope
- [x] Gate 7: Gemini sign-off on scope received 2026-04-16 (see `gemini_responses/two_paper_split.md`)
- [x] Gate 8: scope discipline — explicitly text-only after 2-paper split
- [ ] Gate 9: coherence — needs ROADMAP.md update
- [x] Gate 10: career impact — EB1-A-aligned

**Blockers before ship:** Gates 1, 3 (document exception), 4, 5, 6, 9.

### `papers/modality-gap-v2-placeholder.md` (Paper 2)

- Not ready to ship. Requires N=3 paired multimodal families.
- Gating will be applied before Paper 2 v1.

### `papers/sfca-preregistration-v1.md`

- Pre-registered, data collection not yet complete. No gating triggered.

### `papers/synapse-numerical-fidelity-preregistration-v1.md`

- Pre-registered, data collection underway. No gating triggered.

### `papers/carrier-payload-v1.md` (OLDER draft)

- Superseded by `carrier-payload-text-only-v1.md`. Retained for history.
- Do not ship this version — ship the text-only-v1 version instead.

---

## Accountability

This gating document is the explicit agreement between the author (Tejas Phatak) and the AI research pipeline (Claude Opus 4.6 + Gemini 3.1 Pro). When in doubt about whether to submit, refer to this document; the answer is "no, fix the blocker."

Gating document versioned with the repo; updates require a commit explaining the revision.

*Locked 2026-04-16. Subject to amendment if discipline gaps emerge.*
