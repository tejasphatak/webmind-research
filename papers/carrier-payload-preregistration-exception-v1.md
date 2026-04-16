# Carrier-Payload Paper 1 — Gate-3 Pre-Registration Exception

**Paper under this exception:** `papers/carrier-payload-text-only-v1.md`
**Target venues:** COLM, ACL, or EMNLP (arXiv preprint first)
**Exception date:** 2026-04-16
**Exception author:** Tejas Phatak

---

## Why this paper does not have a pre-registration file

`SUBMISSION_GATING.md` Gate 3 requires a pre-registration file committed
before data collection for empirical claim-testing papers. It also says:

> Exceptions: methodology-only papers, theoretical papers, survey papers.
> Document the exception.

This paper qualifies for the exception for a specific reason, and the
reason is *not* "the hypothesis happened to hold so we skipped the discipline."
It is: **this paper is not a hypothesis-test paper**.

The paper:

1. **Proposes a method** (Carrier-Payload: PCA carrier + sparse residual for
   activation transport on the inter-shard wire) — this is the methodology
   contribution.
2. **Characterizes the method's empirical behavior** across three text-only
   LLM families at short context and one family (Qwen 2.5 32B) across four
   long-context sequence lengths — this is a *measurement / characterization*
   study, not a hypothesis test with a falsifiable prediction committed in
   advance.
3. **Reports the method's limitations honestly**, including the short-context
   rank-bound artifact and the regime transition where the compression law
   breaks — this is characterization, not confirmation of a prior hypothesis.

No statement of the form "we predict effect X; if we see effect Y instead,
we will conclude the hypothesis is false" was committed prior to data
collection. That framing does not fit this paper — the paper's contribution
is the method and its measured behavior, not a falsifiable claim being tested.

## What discipline applies instead

Dropping Gate 3 does not mean dropping empirical rigor. The following
mechanisms substitute for pre-registration and are all already in place for
this paper at ship time:

### 1. Raw-data invariants for every numeric claim (Gate 1)

`tools/paper_invariants.py` programmatically validates every numeric claim
in the paper against the raw JSON files committed under `findings/`.
40/40 invariants pass as of 2026-04-16 on branch `synapse-research/initial`.

If a claim in the paper body drifts from what the raw data actually shows,
the invariant fails and the paper cannot be shipped. This is strictly
stronger than pre-registration for hypothesis-test-style claims, because
pre-registration documents *what would have falsified the hypothesis* in
words, whereas invariants *mechanically refute* any headline number that
doesn't survive the raw data.

### 2. Gate-13 tiered claim rigor for every external or projected number

`SUBMISSION_GATING.md` Gate 13 requires every non-local numeric claim
(co-author numbers, prior-work numbers, deployment projections) to meet one
of three bars:

- **(a)** Raw-data invariant in this repo, OR
- **(b)** Reproducible-methodology citation with runnable code, OR
- **(c)** Explicit projection label in the paper text with scope qualifier
  and a corresponding Limitations entry.

This paper applies (a) for all its headline empirical claims; (c) only for
forward-looking deployment commentary in §5 where the label is explicit.

### 3. Null-results publishing for the claims that did *not* survive

The `findings/` directory preserves the negative results that were
produced during the research leading up to this paper. Two examples that
were initially attractive hypotheses and were later falsified on the same
data:

- **Benford's-law Byzantine detection on LLM activations** —
  `findings/benford_result.md` reports this as empirically falsified on
  synthetic activation data (activations are Gaussian-bulk, only 2 orders
  of magnitude dynamic range; Benford requires 5+). It was retired from
  paper-candidate status before this paper was drafted.
- **"Universal effective rank ≈ 16" for transformer activations** —
  the long-context Qwen run at seq ∈ {256, 512, 1024, 1621} empirically
  falsified the universality claim. Rank for 99% variance grows to 384
  at seq 1621. The paper's §3.4 reports the actual scaling, and §1.2
  Non-contributions explicitly disclaims the falsified universal claim.

Both falsifications are in the commit history and in findings/ — the
null-results-publish discipline is preserved even though the overall paper
is a characterization, not a hypothesis test.

### 4. External cross-verification (Gate 7)

An independent AI research agent reviewed the draft at multiple milestones
with transcripts committed under `gemini_responses/`, and the final
sign-off-for-scope artifact is `gemini_responses/final_signoff_v1.md`
plus `gemini_responses/two_paper_split.md` for the text-only rescope.
An external domain expert (Dettmers, CMU) read a prior draft and provided
a timestamped statement that no deep prior art on activation compression
for distributed consumer inference is known to him
(`findings/2026-04-16-tim-dettmers-external-validation.md`).

## Future work: when this exception will *not* apply again

If a follow-up paper in this research program makes a **falsifiable
prediction** — for example, "at seq 4096, mean rank99 will be between X
and Y under model family M" — that paper must pre-register. The exception
here covers characterization/measurement of an existing method's behavior,
not prediction of future measurements.

Papers 2 (multimodal extension) and 3 (distributed speculative decoding)
in the roadmap will receive pre-registration files before any new data
collection begins, per the `ROADMAP.md` commitment.

## Decision

Gate 3 is **documented-exception PASS** for
`papers/carrier-payload-text-only-v1.md` on the grounds above. The paper
ships subject to all other gates — invariant suite, citations, limitations,
disclosure, reproducibility, cross-verification, scope, coherence, LaTeX,
links, external-claim tiers, and career-fit — with no silent waiver of
discipline.

*Committed to the repository in the same series as the paper itself,
license CC-BY-4.0 matching the paper.*
