---
title: Draft response to Tim Dettmers — thank-you + arXiv endorsement + quote permission
date: 2026-04-16
status: DRAFT — pending Tejas review and send
to: dettmers@cmu.edu
from: Tejas.Phatak@colorado.edu
subject: Re: Building on LLM.int8! — 22x activation compression for distributed inference
---

## Send-ready draft

Subject: Re: Building on LLM.int8! — 22x activation compression for distributed inference

Hi Tim,

Thank you — your note genuinely means a lot, especially the distinction you drew between training-time and inference-time activation compression. That line maps exactly to a scope decision we made in the draft after realizing our first attempt at a broader claim didn't hold.

A short update and two asks, if you have a moment:

**What changed since you read the draft.** The version at `papers/carrier-payload-v1.md` has been split. We kept the text-only, inference-time claim that matches your point — 22–24× compression at rank 16 across Gemma 3 1B / Llama 3.1 8B / Qwen 2.5 32B with KL < 0.1 and 100% top-1 agreement — and moved the cross-modality material to a follow-up paper after our multimodal data came back noisier than the text data. The current ship target is `papers/carrier-payload-text-only-v1.md` on the same branch. Long-context Qwen (seq 256–1621) shows the CR degrading 183× → 13×, which we now report explicitly rather than smoothing over. All claims are guarded by `tools/paper_invariants.py` — the invariants fail the build if any headline number drifts from the underlying JSON.

**Ask 1 — permission to quote.** Your statement that you're "not aware of a deep analysis of how activation compression can be tweaked to be best for distributed inference on regular consumer computers + internet connections" captures the exact gap we're trying to characterize in §1. Would you be comfortable with us quoting that sentence (attributed, timestamped, with your CMU affiliation) in the paper's introduction or acknowledgements? Any rewording is fine — I'll send the exact phrasing for your approval before it goes into the arXiv version.

**Ask 2 — arXiv endorsement.** If you're open to it, would you consider endorsing us for cs.LG on arXiv? That would unblock our submission path directly. The alternative is waiting on a CU Boulder faculty endorsement, and your read of the draft already puts you further into the material than a cold endorsement request typically allows. If you'd rather not, no pressure — the permission-to-quote alone is a big help.

Either way: thank you again. LLM.int8! and the outlier-aware quantization line of work were genuinely formative for how we thought about the sparse-residual payload, and the connection was not cosmetic.

Best,
Tejas Phatak
MS AI, University of Colorado Boulder
Webmind Research — webmind.sh

## Notes for Tejas (not part of the email)

- **Tone.** Warm but brief. Respects his time. One thank-you, two asks, clear escape hatch on both.
- **The quote ask is the load-bearing one.** Even if he declines the endorsement, the quote alone materially strengthens the paper's §1.
- **The endorsement ask is framed as "if you're open to it."** Leaves him a graceful decline path. Don't press if he deflects.
- **Do not oversell "endorsement."** He said "good work, keep it up!" — that's encouragement, not an endorsement commitment. The ask is for a new, explicit commitment.
- **Before sending:** verify `papers/carrier-payload-text-only-v1.md` is actually on `synapse-research/initial` (or the merged atlas gate-close branch) and readable from the link. As of 2026-04-16 Atlas has closed Gates 1 (invariants rescope, 40/40 pass), 3 (pre-reg exception committed at `papers/carrier-payload-preregistration-exception-v1.md`), and 6 (reproduce.sh supports `--quick` and `--full` modes + auto-verify). Only Gate 10 (arXiv endorsement) remains open — that is the specific ask in this email. He should not open the link and see stale state.
- **If he endorses:** Gate 10 closes. File the endorsement confirmation under `findings/` with the endorsement-code arXiv provides.
- **If he declines endorsement but grants quote:** still a net win. Cite quote in §1, proceed with CU Boulder endorsement path for Gate 10.
- **Mailmap:** commits touching this email chain should use the personal-name author identity per `feedback_coauthor_email_squat.md` and MANIFESTO §IV.3 — never `noreply@anthropic.com`.
