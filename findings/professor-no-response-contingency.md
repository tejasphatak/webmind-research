# Contingency: Professors Don't Respond — Alternative Paths to Publication

**Date:** 2026-04-16
**Context:** Tejas emailed CU Boulder professors and Tim Dettmers on 2026-04-15. Two risks if they don't respond:
1. No expert feedback on the paper draft (nice-to-have)
2. No arXiv endorsement for cs.LG (BLOCKER for first-time submitters)

---

## The arXiv-endorsement blocker

**The hard rule:** First-time submitters to cs.LG need an endorsement from an existing arXiv author in that category.

### Path A — Alternative cs.LG endorsers (cold-outreach-friendly)

Reach out to these people with a short, specific ask. Include the paper link. Expect a 20–40% response rate on well-targeted cold emails.

- **Tim Dettmers** (UW, LLM.int8! author) — already emailed 2026-04-15. If no response in 10 days, send polite follow-up referencing the arXiv endorsement ask specifically.
- **Tri Dao** (Princeton, FlashAttention) — responds to student cold emails.
- **Sasha Rush** (Cornell Tech) — publicly supportive of student research; responds to arXiv endorsement requests.
- **Matei Zaharia** (UC Berkeley, Databricks/Apache Spark) — actively endorses systems-ML first-time authors.
- **Tianqi Chen** (CMU, XGBoost/TVM/MLC LLM) — runs MLC LLM which is adjacent to Synapse; may respond.
- **Christopher De Sa** (Cornell) — systems-ML, often endorses.
- **Zhihao Jia** (CMU) — distributed/parallel ML, directly adjacent.

**Ask template (short):**

> Subject: arXiv cs.LG endorsement request — first-time submission on activation compression for decentralized inference
>
> Dear Prof. [Name],
> I'm a CU Boulder MS AI student working on activation compression for decentralized LLM inference (Synapse / webmind.sh). I have a finished v1 draft showing 13–26× compression of inter-shard activations on Qwen 2.5 32B at long context, and I'd like to submit it to arXiv cs.LG. As a first-time arXiv author I need an endorsement.
>
> Paper draft: [URL to carrier-payload-text-only-v1.md]
> Reproduction: [URL to reproduce.sh]
>
> If the paper looks non-crackpot to you, would you be willing to endorse me on arXiv? I'm happy to answer any questions first. arXiv endorsement code available via https://arxiv.org/auth/endorse
>
> Thank you,
> Tejas Phatak

### Path B — Post on OpenReview instead

OpenReview (openreview.net) has no endorsement gate. Submit to:
- **OpenReview Preprints** (general archive)
- **ICLR 2027 Open Submission** (deadline typically late Sept, rolling)
- **NeurIPS 2026 Main Track** (deadline usually May)

OpenReview submissions:
- Get a public timestamp and permanent URL
- Are indexed by Google Scholar
- Can be cited in arXiv endorsement applications later as "here's my published work"
- No gatekeeping for first-time authors

**Pragmatic play:** submit to OpenReview first (no barrier), get a public URL, then use that to apply for arXiv endorsement or to pitch to professors for a warmer endorsement ask ("I already have a publicly posted paper; I'd like to mirror on arXiv").

### Path C — Zenodo for DOI-backed citation

Zenodo (zenodo.org, operated by CERN):
- Free, open, permanent DOI
- No gatekeeping
- Citations work in academic literature
- Fast: minutes from upload to DOI assigned

Zenodo is a legitimate citable archive. Not "arXiv" but provides the same priority-establishing function. Good for establishing priority before figuring out arXiv endorsement.

### Path D — Submit to a workshop directly

Workshops at NeurIPS, ICLR, ICML, MLSys often:
- Don't require arXiv pre-print
- Have their own proceedings with DOI
- Accept first-time authors
- Lower bar than main conference

Candidate workshops for P1:
- **NeurIPS 2026 ENLSP (Efficient Natural Language and Speech Processing)** — deadline Aug–Sep 2026
- **NeurIPS 2026 Workshop on Sys4ML / Distributed ML** — if held
- **ICLR 2027 MCDC (Modularity, Composition, Decentralization, Compression)** — where Pluralis's related paper was published
- **ACL 2026 Efficient NLP Workshop**

### Path E — Ask a coauthor to be a formal coauthor

Harder ask, but sometimes works: a CU Boulder professor or Tim Dettmers may be willing to be a formal coauthor if the paper is truly good and they contribute meaningfully (via review + small amount of writing). This converts endorsement-gating into coauthorship.

**Risk:** dilutes Tejas's sole authorship, which is load-bearing for EB1-A. Probably avoid unless all other paths fail.

### Path F — arXiv endorsement via submission volume

arXiv allows endorsement requests through their system after a submission is prepared. You upload the paper, then request an endorsement, and arXiv emails potential endorsers automatically. Sometimes an author you've never contacted will endorse a prepared submission simply because it looks legitimate.

---

## Decision tree

```
Day 0 (already done): Professor emails sent
Day 7: No response → send polite follow-up
Day 14: Still no response → START PATH B (OpenReview) + PATH D (workshop submission)
Day 14: Also → START PATH A (additional cold-outreach endorsers)
Day 21: Still no response → Path C (Zenodo DOI) as worst-case priority claim
Day 30: If no endorsement at all → Path F (arXiv system auto-endorsement request)
```

## What Tejas should do NOW

1. **Do not wait 30 days**. In parallel with professor emails, immediately:
   - Create an OpenReview account
   - Create a Zenodo account
   - List 2-3 cs.LG authors from Path A who look approachable
2. **Set a calendar reminder for Day 7** (2026-04-22) to send professor follow-up
3. **If paper is ready before endorsement lands:** post on Zenodo for DOI, post on GitHub (already done), and continue seeking arXiv endorsement in parallel

---

## Anti-pattern to avoid

**DON'T** delay publication indefinitely waiting for a specific endorsement. The work's priority is established by first public posting (Zenodo, OpenReview, or arXiv — any works). Delaying 3 months for arXiv endorsement while sitting on a finished paper risks scoop-by-others-before-you-publish.

**DO** publish somewhere within 2-3 weeks of the paper being finalized. arXiv is nice but not essential if you have a DOI-backed alternative.

---

## For Paper 2 and future work

Once Paper 1 is published somewhere (arXiv, Zenodo, OpenReview, workshop proceedings), future submissions are much easier:
- Tejas is no longer a first-time arXiv submitter
- Can cite Paper 1 in endorsement applications
- Existing paper serves as proof-of-competence for reviewers

**This is why Paper 1 shipping matters most — it unlocks everything downstream.**
