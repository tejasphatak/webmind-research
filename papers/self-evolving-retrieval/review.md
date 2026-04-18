# Review: Self-Evolving Retrieval: A Third Architecture for AI Beyond Generation and Search

**Reviewer stance:** COLM/EMNLP reviewer. Harsh but fair.

---

## Overall Score: 4/10 (Borderline Reject)

The paper presents an interesting systems idea — a retrieval engine that grows its knowledge base through use and runs entirely in-browser — but the framing as "a new architecture class" is overclaimed, the evaluation is too small to support the central thesis, and several key related works are missing. The honest limitations section is appreciated and unusual, but the paper needs substantial revision before venue submission.

---

## Strengths

1. **Radical honesty in limitations.** Section 6 is unusually candid. Calling your own in-distribution result "circular by design," reporting a 48% ethics-gate pass rate, and disclaiming "single developer, single session" builds real credibility. Most papers hide this. Keep it.

2. **The human-thinking analogy in Section 1 is effective.** The 2x2=4 / multiplication algorithm / calculator decomposition is intuitive and clearly motivates why separating stored facts, reasoning, and tool use matters. It grounds an abstract architectural argument in something concrete.

3. **Browser deployment is genuinely impressive.** 214MB for encoder + index + 306K-entry KB running offline in a tab is a real engineering contribution. The operational metrics (12ms encoding, <5ms ANN search) are concrete and verifiable.

4. **The HotPotQA generalization result (0% to 72% on held-out) is the strongest finding.** Multi-hop compositional generalization from learning component facts is a meaningful signal. This deserves to be the centerpiece of the paper, not buried in 3.2.

5. **Convergence loop as fixed-point iteration.** Iterating retrieval until embedding delta stabilizes is a clean idea with clear formalization potential. The connection to Banach fixed-point theorem is implicit but present.

---

## Weaknesses

### Major

1. **"Third architecture class" is overclaimed.** The system is a knowledge base with semantic search and a web-search fallback. KB-backed QA systems have existed since the 1970s (LUNAR, SHRDLU). The novelty is in the specific combination (embeddings + self-evolution + browser), not in the architecture class. Framing it as "neither generative nor search" creates a false trichotomy — RAG systems already occupy this middle ground. Downgrade to "a novel combination" or defend the taxonomy more rigorously.

2. **Evaluation is underpowered.** 150 questions per condition, single run, no confidence intervals, no statistical significance tests. At n=50 per dataset, a single question flip changes EM by 2 percentage points. The 72% HotPotQA held-out result could be anywhere from ~58% to ~84% at 95% CI (binomial). This is acknowledged in limitations but not resolved.

3. **No comparison to any baseline system.** The paper compares to zero existing systems. Where does DPR + FAISS land on these same 150 questions? What about BM25? What about a naive TF-IDF lookup over the same KB? Without baselines, the numbers are uninterpretable. The ~8% NQ baseline without augmentation is mentioned in passing but not tabled.

4. **Fixed-point convergence is asserted but not analyzed.** How often does the loop converge? In how many iterations? Does it ever diverge or oscillate? What is epsilon? Is convergence guaranteed for any KB, or only empirically observed? The Banach contraction mapping theorem requires the retrieval function to be a contraction — is it? This is the most novel claim and it gets one paragraph with zero empirical analysis.

5. **The "no hallucination by construction" claim is misleading.** The system can return wrong answers from the KB — answers that were validated by source agreement but are factually incorrect. It can also return stale answers. "No hallucination" is technically true (it doesn't generate) but functionally misleading. A wrong retrieved answer and a hallucinated answer are equivalent from the user's perspective.

### Minor

6. **Section 3.1 starts at 0.0% but the baseline is ~8%.** If the KB already contains 306K entries, why does Run 1 score 0%? This implies the benchmark questions were specifically chosen to not overlap with the existing KB, which is fine, but should be stated explicitly.

7. **The comparison table in Section 1 is unfair.** "Runs offline on phone: No" for LLMs is false as of 2025 (Gemma 2B, Phi-3-mini, llama.cpp on mobile). "Minimum viable hardware: GPU cluster" for LLMs is false. The table should compare honestly or be removed.

8. **Cross-lingual results (Section 3.4) are padding.** Embedding similarity scores between language pairs are not retrieval accuracy. This section reports cosine similarity, not end-to-end QA performance. Either run cross-lingual QA or remove the section.

9. **"Self-evolution" terminology is loaded.** The system learns by having answers injected into its KB. This is closer to "incremental knowledge base population" than "evolution," which implies adaptation of the system's own mechanisms. The system's retrieval logic never changes — only its data does.

10. **Source agreement validation (2+ sources) is not evaluated.** How often do sources agree? What is the precision of the agreement signal? Could a single high-quality source (Wikipedia) outperform the agreement heuristic?

---

## Missing References

- **KILT** (Petroni et al., 2021, NAACL): Knowledge-Intensive Language Tasks benchmark. The standard evaluation framework for exactly this type of system. The paper should evaluate on KILT.
- **ELI5** (Fan et al., 2019): Long-form QA. Relevant as a contrast — this system cannot do long-form answers.
- **REALM** (Guu et al., 2020, ICML): Retrieval-augmented language model pre-training. The original retrieval-augmented architecture.
- **FiD** (Izacard & Grave, 2021): Fusion-in-Decoder. Relevant baseline for open-domain QA.
- **Poly-encoders** (Humeau et al., 2020, ICLR): The direct precedent for using the same encoder architecture for both query and candidate scoring. The "answer alignment with shared encoder" approach is essentially a poly-encoder without the attention layer.
- **Never-Ending Language Learner (NELL)** (Mitchell et al., 2018): A system that continuously learns facts from the web and grows a knowledge base. The most direct precedent for "self-evolving" KB-based QA.
- **QA-pair mining literature** (e.g., Lewis et al., 2021, "PAQ: 65 Million Probably-Asked Questions"): Direct precedent for large-scale QA pair knowledge bases.
- **Banach fixed-point theorem** — if claiming fixed-point iteration, cite the mathematical foundation and discuss contraction conditions.

---

## Questions for Authors

1. What is the retrieval accuracy on the 306K existing KB entries? If the system already has 306K Q&A pairs, what is the end-to-end EM on a random sample of those?
2. How does the convergence loop interact with KB size? Does convergence behavior change as the KB grows from 1K to 306K entries?
3. The web search fallback is the actual source of new knowledge. What happens if you remove the convergence loop and the re-ranking and just do naive embedding search + web fallback? How much do the "novel" components actually contribute?
4. What is the false positive rate of retrieval? When the system returns an answer confidently, how often is it wrong?

---

## Recommendation

**Reject in current form, encourage resubmission.** The core idea (browser-deployable self-growing KB with semantic retrieval) is solid engineering with real practical value. But the paper overclaims novelty, lacks baselines, has underpowered evaluation, and frames incremental KB population as architectural innovation. To reach acceptance:

1. Add baselines (DPR, BM25, BM25+rerank, at minimum)
2. Scale evaluation to 1000+ questions with confidence intervals
3. Analyze the convergence loop empirically (iterations to convergence, failure modes, ablation)
4. Downgrade framing from "new architecture class" to "novel system combining X, Y, Z"
5. Engage with NELL, PAQ, KILT, and poly-encoder literature
6. Ablate: what does each component (re-ranking, convergence loop, source agreement) contribute independently?

The HotPotQA compositional generalization finding is genuinely interesting and could anchor a strong resubmission if properly evaluated.

---

**Reviewer confidence:** 4/5 (familiar with retrieval, QA benchmarks, and browser ML deployment)
