# Email to CU Boulder Professor

**To:** [Professor's name — ideally someone teaching ML Systems, Distributed Systems, or Deep Learning]
**From:** Your CU Boulder email
**Subject:** Seeking feedback on activation compression research — potential MLSys submission

---

Dear Professor [Name],

I'm a current MS AI student at CU Boulder (online program). I've been working on an independent research project on activation compression for decentralized LLM inference, and I have results I'm excited about.

**The short version:** When you shard an LLM across devices for distributed inference, the activations transmitted between shards are ~97% redundant. Using PCA decomposition + sparse residual correction, we achieve 22x compression at shard boundaries with 100% top-1 token agreement (KL divergence 0.023) on Gemma 3 1B IT.

This has direct practical application: it makes distributed inference viable on consumer internet connections, enabling volunteer-device inference networks (similar to Petals or BOINC for AI).

I'm preparing to submit this to MLSys or a NeurIPS workshop, and I would greatly appreciate:
1. Your feedback on the draft (attached / linked below)
2. Guidance on the right venue
3. If you think the work merits it, an arXiv endorsement for cs.LG (I'm a first-time submitter)

**Paper draft:** https://github.com/tejasphatak/webmind-research/blob/synapse-research/initial/papers/carrier-payload-v1.md

**Reproduction:** The experiment runs in 2 minutes on any GPU: https://github.com/tejasphatak/webmind-research/blob/synapse-research/initial/tools/reproduce.sh

I'm also running expanded experiments across 4 model families (Gemma 3, Gemma 4, Llama 3.1, Qwen 2.5) to demonstrate the universality of the low-rank activation structure.

I understand you're busy and appreciate any time you can spare. Even a quick "this looks promising / this has a fatal flaw" would be valuable.

Thank you,
Tejas Phatak
MS AI, University of Colorado Boulder
https://webmind.sh

---

**Notes for Tejas:**
- Send from your @colorado.edu email if you have one
- Best professors to target: whoever teaches CS 5/7XXX Machine Learning, Distributed Systems, or NLP
- Check CU Boulder CS faculty page for ML systems researchers
- If you don't know anyone, check who published at MLSys/NeurIPS from CU Boulder recently
