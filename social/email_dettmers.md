# Email to Tim Dettmers

**To:** Tim Dettmers (find email on his website or UW page)
**From:** Your CU Boulder email
**Subject:** Building on LLM.int8! — 22x activation compression for distributed inference

---

Hi Tim,

I'm a grad student at CU Boulder working on decentralized LLM inference (webmind.sh). Your work on LLM.int8! and activation outliers directly inspired part of our approach, and I wanted to share results that might interest you.

**What we found:** LLM activations at pipeline-parallel shard boundaries have effective dimensionality ~32 out of 1536 hidden dims in Gemma 3 1B. Using PCA carrier + sparse residual payload (the sparse component specifically catches the outliers your work identified), we get 22x compression at KL=0.023 with 100% top-1 agreement.

**Why it matters:** This makes distributed inference over regular internet connections practical. Instead of needing NVLink between GPUs, volunteer devices can transmit 10-22x less activation data at each shard boundary.

**The connection to your work:** The sparse residual is essentially an "outlier catcher" — the PCA carrier handles the smooth manifold structure, while the top-k% sparse component preserves the high-magnitude features that your papers showed dominate quantization error. Without the sparse component, compression above ~8x breaks down due to outlier loss.

Paper draft: https://github.com/tejasphatak/webmind-research/blob/synapse-research/initial/papers/carrier-payload-v1.md

Would love your thoughts if you have a moment. Happy to acknowledge your work's influence more explicitly if you think the framing is fair.

Best,
Tejas Phatak
MS AI, University of Colorado Boulder
https://webmind.sh
