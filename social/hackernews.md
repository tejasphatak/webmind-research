# Hacker News Submission

**Title:** 22x activation compression makes decentralized LLM inference practical on consumer devices

**URL:** https://github.com/tejasphatak/webmind-research (or arXiv link when available)

**Submitter comment:**

Hi HN — I'm building Synapse (webmind.sh), a system that shards LLM inference across volunteer browsers via WebGPU. The main bottleneck was always bandwidth between devices, not compute.

I had an intuition: instead of transmitting full activation tensors between shards, what if we decompose them into a compact "carrier" (predictable structure) and a sparse "payload" (corrections)?

We tested this on Gemma 3 1B and found that the activation manifold has absurdly low intrinsic dimensionality — ~32 out of 1536 hidden dims capture 99.2% of the variance. This gives us 22x compression at shard boundaries with zero quality loss (100% top-1 token agreement, KL=0.023).

The practical implication: normal internet connections (even mobile) become sufficient for distributed LLM inference. You don't need NVLink — you need a creative encoding.

Method: PCA carrier (shared basis, amortized) + sparse residual (top-k outlier corrections). No retraining needed, works post-hoc on any model. All code and data are open.

Testing on larger models (7B-12B) in progress. Paper targeting MLSys.
