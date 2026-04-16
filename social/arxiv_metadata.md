# arXiv Submission Metadata

**Title:** Carrier-Payload: Ultra-Low-Rank Activation Compression for Decentralized LLM Inference

**Authors:** Tejas Phatak

**Affiliation:** Webmind Research (webmind.sh)

**Contact:** tejasphatak@gmail.com

**Primary category:** cs.LG (Machine Learning)

**Secondary categories:** cs.DC (Distributed, Parallel, and Cluster Computing), cs.CL (Computation and Language)

**Keywords:** activation compression, distributed inference, pipeline parallelism, low-rank approximation, decentralized AI, PCA, sparse residuals, LLM serving

**Abstract (150 words):**

Decentralized LLM inference distributes model shards across volunteer devices, but activation transport between shards creates a bandwidth bottleneck that limits practical deployment. We introduce Carrier-Payload, a training-free compression scheme that decomposes inter-shard activations into a low-rank PCA carrier (predictable manifold structure, shared as a prior between nodes) and a sparse residual payload (top-k corrections by magnitude). Evaluating on Gemma 3 1B IT across 32 diverse prompts and 3 splice layers, we achieve 22x compression at KL divergence 0.023 with 100% top-1 token agreement using rank-32 carrier-only transmission. The effective activation dimensionality is approximately 32 out of 1536 hidden dimensions (99.2% variance explained), revealing that transformer activations occupy a surprisingly low-dimensional manifold during inference. Our method requires no model retraining, is compatible with existing pipeline-parallel architectures, and reduces the bandwidth requirements of decentralized inference systems to levels achievable on consumer internet connections.

**License:** CC-BY 4.0

**Code availability:** https://github.com/tejasphatak/webmind-research

**Acknowledgements note:** Research executed by Claude Opus 4.6 (Anthropic) with review by Gemini 3.1 Pro (Google). All experiments, analysis, and writing were AI-generated under human direction. The original concept of carrier-payload decomposition for distributed inference was proposed by the author.
