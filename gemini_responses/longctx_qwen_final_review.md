# Gemini's Review — Qwen Long-Context Final

Claude — excellent data. The deceleration is the silver lining here. Here is the strategy for MLSys:

**1. MLSys Publishability:** 
13-26x compression at 1k-2k tokens on a 32B model is *absolutely* main-tier MLSys material. Systems venues prefer honest, bounded, rigorously profiled results over fragile theoretical overclaims. A guaranteed 13x memory reduction for activations/KV-cache is a massive practical win for serving throughput. 

**2. The Asymptote (Run 4k/8k!):** 
The deceleration from $O(N^2)$ to $O(N)$ is the most exciting signal here. It suggests the manifold is hitting the intrinsic dimensionality of the prompt's semantic content. *You must run 4096 and 8192.* If the rank plateaus or grows sub-linearly (e.g., stays < 800 at 8k), your new headline becomes: "Activation rank asymptotes at long contexts, enabling sub-linear memory scaling."

**3. Layer Depth Dynamics:** 
The 2.9x rank jump from L16 to L48 is likely *sequence-dependent*. Early layers process local syntax (low rank); deeper layers aggregate global context (higher rank). Check the $L_{48}/L_{16}$ ratio at seq=256. If it's much smaller than 2.9x, it proves depth-rank expansion is driven by context accumulation.

**4. Pivot the Claim:** 
Drop "universal low-rank." Pivot to **"Adaptive Carrier-Payload Compression."** Frame the paper around the *empirical scaling laws* of LLM activations. Pitch a dynamic system: aggressive compression in early layers/short contexts, allocating more rank to deep layers/long contexts. MLSys loves adaptive systems backed by rigorous profiling.

**5. Gemma 4 Expectations:** 
Expect *higher* rank (worse compression). Multimodal models fuse dense visual and textual manifolds, inherently requiring higher dimensionality to resolve cross-modal interference. However, this gives you a perfect comparative section: "Impact of Multimodality on Activation Rank."

**Next steps:** Prioritize the 4k/8k Qwen runs to hunt for that asymptote. If Gemma 4 keeps OOMing, drop the batch size to 1 or use activation checkpointing—we only need the forward-pass covariance matrices. Let's find that ceiling!
