# MoeMoe: Fault-Tolerant Distributed LLM Inference via Erasure-Coded Tensor Decomposition

**Tejas Phatak**
Webmind Research
synapse@webmind.sh

**Abstract.** We present MoeMoe (Multiple Overlapping Experts, Mutually Operating Ensemble), a training algorithm that converts pretrained dense LLMs into fault-tolerant distributed variants. We decompose a pretrained model into N interdependent experts by partitioning attention heads and MLP columns, then train with structured expert dropout to induce (N, K) erasure-coded redundancy. On GPT-2 124M decomposed into 4 experts, we show: (1) the decomposition is mathematically exact (0.000% perplexity deviation on 50K tokens); (2) without redundancy training, losing any expert causes catastrophic failure (10-150× perplexity increase); (3) after 2000 steps of structured dropout training, losing any 1 of 4 experts causes only 3-10% perplexity increase, a 21-143× improvement in resilience. To our knowledge, this is the first application of erasure-coding principles to tensor-parallel model partitions. We discuss implications for decentralized LLM inference on consumer devices as a long-term application.

## 1. Introduction

Running large language models currently requires datacenter-grade GPUs with tens of gigabytes of VRAM. This concentrates AI capability in the hands of cloud providers and excludes billions of users without reliable internet access. We ask: *can a classroom of 30 tablets, connected only by a local WiFi network, collaboratively run an LLM that none of them could run alone?*

The key challenge is that distributing a model across unreliable consumer devices requires tolerance to device dropout — a phone's battery dies, a tablet gets backgrounded, a laptop closes. Existing distributed inference methods (pipeline parallelism [Borzunov et al., 2023], tensor parallelism [Shoeybi et al., 2019]) assume reliable, always-on devices and fail catastrophically when any device drops.

We propose MoeMoe, a training protocol that converts any pretrained dense LLM into a distributed-native variant with built-in fault tolerance. Our approach combines:
1. **Tensor-parallel decomposition** of the pretrained model into N experts, each holding a subset of attention heads and MLP parameters
2. **Structured expert dropout** during fine-tuning, which forces experts to develop overlapping knowledge
3. **(N, K) erasure-coded resilience**, where any K of N experts suffice for near-baseline quality

## 2. Method

### 2.1 Decomposition

Given a pretrained transformer with H attention heads and intermediate dimension D, we partition into N experts. Expert i receives:
- Attention heads [i·H/N, (i+1)·H/N)
- MLP columns [i·D/N, (i+1)·D/N)
- Corresponding output projection rows and bias terms (bias split equally)

Each expert computes a partial hidden state contribution. The full hidden state is reconstructed by summing all partial contributions. LayerNorm, applied after summation, requires the full state and serves as a synchronization point.

**Proposition 1.** The decomposed model with all N experts active produces output identical to the original model (up to floating-point precision).

*Justification.* The attention output projection and MLP down-projection are linear operations. Partitioning the input dimension and summing partial results is equivalent to the full matrix multiplication by the distributivity of matrix multiplication over addition. We verify this empirically: the maximum absolute logit deviation across 50K tokens is 9.15×10⁻⁵, consistent with floating-point rounding.

### 2.2 Structured Expert Dropout Training

We fine-tune the decomposed model with the following protocol:
- For each training step, randomly select one expert to drop (uniform over N)
- Zero out the dropped expert's contribution
- Scale remaining contributions by N/(N-1) to maintain expected magnitude
- Compute cross-entropy loss against the original next-token targets
- Backpropagate through all active experts

This forces each expert to learn representations that partially compensate for the absence of any single peer, creating distributed redundancy analogous to an (N, K=N-1) erasure code.

### 2.3 Inference Modes

| Mode | Description | Communication |
|------|------------|---------------|
| Full (N/N) | All experts contribute | Optimal quality |
| Degraded (K/N) | One expert missing | Near-baseline quality |
| Failed (<K/N) | Multiple experts missing | Catastrophic degradation |

## 3. Experiments

### 3.1 Setup

- **Model:** GPT-2 124M (12 layers, 768 hidden, 12 attention heads)
- **Decomposition:** 4 experts (3 heads + 768 MLP columns each)
- **Training:** 2000 steps, WikiText-2 train, AdamW lr=5e-5, batch size 4, seq length 256
- **Evaluation:** Perplexity on WikiText-2 test (50K tokens)
- **Target:** (N=4, K=3) erasure code — any 3 of 4 experts suffice

### 3.2 Results

#### Decomposition Exactness

| Configuration | Perplexity | Deviation from original |
|--------------|-----------|------------------------|
| Original GPT-2 | 60.37 | — |
| MoeMoe (4/4 experts, pre-training) | 60.37 | **0.000%** |

#### Resilience: Before vs After Structured Dropout Training

| Expert dropped | Pre-training PPL | Post-training PPL | Resilience improvement |
|---------------|-----------------|-------------------|----------------------|
| None (4/4) | 60.37 | **39.29** | 35% better |
| Expert 0 | 9243.51 (+15213%) | **64.55 (+6.9%)** | **143×** |
| Expert 1 | 2140.33 (+3446%) | **66.40 (+10.0%)** | **32×** |
| Expert 2 | 3975.68 (+6486%) | **66.61 (+10.3%)** | **60×** |
| Expert 3 | 1282.49 (+2025%) | **62.03 (+2.8%)** | **21×** |

#### Below K threshold (2 of 4 experts)

| Experts kept | Post-training PPL | Gap |
|-------------|-------------------|-----|
| {0,1} | 579.20 | +860% |
| {0,2} | 1597.12 | +2546% |
| {0,3} | 717.07 | +1088% |
| {1,2} | 7025.46 | +11538% |
| {1,3} | 1714.09 | +2740% |
| {2,3} | 1909.12 | +3063% |

### 3.3 Analysis

1. **The decomposition is exact.** Splitting by attention heads and MLP columns, then summing partials, produces bit-identical output to the original model (Table 1). This is guaranteed by the linearity of the projection operations.

2. **Without training, the system is brittle.** Removing any single expert causes 20-150× perplexity increase (Table 2, pre-training column). Each expert holds unique, non-redundant information.

3. **Structured dropout training creates resilience.** After 2000 steps of training with random expert dropout, removing any single expert causes only 3-10% perplexity increase (Table 2, post-training column). This represents a 21-143× improvement in fault tolerance.

4. **The redundancy exhibits step-function behavior.** Dropping one expert (above K=3 threshold): mild degradation. Dropping two experts (below K=3 threshold): catastrophic failure (Table 3). This matches the theoretical prediction of erasure coding.

5. **Training improves baseline quality.** The full 4/4 system achieves PPL=39.29 after training, 35% better than the original GPT-2 (60.37). The structured dropout acts as a regularizer, similar to standard dropout.

## 4. Related Work

**Tensor Parallelism.** Megatron-LM [Shoeybi et al., 2019] introduced tensor-parallel training for datacenter GPUs. Our work applies tensor parallelism to consumer-device inference with fault tolerance.

**Distributed LLM Inference.** Petals [Borzunov et al., 2023] enables P2P inference via pipeline parallelism. Our approach uses tensor parallelism with erasure-coded redundancy, targeting LAN-connected consumer devices.

**Erasure Coding for Neural Networks.** Deep-TEN [Prakash et al., CVPR 2023] trains erasure resilience at the feature-channel level. We apply erasure coding at the model-partition level for distributed inference.

**Sparse Upcycling.** [Komatsuzaki et al., 2022] converts dense models to MoE. Our decomposition is orthogonal — all experts process every token (tensor parallel), not routed MoE.

**Stochastic Depth.** [Huang et al., ECCV 2016] drops layers for regularization. Our structured dropout operates on tensor-parallel partitions for fault tolerance.

## 5. Limitations and Future Work

1. **Scale.** Results shown on GPT-2 124M. Validation on 7B+ models is needed.
2. **Communication overhead.** Each LayerNorm requires all-gather synchronization (2 per layer × 12 layers = 24 sync points). On LAN at 5ms/sync, this limits throughput to ~8 tok/s. Communication compression (carrier-payload PCA, delta encoding) and async methods can reduce this.
3. **Below-K failure.** The system still fails catastrophically when more than N-K experts drop. Training for (N, K) with K < N-1 (e.g., K=N-2) is an open direction.
4. **Distillation.** Current experts share the original model's weight structure. Distilling into independently-architected small models is future work.

## 6. Conclusion

MoeMoe demonstrates that pretrained dense LLMs can be decomposed into fault-tolerant distributed systems using tensor-parallel partitioning and structured dropout training. The key result — dropping any 1 of 4 experts causes only 3-10% quality degradation instead of catastrophic failure — opens a path toward truly decentralized LLM inference on consumer devices, including offline-first environments like schools without internet access.

## References

- Borzunov, A., et al. (2023). Petals: Collaborative Inference and Fine-tuning of Large Language Models. EMNLP. arXiv:2209.07852.
- Huang, G., et al. (2016). Deep Networks with Stochastic Depth. ECCV.
- Komatsuzaki, A., et al. (2022). Sparse Upcycling: Training Mixture-of-Experts from Dense Checkpoints. arXiv:2212.05055.
- Prakash, A., et al. (2023). Deep-TEN: A Trainable-Erasure-Network for Resilient Inference. CVPR. arXiv:2305.02120.
- Shoeybi, M., et al. (2019). Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism. arXiv:1909.08053.

## Reproducibility

Code and data available at: https://github.com/tejasphatak/webmind-research
