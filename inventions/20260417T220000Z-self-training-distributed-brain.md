---
Inventor: Tejas Phatak
Date: 2026-04-17
Status: Concept — needs validation
---

# Self-Training Distributed Brain

## The Idea (Tejas, 2026-04-17)

A distributed model that trains ITSELF as people use it. No pre-training. No distillation. The network starts near-blank and learns through usage — like a human brain.

### Core principles

1. **Self-training through usage**: Users send prompts and answers. The network learns from every interaction. No separate training phase — inference IS training.

2. **Activation paths, not full network**: Like the human brain, not every neuron fires for every input. Only relevant paths activate. The model learns WHICH paths to activate for which inputs.

3. **Connections are not permanent**: Neural connections strengthen through use (Hebbian: "neurons that fire together wire together"). Unused connections weaken and eventually prune. The network is alive — constantly reorganizing.

4. **Distributed by nature**: Each device contributes "neurons" (compute + memory). The network spans all devices. No single device holds the full model — the model IS the network.

5. **Resilient by design**: If a device goes offline, its neurons go silent. But the network has redundant paths (like the brain after injury — other regions compensate). The model degrades gracefully.

6. **Input and output layers guaranteed**: The only fixed structure is: input layer (tokenizer → embeddings) and output layer (hidden → logits). Everything in between is learned, self-organizing.

## How it works

```
User sends prompt → 
  Input layer (on user's device) tokenizes →
  Signal propagates through distributed neurons →
  Only SOME neurons activate (sparse, learned routing) →
  Activated neurons on other devices fire and send results forward →
  Output layer produces response →
  User provides feedback (explicit or implicit) →
  Backprop/Hebbian update strengthens the paths that worked →
  Repeat
```

## Biological analogies

| Brain | This system |
|-------|-------------|
| Neurons | Compute units on devices |
| Synapses | Network connections between devices |
| Synapse strengthening | Weight updates on active paths |
| Synapse pruning | Unused connections weakened/removed |
| Sparse activation | Only relevant paths fire per prompt |
| Neuroplasticity | Network reorganizes through usage |
| Multiple pathways | Redundant routes for resilience |
| No central controller | Distributed, self-organizing |

## What makes this different from existing approaches

| Approach | Pre-training | Distribution | Self-organizing |
|----------|-------------|-------------|----------------|
| Standard LLM | Massive | No | No |
| Federated learning | Yes (rounds) | Yes | No |
| MoE | Yes | Partial | No (fixed routing) |
| Collaborative distillation | Yes (from teacher) | Yes | No |
| **This (Self-Training Brain)** | **No** | **Yes (native)** | **Yes** |

## Key research questions

1. **Convergence**: Will a randomly initialized distributed network actually learn language from user interactions? How many interactions until it produces coherent text?

2. **Sparse routing**: How does a neuron learn whether to activate for a given input? Options:
   - Learned router (like MoE but distributed)
   - Hebbian: activate if input matches learned pattern, strengthen if output was good
   - Random initially, prune what doesn't work (evolutionary)

3. **Training signal**: Where does the gradient come from?
   - User explicitly rates response (thumbs up/down)
   - Next-token prediction (autoregressive loss, like standard LLM training)
   - Contrastive: good responses vs bad responses
   - RLHF-style but distributed

4. **Cold start**: A blank network produces garbage. How to bootstrap?
   - Pre-seed with a small pretrained model on a few devices (warm start)
   - Or: accept garbage for the first N interactions, let users teach it
   - Or: start with a simple task (copying input) and build complexity

5. **Privacy**: User prompts are training data. How to train without exposing user data?
   - Differential privacy on gradients
   - Federated averaging (share weights, not data)
   - Local training, share only activation patterns

6. **Communication**: What gets sent between devices?
   - During forward pass: sparse activation signals (which neurons fired, what values)
   - During backward pass: gradients for active paths only
   - Compression: carrier-payload on activations (proven 36×)

## Connection to prior work

- **Carrier-payload compression**: reduces inter-device communication
- **Async stale-summary**: devices don't wait — use previous activation state
- **DHT + P2P**: decentralized discovery of which devices have which neurons
- **Shadow replicas**: backup critical activation paths on multiple devices
- **MoeMoe resilience**: proven that structured redundancy creates fault tolerance

## Minimum viable experiment

1. Build a tiny self-organizing network (1000 neurons across 4 simulated devices)
2. Input: simple character sequences (not full language — too complex for start)
3. Task: learn to copy input to output (trivial but proves the mechanism)
4. Training: online, from each interaction
5. Test: does it converge? How fast? What happens when a device drops?
6. Then scale: add more neurons, more devices, more complex tasks
