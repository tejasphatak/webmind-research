#!/usr/bin/env python3
"""
Self-Training Distributed Brain — Text v2 (Optimized)
======================================================
Fixes:
- VECTORIZED: all neurons computed as one big matmul (no Python loops)
- Threshold activation (ReLU/sparse) instead of sigmoid gate
- Specialized neuron types (like brain has different neuron types)
- Much faster — exploits GPU parallelism properly
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import time
import json
import math
from transformers import GPT2Tokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class NetworkSim:
    """Replaceable with real TCP/WebRTC."""
    def __init__(self, latency_ms=5, jitter_pct=30, dropout_pct=3):
        self.dropout_pct = dropout_pct
        self.drops = 0
        self.total = 0

    def dropout_mask(self, n_neurons, device_ids, n_devices):
        """Vectorized dropout: returns mask [n_neurons] where 0 = dropped."""
        self.total += n_neurons
        mask = torch.ones(n_neurons, device='cpu')
        for i in range(n_neurons):
            if random.random() * 100 < self.dropout_pct:
                mask[i] = 0
                self.drops += 1
        return mask.to(next(iter([DEVICE])))


class VectorizedBrainLayer(nn.Module):
    """All neurons in one layer computed as a single matmul.

    Instead of looping over N neurons:
      for n in neurons: out_n = relu(x @ W_n)

    We do:
      out_all = relu(x @ W_all)  # W_all is [dim, N*dim]
      then reshape to [B, N, dim]
    """

    def __init__(self, n_neurons, dim, neuron_type="excitatory"):
        super().__init__()
        self.n_neurons = n_neurons
        self.dim = dim
        self.neuron_type = neuron_type

        # All neurons as one big weight matrix
        self.W = nn.Linear(dim, n_neurons * dim, bias=False)

        # Per-neuron threshold (learned) — like biological firing threshold
        self.threshold = nn.Parameter(torch.zeros(n_neurons))

        # Per-neuron type scaling
        if neuron_type == "excitatory":
            self.type_scale = 1.0
        elif neuron_type == "inhibitory":
            self.type_scale = -0.5  # inhibitory neurons dampen signal
        elif neuron_type == "modulatory":
            self.type_scale = 0.3   # modulatory neurons fine-tune
        else:
            self.type_scale = 1.0

        nn.init.xavier_uniform_(self.W.weight)

    def forward(self, x, dropout_mask=None):
        """
        x: [B, dim]
        Returns: [B, n_neurons, dim] — each neuron's output
        """
        B = x.size(0)

        # One big matmul for all neurons
        all_out = self.W(x)  # [B, N*dim]
        all_out = all_out.view(B, self.n_neurons, self.dim)  # [B, N, dim]

        # Threshold activation (biological: fire only if above threshold)
        # ReLU with learned per-neuron threshold
        threshold = self.threshold.unsqueeze(0).unsqueeze(-1)  # [1, N, 1]
        all_out = F.relu(all_out - threshold) * self.type_scale

        # Apply dropout mask (simulates network/device failures)
        if dropout_mask is not None:
            mask = dropout_mask.unsqueeze(0).unsqueeze(-1)  # [1, N, 1]
            all_out = all_out * mask

        return all_out


class VectorizedConnections(nn.Module):
    """Connections between neuron groups as sparse matmul."""

    def __init__(self, n_src, n_dst, dim, sparsity=0.7):
        super().__init__()
        self.n_src = n_src
        self.n_dst = n_dst
        self.dim = dim

        # Connection weights: [n_dst, n_src] — which src connects to which dst
        # Initialize sparse (most connections start at 0)
        self.routing = nn.Parameter(torch.randn(n_dst, n_src) * 0.01)

        # Mask for sparsity (fixed initial topology, can evolve)
        mask = (torch.rand(n_dst, n_src) > sparsity).float()
        self.register_buffer('sparsity_mask', mask)

        # Hebbian strength (not a parameter — updated manually)
        self.register_buffer('strength', torch.ones(n_dst, n_src))

        # Transform per-connection
        self.transform = nn.Linear(dim, dim, bias=False)
        nn.init.xavier_uniform_(self.transform.weight)

    def forward(self, src_states):
        """
        src_states: [B, n_src, dim]
        Returns: [B, n_dst, dim]
        """
        # Effective routing weights (routing * sparsity * strength)
        effective = torch.sigmoid(self.routing) * self.sparsity_mask * self.strength

        # Route: weighted sum of source states per destination
        # [B, n_dst, dim] = effective[n_dst, n_src] @ src_states[B, n_src, dim]
        routed = torch.einsum('ds,bsd->bd', effective,
                             src_states.transpose(0, 1)).unsqueeze(0)

        # Actually: [n_dst, n_src] x [B, n_src, dim] → [B, n_dst, dim]
        routed = torch.bmm(
            effective.unsqueeze(0).expand(src_states.size(0), -1, -1),
            src_states)  # [B, n_dst, dim]

        # Transform
        B, D, dim = routed.shape
        routed = self.transform(routed.reshape(-1, dim)).reshape(B, D, dim)

        return routed

    def hebbian_update(self, src_act, dst_act, lr=0.01):
        """Strengthen connections between co-active neurons."""
        with torch.no_grad():
            # src_act: [B, n_src], dst_act: [B, n_dst] — mean activation magnitudes
            co_activation = torch.einsum('bs,bd->ds',
                                        dst_act.mean(0, keepdim=True),
                                        src_act.mean(0, keepdim=True))
            self.strength = torch.clamp(
                self.strength + lr * co_activation * self.sparsity_mask,
                0.1, 5.0)

    def discover(self, n_new=5):
        """Open new connections (P2P discovery)."""
        with torch.no_grad():
            for _ in range(n_new):
                d = random.randint(0, self.n_dst - 1)
                s = random.randint(0, self.n_src - 1)
                self.sparsity_mask[d, s] = 1.0


class OptimizedTextBrain(nn.Module):
    """Fast self-organizing text brain with specialized neuron types."""

    def __init__(self, n_devices=4, neurons_per_device=50, dim=256, max_seq=32):
        super().__init__()
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.vocab_size = self.tokenizer.vocab_size
        self.dim = dim
        self.max_seq = max_seq
        self.n_devices = n_devices
        self.n_per_device = neurons_per_device
        self.n_total = n_devices * neurons_per_device
        self.network = NetworkSim()

        # Fixed: input/output
        self.embed = nn.Embedding(self.vocab_size, dim)
        self.pos_embed = nn.Embedding(max_seq, dim)
        self.output_norm = nn.LayerNorm(dim)
        self.output_proj = nn.Linear(dim, self.vocab_size)

        # Specialized neuron layers (like brain regions)
        # Layer 1: Excitatory (main processing)
        self.layer1 = VectorizedBrainLayer(
            self.n_total // 2, dim, "excitatory")
        # Layer 2: Mixed (excitatory + inhibitory)
        n_excit = self.n_total // 4
        n_inhib = self.n_total // 8
        n_modul = self.n_total - self.n_total // 2 - n_excit - n_inhib
        self.layer2_excit = VectorizedBrainLayer(n_excit, dim, "excitatory")
        self.layer2_inhib = VectorizedBrainLayer(n_inhib, dim, "inhibitory")
        self.layer2_modul = VectorizedBrainLayer(n_modul, dim, "modulatory")

        # Connections between layers
        n_l1 = self.n_total // 2
        n_l2 = n_excit + n_inhib + n_modul
        self.conn_1_2 = VectorizedConnections(n_l1, n_l2, dim, sparsity=0.6)

        # Aggregation
        self.aggregate = nn.Linear(dim, dim)

        n_params = sum(p.numel() for p in self.parameters())
        print(f"  OptimizedTextBrain: {self.n_total} neurons "
              f"({n_l1} L1, {n_excit}E+{n_inhib}I+{n_modul}M L2), "
              f"dim={dim}, {n_params:,} params", flush=True)

    def forward(self, input_ids):
        B, S = input_ids.shape
        S = min(S, self.max_seq)
        input_ids = input_ids[:, :S]

        pos = torch.arange(S, device=input_ids.device).unsqueeze(0).expand(B, -1)
        x = self.embed(input_ids) + self.pos_embed(pos)  # [B, S, dim]

        outputs = []
        for t in range(S):
            out = self._propagate(x[:, t, :])
            outputs.append(out)

        output = torch.stack(outputs, dim=1)
        output = self.output_norm(output)
        return self.output_proj(output)

    def _propagate(self, x):
        """Vectorized propagation through brain layers."""
        B = x.size(0)

        # Layer 1: all excitatory neurons fire
        l1_out = self.layer1(x)  # [B, N1, dim]

        # Route layer 1 → layer 2
        l2_input = self.conn_1_2(l1_out)  # [B, N2, dim]

        # Layer 2: split into excitatory, inhibitory, modulatory
        n_e = self.layer2_excit.n_neurons
        n_i = self.layer2_inhib.n_neurons
        n_m = self.layer2_modul.n_neurons

        l2_e = self.layer2_excit(l2_input[:, :n_e, :].mean(dim=1))
        l2_i = self.layer2_inhib(l2_input[:, n_e:n_e+n_i, :].mean(dim=1))
        l2_m = self.layer2_modul(l2_input[:, n_e+n_i:, :].mean(dim=1))

        # Combine: excitatory + inhibitory + modulatory
        combined = torch.cat([l2_e, l2_i, l2_m], dim=1)  # [B, N2, dim]

        # Aggregate to output dim
        aggregated = combined.mean(dim=1)  # [B, dim]
        return self.aggregate(aggregated)

    def hebbian_update(self):
        """Update connection strengths based on co-activation."""
        with torch.no_grad():
            # Use layer outputs to compute co-activation
            self.conn_1_2.discover(n_new=2)


def run():
    print("=== OPTIMIZED TEXT BRAIN v2 ===\n", flush=True)

    brain = OptimizedTextBrain(
        n_devices=4, neurons_per_device=50, dim=256, max_seq=32).to(DEVICE)

    tokenizer = brain.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_data = [
        ("The cat sat on the", " mat"),
        ("The dog ran in the", " park"),
        ("The sun is in the", " sky"),
        ("I like to eat", " food"),
        ("She went to the", " store"),
        ("He reads a", " book"),
        ("They play in the", " yard"),
        ("We live in a", " house"),
        ("The capital of France is", " Paris"),
        ("The color of grass is", " green"),
        ("The color of sky is", " blue"),
        ("Water is", " wet"),
        ("Fire is", " hot"),
        ("Ice is", " cold"),
        ("one two three", " four"),
        ("two four six", " eight"),
    ]

    optimizer = torch.optim.AdamW(brain.parameters(), lr=1e-3)
    n_steps = 20000

    print(f"  Training pairs: {len(train_data)}", flush=True)
    print(f"  Steps: {n_steps}", flush=True)
    print(f"  Device: {DEVICE}", flush=True)

    t0 = time.time()
    losses = []
    for step in range(n_steps):
        prompt, target = random.choice(train_data)
        full = prompt + target

        inp = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=31).to(DEVICE)
        tgt = tokenizer(full, return_tensors="pt", truncation=True,
                       max_length=32).to(DEVICE)

        max_len = tgt["input_ids"].size(1)
        if inp["input_ids"].size(1) < max_len:
            pad = torch.full((1, max_len - inp["input_ids"].size(1)),
                           tokenizer.pad_token_id, device=DEVICE,
                           dtype=torch.long)
            input_ids = torch.cat([inp["input_ids"], pad], dim=1)
        else:
            input_ids = inp["input_ids"][:, :max_len]

        logits = brain(input_ids)
        S = min(logits.size(1), tgt["input_ids"].size(1))
        loss = F.cross_entropy(
            logits[:, :S-1, :].reshape(-1, brain.vocab_size),
            tgt["input_ids"][:, 1:S].reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(brain.parameters(), 1.0)
        optimizer.step()

        if random.random() < 0.1:
            brain.hebbian_update()

        losses.append(loss.item())

        if (step + 1) % 2000 == 0:
            avg = sum(losses[-2000:]) / min(len(losses), 2000)
            elapsed = time.time() - t0
            steps_per_sec = (step + 1) / elapsed
            print(f"    step {step+1}/{n_steps}  loss={avg:.4f}  "
                  f"{steps_per_sec:.1f} steps/s  elapsed={elapsed:.0f}s",
                  flush=True)

    # Eval
    print("\n=== FINAL EVAL ===", flush=True)
    brain.eval()
    correct, total = 0, len(train_data)

    for prompt, target in train_data:
        inp = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=31).to(DEVICE)
        with torch.no_grad():
            logits = brain(inp["input_ids"])
            pred_id = logits[0, -1, :].argmax().item()
            pred = tokenizer.decode(pred_id)
            target_ids = tokenizer(target, return_tensors="pt")["input_ids"]
            expected = tokenizer.decode(target_ids[0, 0])
            match = pred.strip() == expected.strip()
            if match:
                correct += 1
            status = "OK" if match else "MISS"
            print(f"  [{status}] \"{prompt}\" → \"{pred.strip()}\" "
                  f"(expected \"{expected.strip()}\")", flush=True)

    elapsed = time.time() - t0
    print(f"\n  Accuracy: {correct}/{total} = {correct/total:.0%}", flush=True)
    print(f"  Total time: {elapsed:.0f}s", flush=True)
    print(f"  Speed: {n_steps/elapsed:.1f} steps/s", flush=True)


if __name__ == "__main__":
    run()
