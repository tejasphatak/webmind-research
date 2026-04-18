#!/usr/bin/env python3
"""
Self-Training Distributed Brain — v1 proof of concept
======================================================
A network of neurons distributed across simulated devices that learns
through usage. No pre-training. Neurons discover each other via
simulated P2P. Connections strengthen with use (Hebbian).

Task: Learn simple sequence patterns (e.g., "AB" → "CD", "12" → "34")
This is the smallest possible test of the concept.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import json
import time
from collections import defaultdict

DEVICE = "cpu"  # Tiny enough for CPU


class Neuron(nn.Module):
    """A single neuron that lives on a device.
    Has input weights, output weights, and a Hebbian learning rule."""

    def __init__(self, neuron_id, dim=32):
        super().__init__()
        self.id = neuron_id
        self.dim = dim
        # Input transform: receives signals, produces activation
        self.w_in = nn.Parameter(torch.randn(dim, dim) * 0.01)
        self.bias = nn.Parameter(torch.zeros(dim))
        # Activation history (for Hebbian learning)
        self.activation_count = 0
        self.last_activation = None

    def forward(self, x):
        """Compute activation from input signal."""
        out = F.relu(F.linear(x, self.w_in, self.bias))
        self.last_activation = out.detach()
        self.activation_count += 1
        return out


class Connection:
    """A connection between two neurons. Strength changes with use."""

    def __init__(self, src_id, dst_id, dim=32):
        self.src = src_id
        self.dst = dst_id
        self.weight = nn.Parameter(torch.randn(dim, dim) * 0.01)
        self.strength = 1.0  # Hebbian strength (grows with use)
        self.use_count = 0

    def transmit(self, signal):
        """Send signal through this connection, scaled by strength."""
        self.use_count += 1
        return F.linear(signal, self.weight) * self.strength


class DistributedBrain:
    """A self-organizing network of neurons across simulated devices."""

    def __init__(self, n_devices=4, neurons_per_device=10, dim=32,
                 vocab_size=128):
        self.dim = dim
        self.vocab_size = vocab_size
        self.n_devices = n_devices
        self.neurons_per_device = neurons_per_device

        # Input/output layers (fixed structure)
        self.embed = nn.Embedding(vocab_size, dim)
        self.output_proj = nn.Linear(dim, vocab_size)

        # Create neurons on devices
        self.neurons = {}
        self.device_map = {}  # neuron_id → device_id
        nid = 0
        for dev in range(n_devices):
            for _ in range(neurons_per_device):
                self.neurons[nid] = Neuron(nid, dim)
                self.device_map[nid] = dev
                nid += 1

        self.total_neurons = nid

        # Connections (start sparse, grow with discovery)
        self.connections = {}  # (src, dst) → Connection
        # Initially: random sparse connections (simulating P2P discovery)
        n_initial = self.total_neurons * 2  # 2 connections per neuron avg
        for _ in range(n_initial):
            src = random.randint(0, self.total_neurons - 1)
            dst = random.randint(0, self.total_neurons - 1)
            if src != dst and (src, dst) not in self.connections:
                self.connections[(src, dst)] = Connection(src, dst, dim)

        # Routing table: which neurons to activate for input
        # Learned through usage
        self.routing_weights = torch.randn(dim, self.total_neurons) * 0.01

        print(f"  Brain: {self.total_neurons} neurons across {n_devices} devices, "
              f"{len(self.connections)} initial connections", flush=True)

    def parameters(self):
        """All learnable parameters."""
        params = list(self.embed.parameters()) + list(self.output_proj.parameters())
        params.append(self.routing_weights)
        for n in self.neurons.values():
            params.extend(n.parameters())
        for c in self.connections.values():
            params.append(c.weight)
        return params

    def forward(self, input_ids):
        """Process input through the distributed brain.
        1. Embed input tokens
        2. Route to active neurons (sparse)
        3. Propagate through connections
        4. Collect at output layer
        """
        B, S = input_ids.shape

        # Embed
        x = self.embed(input_ids)  # [B, S, dim]
        x_flat = x.mean(dim=1)     # [B, dim] — aggregate input

        # Route: decide which neurons to activate (top-k sparse)
        routing_scores = F.linear(x_flat, self.routing_weights.T)  # [B, n_neurons]
        k = max(self.total_neurons // 4, 3)  # activate ~25% of neurons
        top_k = torch.topk(routing_scores, k, dim=-1)
        active_ids = top_k.indices[0].tolist()  # same routing for all batch

        # Phase 1: Input neurons process signal
        neuron_states = {}
        for nid in active_ids:
            neuron_states[nid] = self.neurons[nid](x_flat)

        # Phase 2: Propagate through connections (1 hop)
        next_states = {}
        for (src, dst), conn in self.connections.items():
            if src in neuron_states and dst in active_ids:
                signal = conn.transmit(neuron_states[src])
                if dst in next_states:
                    next_states[dst] = next_states[dst] + signal
                else:
                    next_states[dst] = signal

        # Phase 3: Second activation
        for nid, signal in next_states.items():
            if nid in neuron_states:
                neuron_states[nid] = neuron_states[nid] + self.neurons[nid](signal)
            else:
                neuron_states[nid] = self.neurons[nid](signal)

        # Phase 4: Aggregate and project to output
        if neuron_states:
            aggregated = torch.stack(list(neuron_states.values())).mean(dim=0)
        else:
            aggregated = torch.zeros(B, self.dim)

        logits = self.output_proj(aggregated)  # [B, vocab_size]
        return logits.unsqueeze(1).expand(-1, S, -1)  # [B, S, vocab_size]

    def discover_new_connection(self):
        """Simulate P2P discovery: randomly create a new connection."""
        src = random.randint(0, self.total_neurons - 1)
        dst = random.randint(0, self.total_neurons - 1)
        if src != dst and (src, dst) not in self.connections:
            self.connections[(src, dst)] = Connection(src, dst, self.dim)
            return True
        return False

    def prune_weak_connections(self, threshold=0.1):
        """Remove connections that are rarely used."""
        to_remove = []
        for key, conn in self.connections.items():
            if conn.strength < threshold and conn.use_count < 5:
                to_remove.append(key)
        for key in to_remove:
            del self.connections[key]
        return len(to_remove)

    def hebbian_update(self, lr=0.01):
        """Strengthen connections between co-activated neurons."""
        for (src, dst), conn in self.connections.items():
            src_n = self.neurons[src]
            dst_n = self.neurons[dst]
            if src_n.last_activation is not None and dst_n.last_activation is not None:
                # Hebbian: if both fired, strengthen connection
                co_activation = (src_n.last_activation.abs().mean() *
                                dst_n.last_activation.abs().mean()).item()
                conn.strength = min(conn.strength + lr * co_activation, 5.0)
            else:
                # Decay unused connections
                conn.strength *= 0.999


def run_experiment():
    print("=== SELF-TRAINING DISTRIBUTED BRAIN ===", flush=True)

    brain = DistributedBrain(n_devices=4, neurons_per_device=10, dim=32,
                             vocab_size=128)

    # Simple training data: learn character mappings
    # "AB" → "CD", "EF" → "GH", etc.
    train_pairs = []
    for i in range(0, 52, 4):  # 13 pairs
        inp = [65 + i, 65 + i + 1]       # e.g., [65, 66] = "AB"
        out = [65 + i + 2, 65 + i + 3]   # e.g., [67, 68] = "CD"
        train_pairs.append((inp, out))

    optimizer = torch.optim.Adam(brain.parameters(), lr=1e-3)

    print(f"\n  Training pairs: {len(train_pairs)}", flush=True)
    print(f"  Example: {chr(train_pairs[0][0][0])}{chr(train_pairs[0][0][1])} → "
          f"{chr(train_pairs[0][1][0])}{chr(train_pairs[0][1][1])}", flush=True)

    # Training loop (simulating user interactions)
    n_interactions = 5000
    correct_window = []
    losses = []

    for step in range(n_interactions):
        # Random user interaction
        inp, target = random.choice(train_pairs)
        input_ids = torch.tensor([inp])
        target_ids = torch.tensor([target])

        # Forward
        logits = brain.forward(input_ids)  # [1, 2, vocab_size]

        # Loss
        loss = F.cross_entropy(logits.view(-1, brain.vocab_size),
                              target_ids.view(-1))

        # Backward (gradient-based learning)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Hebbian update (reinforces active paths)
        brain.hebbian_update()

        # P2P discovery (occasionally find new connections)
        if random.random() < 0.05:  # 5% chance per step
            brain.discover_new_connection()

        # Pruning (occasionally remove weak connections)
        if (step + 1) % 500 == 0:
            pruned = brain.prune_weak_connections()

        # Track accuracy
        with torch.no_grad():
            pred = logits.argmax(dim=-1)
            correct = (pred == target_ids).all().item()
            correct_window.append(correct)
            if len(correct_window) > 100:
                correct_window.pop(0)
            losses.append(loss.item())

        if (step + 1) % 500 == 0:
            acc = sum(correct_window) / len(correct_window)
            avg_loss = sum(losses[-500:]) / min(len(losses), 500)
            n_conns = len(brain.connections)
            print(f"    step {step+1}/{n_interactions}  "
                  f"acc={acc:.3f}  loss={avg_loss:.4f}  "
                  f"connections={n_conns}", flush=True)

    # Final eval
    print("\n=== FINAL EVAL ===", flush=True)
    correct, total = 0, 0
    for inp, target in train_pairs:
        input_ids = torch.tensor([inp])
        target_ids = torch.tensor([target])
        with torch.no_grad():
            logits = brain.forward(input_ids)
            pred = logits.argmax(dim=-1)
            match = (pred == target_ids).all().item()
            correct += match
            total += 1
            in_str = ''.join(chr(c) for c in inp)
            tgt_str = ''.join(chr(c) for c in target)
            pred_str = ''.join(chr(c.item()) for c in pred[0] if 32 <= c.item() < 128)
            status = "OK" if match else "WRONG"
            print(f"    {in_str} → {tgt_str}  pred: {pred_str}  [{status}]",
                  flush=True)

    acc = correct / total
    print(f"\n  Final accuracy: {correct}/{total} = {acc:.1%}", flush=True)
    print(f"  Connections: {len(brain.connections)}", flush=True)
    print(f"  Neurons activated: avg {sum(n.activation_count for n in brain.neurons.values()) / brain.total_neurons:.0f} times each",
          flush=True)

    # Resilience test: kill a device
    print("\n=== RESILIENCE: Kill device 0 ===", flush=True)
    killed_neurons = [nid for nid, dev in brain.device_map.items() if dev == 0]
    # Zero out killed neurons
    for nid in killed_neurons:
        brain.neurons[nid].w_in.data.zero_()
        brain.neurons[nid].bias.data.zero_()

    correct_after = 0
    for inp, target in train_pairs:
        input_ids = torch.tensor([inp])
        target_ids = torch.tensor([target])
        with torch.no_grad():
            logits = brain.forward(input_ids)
            pred = logits.argmax(dim=-1)
            if (pred == target_ids).all().item():
                correct_after += 1

    print(f"  Before kill: {correct}/{total} = {correct/total:.1%}", flush=True)
    print(f"  After kill: {correct_after}/{total} = {correct_after/total:.1%}",
          flush=True)
    print(f"  Degradation: {(correct - correct_after)}/{total} lost", flush=True)

    return acc, correct_after / total


if __name__ == "__main__":
    run_experiment()
