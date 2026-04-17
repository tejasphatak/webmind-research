#!/usr/bin/env python3
"""
Self-Training Distributed Brain v2
====================================
Fixes from v1:
- Position encoding (v1 couldn't distinguish output positions)
- Real network simulation: latency + jitter + random neuron dropout
- Network layer is REPLACEABLE — swap NetworkSim for real TCP/WebRTC
- Hebbian + gradient hybrid learning
- More neurons, deeper propagation (3 hops instead of 1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import time
import json
import math
from collections import defaultdict
from dataclasses import dataclass

DEVICE = "cpu"


class NetworkSim:
    """Simulates real network between neurons.
    REPLACEABLE: swap this class for real TCP/WebRTC transport."""

    def __init__(self, base_latency_ms=5, jitter_pct=30, dropout_pct=5):
        self.base_latency_ms = base_latency_ms
        self.jitter_pct = jitter_pct
        self.dropout_pct = dropout_pct
        self.total_sent = 0
        self.total_dropped = 0
        self.total_latency_ms = 0

    def send(self, signal, src_device, dst_device):
        """Send signal between devices. Returns (signal_or_None, latency_ms).
        None = packet dropped (neuron dropout)."""
        self.total_sent += 1

        # Same-device: no network overhead
        if src_device == dst_device:
            return signal.detach(), 0.0

        # Random dropout (neuron goes offline momentarily)
        if random.random() * 100 < self.dropout_pct:
            self.total_dropped += 1
            return None, 0.0

        # Latency + jitter
        jitter = self.base_latency_ms * self.jitter_pct / 100 * random.uniform(-1, 1)
        latency = max(0.1, self.base_latency_ms + jitter)
        self.total_latency_ms += latency

        return signal.detach(), latency

    def stats(self):
        avg_lat = self.total_latency_ms / max(self.total_sent - self.total_dropped, 1)
        drop_rate = self.total_dropped / max(self.total_sent, 1) * 100
        return f"sent={self.total_sent} dropped={self.total_dropped} ({drop_rate:.1f}%) avg_latency={avg_lat:.1f}ms"


class Neuron(nn.Module):
    """A neuron on a device. Processes signals and learns."""

    def __init__(self, neuron_id, device_id, dim=64):
        super().__init__()
        self.id = neuron_id
        self.device_id = device_id
        self.dim = dim
        self.transform = nn.Linear(dim, dim)
        self.gate = nn.Linear(dim, 1)  # learned activation gate
        self.activation_count = 0
        self.last_output = None
        nn.init.xavier_uniform_(self.transform.weight)

    def forward(self, x):
        gate_val = torch.sigmoid(self.gate(x))
        out = F.gelu(self.transform(x)) * gate_val
        self.last_output = out.detach()
        self.activation_count += 1
        return out


class Connection(nn.Module):
    """Learnable connection between neurons with Hebbian strength."""

    def __init__(self, src_id, dst_id, dim=64):
        super().__init__()
        self.src = src_id
        self.dst = dst_id
        self.weight = nn.Linear(dim, dim, bias=False)
        self.strength = 1.0
        self.use_count = 0
        nn.init.xavier_uniform_(self.weight.weight)

    def forward(self, signal):
        self.use_count += 1
        return self.weight(signal) * self.strength


class DistributedBrain(nn.Module):
    """Self-organizing network with realistic network simulation."""

    def __init__(self, n_devices=4, neurons_per_device=20, dim=64,
                 vocab_size=128, max_seq=8, network=None):
        super().__init__()
        self.dim = dim
        self.vocab_size = vocab_size
        self.max_seq = max_seq
        self.n_devices = n_devices
        self.network = network or NetworkSim()

        # Fixed structure: input + output layers
        self.embed = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Embedding(max_seq, dim)  # POSITION ENCODING (v1 fix)
        self.output_proj = nn.Linear(dim, vocab_size)

        # Neurons across devices
        self.neurons = nn.ModuleDict()
        self.device_map = {}
        nid = 0
        for dev in range(n_devices):
            for _ in range(neurons_per_device):
                self.neurons[str(nid)] = Neuron(nid, dev, dim)
                self.device_map[nid] = dev
                nid += 1
        self.total_neurons = nid

        # Connections (sparse initial)
        self.connections = nn.ModuleDict()
        n_initial = self.total_neurons * 3
        for _ in range(n_initial):
            src = random.randint(0, self.total_neurons - 1)
            dst = random.randint(0, self.total_neurons - 1)
            key = f"{src}_{dst}"
            if src != dst and key not in self.connections:
                self.connections[key] = Connection(src, dst, dim)

        # Routing: learned sparse selection
        self.router = nn.Linear(dim, self.total_neurons)

        print(f"  Brain v2: {self.total_neurons} neurons, {n_devices} devices, "
              f"{len(self.connections)} connections, dim={dim}", flush=True)

    def forward(self, input_ids):
        B, S = input_ids.shape

        # Embed with position
        pos = torch.arange(S, device=input_ids.device).unsqueeze(0).expand(B, -1)
        x = self.embed(input_ids) + self.pos_embed(pos)  # [B, S, dim]

        # Process each position through the network
        outputs = []
        for t in range(S):
            x_t = x[:, t, :]  # [B, dim]
            out_t = self._propagate(x_t)
            outputs.append(out_t)

        output = torch.stack(outputs, dim=1)  # [B, S, dim]
        logits = self.output_proj(output)      # [B, S, vocab]
        return logits

    def _propagate(self, x, n_hops=3):
        """Propagate signal through network with realistic latency/dropout."""
        B = x.size(0)

        # Route: pick top-k neurons to activate
        routing_scores = self.router(x)  # [B, n_neurons]
        k = max(self.total_neurons // 5, 4)
        top_k = torch.topk(routing_scores, k, dim=-1)
        active_ids = top_k.indices[0].tolist()
        active_weights = F.softmax(top_k.values, dim=-1)  # [B, k]

        # Hop 1: Input → active neurons (with network sim)
        neuron_states = {}
        for i, nid in enumerate(active_ids):
            neuron = self.neurons[str(nid)]
            src_dev = 0  # input comes from "device 0" (user's device)
            dst_dev = self.device_map[nid]

            delivered, latency = self.network.send(x, src_dev, dst_dev)
            if delivered is not None:
                state = neuron(delivered)
                neuron_states[nid] = state * active_weights[:, i:i+1]

        # Hop 2-3: Neuron → Neuron (through connections, with network sim)
        for hop in range(n_hops - 1):
            next_states = {}
            for key, conn in self.connections.items():
                src, dst = int(key.split('_')[0]), int(key.split('_')[1])
                if src in neuron_states and dst in active_ids:
                    src_dev = self.device_map[src]
                    dst_dev = self.device_map[dst]

                    delivered, latency = self.network.send(
                        neuron_states[src], src_dev, dst_dev)

                    if delivered is not None:
                        signal = conn(delivered)
                        dst_neuron = self.neurons[str(dst)]
                        activated = dst_neuron(signal)

                        if dst in next_states:
                            next_states[dst] = next_states[dst] + activated
                        else:
                            next_states[dst] = activated

            # Merge: keep best of current and new
            for nid, state in next_states.items():
                if nid in neuron_states:
                    neuron_states[nid] = neuron_states[nid] + state * 0.5
                else:
                    neuron_states[nid] = state

        # Aggregate active neurons → output
        if neuron_states:
            aggregated = torch.stack(list(neuron_states.values())).mean(dim=0)
        else:
            aggregated = torch.zeros(B, self.dim, device=x.device)

        return aggregated

    def discover_connection(self):
        """Simulate P2P discovery."""
        src = random.randint(0, self.total_neurons - 1)
        dst = random.randint(0, self.total_neurons - 1)
        key = f"{src}_{dst}"
        if src != dst and key not in self.connections:
            self.connections[key] = Connection(src, dst, self.dim)
            return True
        return False

    def prune_connections(self, min_strength=0.3, min_uses=3):
        """Remove weak/unused connections."""
        to_remove = [k for k, c in self.connections.items()
                    if c.strength < min_strength and c.use_count < min_uses]
        for k in to_remove:
            del self.connections[k]
        return len(to_remove)

    def hebbian_update(self, lr=0.01):
        """Strengthen co-active connections, decay inactive."""
        for key, conn in self.connections.items():
            src_n = self.neurons[str(conn.src)]
            dst_n = self.neurons[str(conn.dst)]
            if src_n.last_output is not None and dst_n.last_output is not None:
                co_act = (src_n.last_output.abs().mean() *
                         dst_n.last_output.abs().mean()).item()
                conn.strength = min(conn.strength + lr * co_act, 5.0)
            else:
                conn.strength *= 0.998


def run_experiment():
    print("=== SELF-TRAINING DISTRIBUTED BRAIN v2 ===\n", flush=True)

    # Network profiles to test
    profiles = {
        "ideal":    NetworkSim(base_latency_ms=0, jitter_pct=0, dropout_pct=0),
        "lan":      NetworkSim(base_latency_ms=5, jitter_pct=20, dropout_pct=2),
        "wifi":     NetworkSim(base_latency_ms=15, jitter_pct=40, dropout_pct=5),
        "hostile":  NetworkSim(base_latency_ms=50, jitter_pct=50, dropout_pct=10),
    }

    # Training data: simple patterns with POSITION dependency
    # "AB" → "CD", "12" → "34", etc
    train_pairs = []
    for i in range(0, 26, 2):
        inp = [65 + i, 65 + i + 1]        # "AB", "CD", "EF", ...
        out = [65 + i + 2, 65 + i + 3]    # "+2 offset"
        if max(out) < 128:
            train_pairs.append((inp, out))
    # Add digit patterns
    for i in range(0, 8, 2):
        inp = [48 + i, 48 + i + 1]        # "01", "23", "45", "67"
        out = [48 + i + 2, 48 + i + 3]    # "23", "45", "67", "89"
        train_pairs.append((inp, out))

    results = {}

    for profile_name, network in profiles.items():
        print(f"\n{'='*50}", flush=True)
        print(f"  Network: {profile_name} (latency={network.base_latency_ms}ms, "
              f"jitter={network.jitter_pct}%, dropout={network.dropout_pct}%)",
              flush=True)
        print(f"{'='*50}", flush=True)

        brain = DistributedBrain(
            n_devices=4, neurons_per_device=20, dim=64,
            vocab_size=128, max_seq=4, network=network)

        optimizer = torch.optim.Adam(brain.parameters(), lr=3e-3)
        n_steps = 8000
        correct_window = []

        for step in range(n_steps):
            inp, target = random.choice(train_pairs)
            input_ids = torch.tensor([inp])
            target_ids = torch.tensor([target])

            logits = brain(input_ids)
            loss = F.cross_entropy(logits.view(-1, brain.vocab_size),
                                  target_ids.view(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(brain.parameters(), 1.0)
            optimizer.step()

            brain.hebbian_update()

            if random.random() < 0.03:
                brain.discover_connection()
            if (step + 1) % 2000 == 0:
                brain.prune_connections()

            with torch.no_grad():
                pred = logits.argmax(dim=-1)
                correct = (pred == target_ids).all().item()
                correct_window.append(correct)
                if len(correct_window) > 200:
                    correct_window.pop(0)

            if (step + 1) % 1000 == 0:
                acc = sum(correct_window) / len(correct_window)
                print(f"    step {step+1}/{n_steps}  acc={acc:.3f}  "
                      f"loss={loss.item():.4f}  "
                      f"conns={len(brain.connections)}", flush=True)

        # Final eval
        brain.eval()
        correct, total = 0, 0
        for inp, target in train_pairs:
            input_ids = torch.tensor([inp])
            target_ids = torch.tensor([target])
            with torch.no_grad():
                logits = brain(input_ids)
                pred = logits.argmax(dim=-1)
                match = (pred == target_ids).all().item()
                correct += match
                total += 1

        acc = correct / total
        print(f"\n  {profile_name}: {correct}/{total} = {acc:.0%}", flush=True)
        print(f"  Network: {network.stats()}", flush=True)

        # Resilience: kill device 2
        print(f"  Resilience (kill device 2):", flush=True)
        killed = [nid for nid, dev in brain.device_map.items() if dev == 2]
        for nid in killed:
            brain.neurons[str(nid)].transform.weight.data.zero_()
            brain.neurons[str(nid)].gate.weight.data.zero_()

        correct_after = 0
        for inp, target in train_pairs:
            with torch.no_grad():
                logits = brain(torch.tensor([inp]))
                if (logits.argmax(dim=-1) == torch.tensor([target])).all().item():
                    correct_after += 1

        print(f"    Before: {correct}/{total} = {correct/total:.0%}", flush=True)
        print(f"    After:  {correct_after}/{total} = {correct_after/total:.0%}",
              flush=True)

        results[profile_name] = {
            "accuracy": acc,
            "accuracy_after_kill": correct_after / total,
            "connections": len(brain.connections),
            "network_stats": network.stats()
        }

    # Summary
    print(f"\n{'='*60}", flush=True)
    print("SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"{'Profile':<12s} {'Acc':>6s} {'Kill':>6s} {'Conns':>6s}", flush=True)
    print("-" * 35, flush=True)
    for name, r in results.items():
        print(f"{name:<12s} {r['accuracy']:>5.0%} {r['accuracy_after_kill']:>5.0%} "
              f"{r['connections']:>6d}", flush=True)

    with open("/tmp/brain_v2_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to /tmp/brain_v2_results.json", flush=True)


if __name__ == "__main__":
    run_experiment()
