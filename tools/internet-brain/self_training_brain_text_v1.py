#!/usr/bin/env python3
"""
Self-Training Distributed Brain — Text Task on GPU
====================================================
Scale up: learn actual text patterns with a tokenizer.
Tasks:
1. Sentence completion: "The cat sat on the" → "mat"
2. Simple QA: "capital of France?" → "Paris"
3. Pattern: "one two three" → "four five six"

Neurons across 4 simulated devices. P2P discovery. Hebbian.
Network layer replaceable with real transport.
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
    def __init__(self, latency_ms=5, jitter_pct=30, dropout_pct=5):
        self.latency_ms = latency_ms
        self.jitter_pct = jitter_pct
        self.dropout_pct = dropout_pct
        self.drops = 0
        self.total = 0

    def send(self, signal, src_dev, dst_dev):
        self.total += 1
        if src_dev == dst_dev:
            return signal.detach(), 0
        if random.random() * 100 < self.dropout_pct:
            self.drops += 1
            return None, 0
        jitter = self.latency_ms * self.jitter_pct / 100 * random.uniform(-1, 1)
        return signal.detach(), max(0, self.latency_ms + jitter)


class Neuron(nn.Module):
    def __init__(self, nid, dev_id, dim):
        super().__init__()
        self.id = nid
        self.dev = dev_id
        self.fc = nn.Linear(dim, dim)
        self.gate = nn.Linear(dim, 1)
        self.count = 0
        self.last_out = None

    def forward(self, x):
        g = torch.sigmoid(self.gate(x))
        out = F.gelu(self.fc(x)) * g
        self.last_out = out.detach()
        self.count += 1
        return out


class TextBrain(nn.Module):
    """Self-organizing text brain with GPT-2 tokenizer."""

    def __init__(self, n_devices=4, neurons_per_device=50, dim=256,
                 max_seq=32, network=None):
        super().__init__()
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.vocab_size = self.tokenizer.vocab_size
        self.dim = dim
        self.max_seq = max_seq
        self.network = network or NetworkSim()

        # Fixed: input + output
        self.embed = nn.Embedding(self.vocab_size, dim)
        self.pos_embed = nn.Embedding(max_seq, dim)
        self.output_proj = nn.Linear(dim, self.vocab_size)
        self.output_norm = nn.LayerNorm(dim)

        # Neurons
        self.neurons = nn.ModuleDict()
        self.dev_map = {}
        nid = 0
        for dev in range(n_devices):
            for _ in range(neurons_per_device):
                self.neurons[str(nid)] = Neuron(nid, dev, dim)
                self.dev_map[nid] = dev
                nid += 1
        self.n_neurons = nid

        # Connections (sparse)
        self.conn_weights = nn.ParameterDict()
        self.conn_meta = {}  # (src, dst) → {strength, uses}
        for _ in range(self.n_neurons * 4):
            src = random.randint(0, self.n_neurons - 1)
            dst = random.randint(0, self.n_neurons - 1)
            key = f"{src}_{dst}"
            if src != dst and key not in self.conn_weights:
                self.conn_weights[key] = nn.Parameter(
                    torch.randn(dim, dim) * 0.02)
                self.conn_meta[key] = {"strength": 1.0, "uses": 0}

        # Router
        self.router = nn.Linear(dim, self.n_neurons)

        print(f"  TextBrain: {self.n_neurons} neurons, {n_devices} devices, "
              f"{len(self.conn_weights)} connections, dim={dim}, "
              f"vocab={self.vocab_size}", flush=True)

    def forward(self, input_ids):
        B, S = input_ids.shape
        S = min(S, self.max_seq)
        input_ids = input_ids[:, :S]

        pos = torch.arange(S, device=input_ids.device).unsqueeze(0).expand(B, -1)
        x = self.embed(input_ids) + self.pos_embed(pos)

        outputs = []
        for t in range(S):
            out = self._propagate(x[:, t, :])
            outputs.append(out)

        output = torch.stack(outputs, dim=1)
        output = self.output_norm(output)
        return self.output_proj(output)

    def _propagate(self, x, n_hops=2):
        B = x.size(0)

        # Route
        scores = self.router(x)
        k = max(self.n_neurons // 5, 8)
        top_k = torch.topk(scores, k, dim=-1)
        active = top_k.indices[0].tolist()
        weights = F.softmax(top_k.values, dim=-1)

        # Hop 1: input → neurons
        states = {}
        for i, nid in enumerate(active):
            neuron = self.neurons[str(nid)]
            delivered, _ = self.network.send(x, 0, self.dev_map[nid])
            if delivered is not None:
                states[nid] = neuron(delivered) * weights[:, i:i+1]

        # Hop 2: neuron → neuron
        next_states = {}
        for key in self.conn_weights:
            src, dst = int(key.split('_')[0]), int(key.split('_')[1])
            if src in states and dst in active:
                delivered, _ = self.network.send(
                    states[src], self.dev_map[src], self.dev_map[dst])
                if delivered is not None:
                    meta = self.conn_meta[key]
                    meta["uses"] += 1
                    signal = F.linear(delivered, self.conn_weights[key]) * meta["strength"]
                    activated = self.neurons[str(dst)](signal)
                    if dst in next_states:
                        next_states[dst] = next_states[dst] + activated
                    else:
                        next_states[dst] = activated

        for nid, s in next_states.items():
            if nid in states:
                states[nid] = states[nid] + s * 0.5
            else:
                states[nid] = s

        if states:
            return torch.stack(list(states.values())).mean(dim=0)
        return torch.zeros(B, self.dim, device=x.device)

    def hebbian_update(self, lr=0.005):
        for key in self.conn_meta:
            src, dst = int(key.split('_')[0]), int(key.split('_')[1])
            sn = self.neurons[str(src)]
            dn = self.neurons[str(dst)]
            if sn.last_out is not None and dn.last_out is not None:
                co = (sn.last_out.abs().mean() * dn.last_out.abs().mean()).item()
                self.conn_meta[key]["strength"] = min(
                    self.conn_meta[key]["strength"] + lr * co, 5.0)
            else:
                self.conn_meta[key]["strength"] *= 0.999

    def discover(self):
        src = random.randint(0, self.n_neurons - 1)
        dst = random.randint(0, self.n_neurons - 1)
        key = f"{src}_{dst}"
        if src != dst and key not in self.conn_weights:
            self.conn_weights[key] = nn.Parameter(
                torch.randn(self.dim, self.dim, device=DEVICE) * 0.02)
            self.conn_meta[key] = {"strength": 1.0, "uses": 0}


def run():
    print("=== SELF-TRAINING TEXT BRAIN ===\n", flush=True)

    # Training data: simple text patterns
    train_data = [
        # Sentence completion
        ("The cat sat on the", " mat"),
        ("The dog ran in the", " park"),
        ("The sun is in the", " sky"),
        ("I like to eat", " food"),
        ("She went to the", " store"),
        ("He reads a", " book"),
        ("They play in the", " yard"),
        ("We live in a", " house"),
        # Simple QA
        ("The capital of France is", " Paris"),
        ("The color of grass is", " green"),
        ("The color of sky is", " blue"),
        ("Water is", " wet"),
        ("Fire is", " hot"),
        ("Ice is", " cold"),
        # Number patterns
        ("one two three", " four"),
        ("two four six", " eight"),
    ]

    network = NetworkSim(latency_ms=5, jitter_pct=30, dropout_pct=3)
    brain = TextBrain(n_devices=4, neurons_per_device=50, dim=256,
                      max_seq=32, network=network).to(DEVICE)

    tokenizer = brain.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    optimizer = torch.optim.AdamW(brain.parameters(), lr=1e-3)
    n_steps = 20000

    print(f"  Training pairs: {len(train_data)}", flush=True)
    print(f"  Steps: {n_steps}", flush=True)

    losses = []
    for step in range(n_steps):
        prompt, target = random.choice(train_data)
        full = prompt + target

        inp = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=31).to(DEVICE)
        tgt = tokenizer(full, return_tensors="pt", truncation=True,
                       max_length=32).to(DEVICE)

        # Pad to same length
        max_len = tgt["input_ids"].size(1)
        if inp["input_ids"].size(1) < max_len:
            pad = torch.full((1, max_len - inp["input_ids"].size(1)),
                           tokenizer.pad_token_id, device=DEVICE)
            input_ids = torch.cat([inp["input_ids"], pad], dim=1)
        else:
            input_ids = inp["input_ids"][:, :max_len]

        target_ids = tgt["input_ids"]

        logits = brain(input_ids)

        # Loss on target tokens only
        S = min(logits.size(1), target_ids.size(1))
        loss = F.cross_entropy(
            logits[:, :S-1, :].reshape(-1, brain.vocab_size),
            target_ids[:, 1:S].reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(brain.parameters(), 1.0)
        optimizer.step()

        brain.hebbian_update()
        if random.random() < 0.02:
            brain.discover()

        losses.append(loss.item())

        if (step + 1) % 2000 == 0:
            avg = sum(losses[-2000:]) / min(len(losses), 2000)
            print(f"    step {step+1}/{n_steps}  loss={avg:.4f}  "
                  f"conns={len(brain.conn_weights)}", flush=True)

    # Final eval: generate completions
    print("\n=== FINAL EVAL ===", flush=True)
    brain.eval()
    correct = 0
    total = len(train_data)

    for prompt, target in train_data:
        inp = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=31).to(DEVICE)

        with torch.no_grad():
            logits = brain(inp["input_ids"])
            # Get prediction for next token after prompt
            next_token_logits = logits[0, -1, :]
            pred_id = next_token_logits.argmax().item()
            pred_token = tokenizer.decode(pred_id)

            # Compare with target's first token
            target_ids = tokenizer(target, return_tensors="pt")["input_ids"]
            target_first = tokenizer.decode(target_ids[0, 0])

            match = pred_token.strip() == target_first.strip()
            if match:
                correct += 1

            status = "OK" if match else "MISS"
            print(f"  [{status}] \"{prompt}\" → pred:\"{pred_token}\" "
                  f"target:\"{target_first}\"", flush=True)

    print(f"\n  Accuracy: {correct}/{total} = {correct/total:.0%}", flush=True)
    print(f"  Network: drops={network.drops}/{network.total} "
          f"({network.drops/max(network.total,1)*100:.1f}%)", flush=True)
    print(f"  Connections: {len(brain.conn_weights)}", flush=True)

    # Resilience: kill device 1
    print("\n=== RESILIENCE: Kill device 1 ===", flush=True)
    killed = [nid for nid, dev in brain.dev_map.items() if dev == 1]
    for nid in killed:
        brain.neurons[str(nid)].fc.weight.data.zero_()

    correct_after = 0
    for prompt, target in train_data:
        inp = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=31).to(DEVICE)
        with torch.no_grad():
            logits = brain(inp["input_ids"])
            pred_id = logits[0, -1, :].argmax().item()
            target_ids = tokenizer(target, return_tensors="pt")["input_ids"]
            if tokenizer.decode(pred_id).strip() == tokenizer.decode(target_ids[0, 0]).strip():
                correct_after += 1

    print(f"  Before kill: {correct}/{total} = {correct/total:.0%}", flush=True)
    print(f"  After kill:  {correct_after}/{total} = {correct_after/total:.0%}",
          flush=True)


if __name__ == "__main__":
    run()
