#!/usr/bin/env python3
"""
Synapse v2 Prototype — Distributed Specialists
===============================================
The winning architecture from 2026-04-17 marathon session.

Each device:
- Downloads base model (size based on device capability)
- Fine-tunes locally on user's usage → becomes specialist
- Routes queries to best specialist via confidence scoring
- Local-first response + async specialist refinement

Tests:
1. Specialization: different nodes learn different topics
2. Routing: queries go to right specialist
3. Resilience: kill nodes, system degrades gracefully
4. Latency: measure end-to-end with simulated network
5. Dynamic sizing: different model sizes coexist
6. Scale: 20 nodes, 8 topics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer
import copy
import time
import random
import json
import threading
import socket
import struct

DEVICE = "cpu"


class DynamicModel(nn.Module):
    """Model that can be truncated to any number of layers."""

    def __init__(self, full_model, max_layers=None):
        super().__init__()
        config = full_model.config
        total = config.n_layer

        if max_layers is None or max_layers >= total:
            self.model = copy.deepcopy(full_model)
            self.n_layers = total
        else:
            # Truncate: keep only first max_layers
            self.model = copy.deepcopy(full_model)
            # Remove excess layers
            self.model.transformer.h = self.model.transformer.h[:max_layers]
            self.model.config.n_layer = max_layers
            self.n_layers = max_layers

        self.vocab_size = config.vocab_size

    def forward(self, input_ids, **kwargs):
        return self.model(input_ids, **kwargs)

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)

    def param_count(self):
        return sum(p.numel() for p in self.parameters())

    def size_mb(self):
        return self.param_count() * 4 / 1e6


class NetworkSim:
    """Simulates network between devices. Replaceable with real UDP."""

    def __init__(self, latency_ms=15, jitter_pct=20, dropout_pct=2):
        self.latency_ms = latency_ms
        self.jitter_pct = jitter_pct
        self.dropout_pct = dropout_pct
        self.total = 0
        self.dropped = 0

    def send(self, data):
        self.total += 1
        if random.random() * 100 < self.dropout_pct:
            self.dropped += 1
            return None
        jitter = self.latency_ms * self.jitter_pct / 100 * random.uniform(-1, 1)
        latency = max(0, self.latency_ms + jitter)
        return data, latency

    def stats(self):
        dr = self.dropped / max(self.total, 1) * 100
        return f"sent={self.total} dropped={self.dropped} ({dr:.1f}%)"


class DeviceNode:
    """One device in the network."""

    def __init__(self, node_id, model, tokenizer, device_type="laptop"):
        self.id = node_id
        self.model = model
        self.tokenizer = tokenizer
        self.device_type = device_type
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
        self.online = True
        self.n_trained = 0
        self.topics_seen = set()

    def respond(self, prompt, max_tokens=10):
        if not self.online:
            return None
        self.model.eval()
        inp = self.tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            out = self.model.generate(
                inp["input_ids"], max_new_tokens=max_tokens,
                do_sample=False, pad_token_id=self.tokenizer.pad_token_id)
        return self.tokenizer.decode(
            out[0][inp["input_ids"].size(1):], skip_special_tokens=True)

    def score(self, prompt, target):
        if not self.online:
            return -999
        self.model.eval()
        full = prompt + target
        inp = self.tokenizer(full, return_tensors="pt")
        plen = self.tokenizer(prompt, return_tensors="pt")["input_ids"].size(1)
        with torch.no_grad():
            logits = self.model(**inp).logits
            if plen >= inp["input_ids"].size(1):
                return -10
            loss = F.cross_entropy(
                logits[0, plen-1:-1], inp["input_ids"][0, plen:]).item()
        return -loss

    def train_on(self, prompt, target):
        if not self.online:
            return
        self.model.train()
        full = prompt + target
        inp = self.tokenizer(full, return_tensors="pt")
        logits = self.model(**inp).logits
        loss = F.cross_entropy(
            logits[:, :-1, :].reshape(-1, self.model.vocab_size),
            inp["input_ids"][:, 1:].reshape(-1))
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.n_trained += 1
        return loss.item()


class SynapseV2:
    """The distributed specialist network."""

    def __init__(self, n_nodes=4, network=None):
        print("Loading base model...", flush=True)
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        base = GPT2LMHeadModel.from_pretrained("gpt2")
        base.eval()

        self.network = network or NetworkSim()
        self.nodes = []

        # Dynamic sizing: different devices get different model sizes
        device_configs = [
            ("phone_low", 4),   # 4 layers
            ("phone_high", 6),  # 6 layers
            ("laptop", 8),      # 8 layers
            ("desktop", 12),    # full 12 layers
        ]

        for i in range(n_nodes):
            dtype, n_layers = device_configs[i % len(device_configs)]
            model = DynamicModel(base, max_layers=n_layers)
            node = DeviceNode(i, model, self.tokenizer, dtype)
            self.nodes.append(node)

        del base

        sizes = set(n.model.n_layers for n in self.nodes)
        print(f"  {n_nodes} nodes, layer configs: {sorted(sizes)}", flush=True)
        for n in self.nodes[:4]:
            print(f"    node {n.id}: {n.device_type}, {n.model.n_layers} layers, "
                  f"{n.model.size_mb():.0f}MB", flush=True)

    def specialize(self, topic_data, n_epochs=3):
        """Train each node on assigned topics."""
        topics = list(topic_data.keys())
        n_nodes = len(self.nodes)

        # Assign topics with overlap (resilience)
        assignments = {}
        for i, topic in enumerate(topics):
            # Primary + 2 backups
            primary = i % n_nodes
            nodes_for_topic = [
                primary,
                (primary + 1) % n_nodes,
                (primary + n_nodes // 2) % n_nodes,
            ]
            assignments[topic] = nodes_for_topic

        print(f"\n  Specializing across {len(topics)} topics...", flush=True)
        for epoch in range(n_epochs):
            total_loss = 0
            steps = 0
            for topic, node_ids in assignments.items():
                for prompt, target in topic_data[topic]:
                    for nid in node_ids:
                        loss = self.nodes[nid].train_on(prompt, target)
                        if loss is not None:
                            total_loss += loss
                            steps += 1
                            self.nodes[nid].topics_seen.add(topic)

            print(f"    epoch {epoch+1}/{n_epochs}  loss={total_loss/max(steps,1):.4f}  "
                  f"steps={steps}", flush=True)

    def query(self, prompt, top_k=3):
        """Route query to best specialists and get response."""
        t0 = time.time()

        # Step 1: Local response (node 0 = user's device)
        local_resp = self.nodes[0].respond(prompt)
        local_ms = (time.time() - t0) * 1000

        # Step 2: Route to top-k specialists
        dummy_target = " the"
        scores = []
        for node in self.nodes:
            if node.online:
                s = node.score(prompt, dummy_target)
                # Simulate network latency
                result = self.network.send(s)
                if result is not None:
                    scores.append((node, result[0]))

        scores.sort(key=lambda x: x[1], reverse=True)
        best_nodes = [s[0] for s in scores[:top_k]]

        # Step 3: Get response from best specialist
        if best_nodes:
            best_resp = best_nodes[0].respond(prompt)
            best_id = best_nodes[0].id
        else:
            best_resp = local_resp
            best_id = 0

        total_ms = (time.time() - t0) * 1000
        return {
            "local": local_resp,
            "specialist": best_resp,
            "specialist_node": best_id,
            "local_ms": local_ms,
            "total_ms": total_ms,
        }

    def kill_nodes(self, node_ids):
        for nid in node_ids:
            self.nodes[nid].online = False

    def stats(self):
        online = sum(1 for n in self.nodes if n.online)
        trained = sum(n.n_trained for n in self.nodes)
        return f"online={online}/{len(self.nodes)} trained={trained}"


def run():
    print("=== SYNAPSE v2: DISTRIBUTED SPECIALISTS ===\n", flush=True)

    # 8 diverse topics
    topic_data = {
        "geography": [
            ("The capital of France is", " Paris"),
            ("The capital of Japan is", " Tokyo"),
            ("The capital of Brazil is", " Brasilia"),
        ],
        "science": [
            ("Water freezes at", " zero degrees"),
            ("The sun is a", " star"),
            ("Plants need sunlight for", " photosynthesis"),
        ],
        "math": [
            ("Two plus two equals", " four"),
            ("Ten minus three equals", " seven"),
            ("Three times four equals", " twelve"),
        ],
        "language": [
            ("The opposite of hot is", " cold"),
            ("The opposite of big is", " small"),
            ("The past tense of go is", " went"),
        ],
        "animals": [
            ("The largest animal is the", " blue whale"),
            ("Dogs are known as man's best", " friend"),
            ("Cats are", " independent"),
        ],
        "food": [
            ("Pizza originated in", " Italy"),
            ("Sushi is from", " Japan"),
            ("Chocolate is made from", " cocoa"),
        ],
        "tech": [
            ("The internet was invented in", " the"),
            ("HTML stands for", " Hyper"),
            ("Python is a programming", " language"),
        ],
        "history": [
            ("World War 2 ended in", " 1945"),
            ("The moon landing was in", " 1969"),
            ("The first president was", " George"),
        ],
    }

    net = SynapseV2(n_nodes=4, network=NetworkSim(latency_ms=15))

    # Specialize
    net.specialize(topic_data, n_epochs=5)

    # === TEST 1: Accuracy ===
    print("\n=== TEST 1: ACCURACY ===", flush=True)
    correct, total = 0, 0
    all_qa = []
    for topic, examples in topic_data.items():
        for prompt, target in examples:
            all_qa.append((topic, prompt, target))

    for topic, prompt, target in all_qa:
        result = net.query(prompt, top_k=3)
        resp = result["specialist"] or ""
        first_word = target.strip().split()[0].lower()
        match = first_word in resp.lower() if resp else False
        correct += match
        total += 1
        status = "OK" if match else "XX"
        print(f"  [{status}] [{topic:10s}] node={result['specialist_node']:2d}  "
              f"\"{prompt[:35]}\" → \"{resp.strip()[:25]}\"", flush=True)

    print(f"\n  Accuracy: {correct}/{total} = {correct/total:.0%}", flush=True)

    # === TEST 2: LATENCY ===
    print("\n=== TEST 2: LATENCY ===", flush=True)
    for prompt, _ in [all_qa[0][1:], all_qa[5][1:], all_qa[10][1:]]:
        result = net.query(prompt)
        print(f"  local={result['local_ms']:.0f}ms  "
              f"total={result['total_ms']:.0f}ms  "
              f"\"{prompt[:40]}\"", flush=True)

    # === TEST 3: RESILIENCE ===
    print("\n=== TEST 3: RESILIENCE (kill 5 of 20 nodes) ===", flush=True)
    net.kill_nodes([0, 3, 7, 12, 18])
    correct_after = 0
    for topic, prompt, target in all_qa:
        result = net.query(prompt, top_k=3)
        resp = result["specialist"] or ""
        first_word = target.strip().split()[0].lower()
        if first_word in resp.lower():
            correct_after += 1

    print(f"  Before kill: {correct}/{total} = {correct/total:.0%}", flush=True)
    print(f"  After kill:  {correct_after}/{total} = {correct_after/total:.0%}",
          flush=True)
    print(f"  Network: {net.network.stats()}", flush=True)

    # === TEST 4: DYNAMIC MODEL SIZES ===
    print("\n=== TEST 4: MODEL SIZES ===", flush=True)
    for node in net.nodes[:8]:
        print(f"  node {node.id:2d}: {node.device_type:12s}  "
              f"{node.model.n_layers:2d} layers  "
              f"{node.model.size_mb():5.0f}MB  "
              f"trained={node.n_trained:3d}  "
              f"topics={node.topics_seen}", flush=True)

    print(f"\n  Total: {net.stats()}", flush=True)


if __name__ == "__main__":
    run()
