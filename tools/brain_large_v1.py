#!/usr/bin/env python3
"""
Large Distributed Brain — 200 neurons, 8 devices
=================================================
Scale up from 20 neurons (didn't learn) to 200 neurons.
Brain v2 proved 80 neurons with torch autograd = 100% accuracy.
This version: 200 neurons across 8 devices, biological learning
for neurons + gradient for coordinator. 50K steps.

Key change: neurons use threads with SHARED NOTHING (each has own weights
in thread-local storage). Communication via thread-safe queues
(simulating UDP but faster for prototyping).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import threading
import queue
import time
import random
import json

DIM = 64  # bigger dim for more capacity
VOCAB_SIZE = 128
N_DEVICES = 8
NEURONS_PER_DEVICE = 25  # 200 total
N_STEPS = 30000
DEVICE = "cpu"


class NeuronWorker:
    """One neuron. Thread-local weights. No shared memory."""

    def __init__(self, nid, device_id, dim=DIM):
        self.id = nid
        self.device_id = device_id
        self.dim = dim

        # Private weights
        self.w = np.random.randn(dim, dim).astype(np.float32) * 0.05
        self.bias = np.zeros(dim, dtype=np.float32)
        self.threshold = np.float32(0.0)

        # State
        self.last_input = None
        self.last_output = None
        self.fired = False
        self.fire_count = 0
        self.rest_count = 0

        # Downstream connections
        self.downstream = []

    def activate(self, x):
        self.last_input = x.copy()
        pre = x @ self.w + self.bias
        post = np.maximum(pre - self.threshold, 0)
        self.fired = np.mean(np.abs(post)) > 0.005

        if self.fired:
            self.last_output = post.copy()
            self.fire_count += 1
            self.rest_count = 0
        else:
            self.last_output = np.zeros(self.dim, dtype=np.float32)
            self.rest_count += 1

        # Homeostasis
        total = self.fire_count + self.rest_count
        if total > 10:
            rate = self.fire_count / total
            if rate > 0.6:
                self.threshold += 0.005
            elif rate < 0.15:
                self.threshold -= 0.005
            self.threshold = np.clip(self.threshold, -2.0, 2.0)

        return post

    def hebbian_update(self, dopamine=0.0, lr=0.02):
        if self.last_input is None or not self.fired:
            return
        activity = np.outer(self.last_input, self.last_output)
        modulation = 1.0 + dopamine
        self.w += lr * modulation * activity.T
        self.bias += lr * modulation * self.last_output * 0.05
        self.w *= 0.9995  # decay
        self.w = np.clip(self.w, -3.0, 3.0)
        self.bias = np.clip(self.bias, -2.0, 2.0)


class DeviceSim:
    """Simulates one device with N neurons. Uses queues for inter-device comm."""

    def __init__(self, dev_id, neurons, all_queues, neuron_device_map):
        self.id = dev_id
        self.neurons = {n.id: n for n in neurons}
        self.all_queues = all_queues  # {dev_id → queue}
        self.neuron_map = neuron_device_map
        self.inbox = {}
        self.local_count = 0
        self.remote_count = 0

    def process(self, input_signal, dropout_rate=0.0):
        results = []
        for nid, neuron in self.neurons.items():
            # Random dropout (simulates device instability)
            if random.random() < dropout_rate:
                continue

            combined = input_signal.copy()
            if nid in self.inbox and self.inbox[nid]:
                for sig in self.inbox[nid]:
                    combined += sig
                combined /= (len(self.inbox[nid]) + 1)
                self.inbox[nid] = []

            output = neuron.activate(combined)
            if neuron.fired:
                results.append((nid, output))
                for dst in neuron.downstream:
                    self._route(nid, dst, output)

        return results

    def _route(self, src, dst, signal):
        dst_dev = self.neuron_map.get(dst)
        if dst_dev == self.id:
            if dst not in self.inbox:
                self.inbox[dst] = []
            self.inbox[dst].append(signal.copy())
            self.local_count += 1
        elif dst_dev is not None and dst_dev in self.all_queues:
            try:
                self.all_queues[dst_dev].put_nowait((src, dst, signal.copy()))
                self.remote_count += 1
            except queue.Full:
                pass

    def drain_queue(self):
        """Pull incoming messages from queue."""
        q = self.all_queues.get(self.id)
        if q:
            while True:
                try:
                    src, dst, signal = q.get_nowait()
                    if dst in self.neurons:
                        if dst not in self.inbox:
                            self.inbox[dst] = []
                        self.inbox[dst].append(signal)
                except queue.Empty:
                    break

    def send_dopamine(self, dopamine):
        for neuron in self.neurons.values():
            neuron.hebbian_update(dopamine)


class LargeBrain(nn.Module):
    """200-neuron brain across 8 devices with torch coordinator."""

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, DIM)
        self.pos_embed = nn.Embedding(16, DIM)
        self.out_proj = nn.Linear(DIM, VOCAB_SIZE)
        self.out_norm = nn.LayerNorm(DIM)

        # Build neurons and devices
        neuron_map = {}
        all_queues = {d: queue.Queue(maxsize=10000) for d in range(N_DEVICES)}
        self.devices = {}
        nid = 0

        for dev_id in range(N_DEVICES):
            neurons = []
            for _ in range(NEURONS_PER_DEVICE):
                n = NeuronWorker(nid, dev_id, DIM)
                neurons.append(n)
                neuron_map[nid] = dev_id
                nid += 1
            self.devices[dev_id] = DeviceSim(dev_id, neurons, all_queues, neuron_map)

        self.total = nid

        # Wire connections (random sparse)
        all_nids = list(range(self.total))
        for n_id in all_nids:
            dev = self.devices[neuron_map[n_id]]
            neuron = dev.neurons[n_id]
            targets = random.sample([j for j in all_nids if j != n_id],
                                   min(5, len(all_nids) - 1))
            neuron.downstream = targets

        print(f"  LargeBrain: {self.total} neurons, {N_DEVICES} devices, "
              f"dim={DIM}", flush=True)

    def forward(self, input_ids):
        B, S = input_ids.shape
        pos = torch.arange(S)
        x = self.embed(input_ids) + self.pos_embed(pos.unsqueeze(0))

        outputs = []
        for t in range(S):
            x_t = x[0, t, :].detach().numpy()

            # Drain inter-device queues
            for dev in self.devices.values():
                dev.drain_queue()

            # Hop 1: broadcast to all devices
            all_results = []
            for dev in self.devices.values():
                results = dev.process(x_t, dropout_rate=0.02)
                all_results.extend(results)

            # Drain + Hop 2
            for dev in self.devices.values():
                dev.drain_queue()
            for dev in self.devices.values():
                results = dev.process(np.zeros(DIM, dtype=np.float32))
                all_results.extend(results)

            if all_results:
                agg = np.mean([r[1] for r in all_results], axis=0)
                outputs.append(torch.tensor(agg, dtype=torch.float32))
            else:
                outputs.append(torch.zeros(DIM))

        output = torch.stack(outputs).unsqueeze(0)  # [1, S, DIM]
        output = self.out_norm(output)
        return self.out_proj(output)

    def send_dopamine(self, dopamine):
        for dev in self.devices.values():
            dev.send_dopamine(dopamine)


def run():
    print("=== LARGE DISTRIBUTED BRAIN (200 neurons) ===\n", flush=True)

    brain = LargeBrain()
    optimizer = torch.optim.Adam(brain.parameters(), lr=1e-3)

    train_pairs = []
    for i in range(0, 26, 2):
        inp = [65 + i, 65 + i + 1]
        out = [65 + i + 2, 65 + i + 3]
        if max(out) < 128:
            train_pairs.append((inp, out))

    print(f"  Training: {N_STEPS} steps, {len(train_pairs)} patterns\n",
          flush=True)

    t0 = time.time()
    correct_window = []
    loss_history = []

    for step in range(N_STEPS):
        inp, target = random.choice(train_pairs)
        input_ids = torch.tensor([inp])
        target_ids = torch.tensor([target])

        logits = brain(input_ids)
        loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), target_ids.view(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(brain.parameters(), 1.0)
        optimizer.step()

        # Natural dopamine
        loss_val = loss.item()
        loss_history.append(loss_val)
        if len(loss_history) > 100:
            loss_history.pop(0)
        avg_loss = np.mean(loss_history)
        dopamine = (avg_loss - loss_val) * 3.0

        brain.send_dopamine(dopamine)

        with torch.no_grad():
            pred = logits.argmax(dim=-1)
            correct = (pred == target_ids).all().item()
            correct_window.append(correct)
            if len(correct_window) > 500:
                correct_window.pop(0)

        if (step + 1) % 3000 == 0:
            acc = sum(correct_window) / len(correct_window)
            elapsed = time.time() - t0
            sps = (step + 1) / elapsed
            print(f"    step {step+1}/{N_STEPS}  acc={acc:.3f}  "
                  f"loss={loss_val:.4f}  {sps:.1f} steps/s", flush=True)

    # Eval
    print("\n=== EVAL ===", flush=True)
    brain.eval()
    correct, total = 0, len(train_pairs)
    for inp, target in train_pairs:
        with torch.no_grad():
            logits = brain(torch.tensor([inp]))
            pred = logits.argmax(dim=-1)
            match = (pred == torch.tensor([target])).all().item()
            correct += match
            in_s = ''.join(chr(c) for c in inp)
            tgt_s = ''.join(chr(c) for c in target)
            pred_s = ''.join(chr(c.item()) for c in pred[0]
                           if 32 <= c.item() < 128)
            print(f"  {'OK' if match else 'XX'} {in_s}→{tgt_s} pred:{pred_s}",
                  flush=True)

    print(f"\n  Accuracy: {correct}/{total} = {correct/total:.0%}", flush=True)

    # Stats
    for dev_id, dev in brain.devices.items():
        n_fired = sum(1 for n in dev.neurons.values() if n.fire_count > 0)
        print(f"  Dev {dev_id}: local={dev.local_count} "
              f"remote={dev.remote_count} fired={n_fired}/{NEURONS_PER_DEVICE}",
              flush=True)

    print(f"\n  Time: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    run()
