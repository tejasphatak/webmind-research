#!/usr/bin/env python3
"""
Self-Training Distributed Brain — UDP Multi-Process
=====================================================
Each neuron = separate process with UDP socket.
Fire-and-forget signals. Lost packets = natural dropout.
Fast. Honest. Deployable on ESP32.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import socket
import struct
import time
import random
import json
import sys
import os
from multiprocessing import Process, Queue, Event
import numpy as np

DIM = 32
VOCAB_SIZE = 128
BASE_PORT = 10000
N_NEURONS = 16
N_DEVICES = 4


def pack_signal(neuron_id, signal_array):
    """Pack neuron ID + float array into bytes. ESP32-compatible."""
    return struct.pack(f'!I{DIM}f', neuron_id, *signal_array)


def unpack_signal(data):
    """Unpack bytes to neuron ID + float array."""
    vals = struct.unpack(f'!I{DIM}f', data)
    return vals[0], list(vals[1:])


# ---------------------------------------------------------------------------
# Neuron Process — UDP
# ---------------------------------------------------------------------------
def neuron_process(neuron_id, device_id, port, coord_port, downstream_ports,
                   stop_event):
    """One neuron: listens on UDP, activates, forwards to downstream."""

    # Private weights
    w = np.random.randn(DIM, DIM).astype(np.float32) * 0.1
    bias = np.zeros(DIM, dtype=np.float32)
    threshold = 0.0

    # UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(0.1)
    sock.bind(('127.0.0.1', port))

    coord_addr = ('127.0.0.1', coord_port)
    activation_count = 0

    while not stop_event.is_set():
        try:
            data, addr = sock.recvfrom(4096)
            src_id, signal = unpack_signal(data)

            # Activate: ReLU with threshold
            x = np.array(signal, dtype=np.float32)
            pre = x @ w + bias
            post = np.maximum(pre - threshold, 0)  # ReLU

            fired = np.mean(np.abs(post)) > 0.01
            activation_count += 1

            if fired:
                # Forward to downstream neurons (fire and forget — UDP)
                out_data = pack_signal(neuron_id, post.tolist())
                for dst_port in downstream_ports:
                    try:
                        sock.sendto(out_data, ('127.0.0.1', dst_port))
                    except OSError:
                        pass

                # Send result to coordinator
                sock.sendto(out_data, coord_addr)

                # Hebbian: adjust threshold based on firing
                threshold = max(threshold - 0.001 * np.mean(np.abs(post)), -1.0)

        except socket.timeout:
            pass
        except Exception:
            pass

    sock.close()


# ---------------------------------------------------------------------------
# Coordinator
# ---------------------------------------------------------------------------
class UDPBrainCoordinator:
    def __init__(self, n_neurons=N_NEURONS, n_devices=N_DEVICES):
        self.n_neurons = n_neurons
        self.n_devices = n_devices
        self.coord_port = BASE_PORT + n_neurons  # coordinator's port
        self.processes = []
        self.stop_events = []

        # Fixed layers
        self.embed = nn.Embedding(VOCAB_SIZE, DIM)
        self.pos_embed = nn.Embedding(16, DIM)
        self.output_proj = nn.Linear(DIM, VOCAB_SIZE)

        # UDP socket for coordinator
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(0.3)  # 300ms collection window
        self.sock.bind(('127.0.0.1', self.coord_port))

        # Build downstream connections
        self.downstream = {}
        for i in range(n_neurons):
            n_conns = random.randint(1, min(3, n_neurons - 1))
            targets = random.sample([j for j in range(n_neurons) if j != i],
                                   n_conns)
            self.downstream[i] = [BASE_PORT + t for t in targets]

    def start(self):
        print(f"  Starting {self.n_neurons} neuron processes (UDP)...",
              flush=True)
        for i in range(self.n_neurons):
            se = Event()
            p = Process(target=neuron_process,
                       args=(i, i % self.n_devices, BASE_PORT + i,
                             self.coord_port, self.downstream[i], se))
            p.daemon = True
            p.start()
            self.processes.append(p)
            self.stop_events.append(se)

        time.sleep(0.3)
        print(f"  All {self.n_neurons} neurons online (UDP).", flush=True)

    def forward(self, input_ids, active_neurons=None):
        if active_neurons is None:
            active_neurons = list(range(self.n_neurons))

        B, S = input_ids.shape
        pos = torch.arange(S)
        x = self.embed(input_ids) + self.pos_embed(pos.unsqueeze(0))

        outputs = []
        for t in range(S):
            x_t = x[0, t, :].detach().numpy().tolist()

            # Broadcast input to all active neurons (UDP — fire and forget)
            data = pack_signal(999, x_t)  # 999 = coordinator
            for nid in active_neurons:
                try:
                    self.sock.sendto(data, ('127.0.0.1', BASE_PORT + nid))
                except OSError:
                    pass

            # Collect responses (with timeout)
            results = []
            deadline = time.time() + 0.2  # 200ms
            while time.time() < deadline:
                try:
                    data, addr = self.sock.recvfrom(4096)
                    nid, signal = unpack_signal(data)
                    results.append(torch.tensor(signal))
                except socket.timeout:
                    break
                except Exception:
                    break

            if results:
                aggregated = torch.stack(results).mean(dim=0)
            else:
                aggregated = torch.zeros(DIM)

            outputs.append(aggregated)

        output = torch.stack(outputs).unsqueeze(0)
        return self.output_proj(output)

    def kill_device(self, device_id):
        killed = 0
        for i in range(self.n_neurons):
            if i % self.n_devices == device_id:
                self.stop_events[i].set()
                killed += 1
        return killed

    def stop_all(self):
        for se in self.stop_events:
            se.set()
        time.sleep(0.5)
        self.sock.close()


def run():
    print("=== UDP MULTI-PROCESS DISTRIBUTED BRAIN ===\n", flush=True)

    brain = UDPBrainCoordinator(N_NEURONS, N_DEVICES)
    brain.start()

    train_pairs = []
    for i in range(0, 26, 2):
        inp = [65 + i, 65 + i + 1]
        out = [65 + i + 2, 65 + i + 3]
        if max(out) < 128:
            train_pairs.append((inp, out))

    optimizer = torch.optim.Adam(
        list(brain.embed.parameters()) +
        list(brain.pos_embed.parameters()) +
        list(brain.output_proj.parameters()),
        lr=1e-3)

    n_steps = 2000
    correct_window = []
    t0 = time.time()

    print(f"  Training: {n_steps} steps, {len(train_pairs)} patterns\n",
          flush=True)

    for step in range(n_steps):
        inp, target = random.choice(train_pairs)
        input_ids = torch.tensor([inp])
        target_ids = torch.tensor([target])

        logits = brain.forward(input_ids)
        loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE),
                              target_ids.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            pred = logits.argmax(dim=-1)
            correct = (pred == target_ids).all().item()
            correct_window.append(correct)
            if len(correct_window) > 100:
                correct_window.pop(0)

        if (step + 1) % 200 == 0:
            acc = sum(correct_window) / len(correct_window)
            elapsed = time.time() - t0
            sps = (step + 1) / elapsed
            print(f"    step {step+1}/{n_steps}  acc={acc:.3f}  "
                  f"loss={loss.item():.4f}  {sps:.1f} steps/s", flush=True)

    # Eval
    print("\n=== EVAL: All neurons ===", flush=True)
    correct, total = 0, len(train_pairs)
    for inp, target in train_pairs:
        with torch.no_grad():
            logits = brain.forward(torch.tensor([inp]))
            pred = logits.argmax(dim=-1)
            match = (pred == torch.tensor([target])).all().item()
            correct += match
            in_s = ''.join(chr(c) for c in inp)
            tgt_s = ''.join(chr(c) for c in target)
            pred_s = ''.join(chr(c.item()) for c in pred[0])
            print(f"  {'OK' if match else 'XX'} {in_s}→{tgt_s} pred:{pred_s}",
                  flush=True)

    print(f"\n  Accuracy: {correct}/{total} = {correct/total:.0%}", flush=True)

    # Resilience
    print("\n=== RESILIENCE: Kill device 0 ===", flush=True)
    killed = brain.kill_device(0)
    print(f"  Killed {killed} neurons", flush=True)
    time.sleep(0.5)

    active = [i for i in range(N_NEURONS) if i % N_DEVICES != 0]
    correct_after = 0
    for inp, target in train_pairs:
        with torch.no_grad():
            logits = brain.forward(torch.tensor([inp]), active_neurons=active)
            if (logits.argmax(dim=-1) == torch.tensor([target])).all().item():
                correct_after += 1

    print(f"  Before: {correct}/{total} = {correct/total:.0%}", flush=True)
    print(f"  After:  {correct_after}/{total} = {correct_after/total:.0%}",
          flush=True)
    print(f"\n  Total time: {time.time()-t0:.0f}s", flush=True)

    brain.stop_all()


if __name__ == "__main__":
    run()
