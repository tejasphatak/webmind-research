#!/usr/bin/env python3
"""
Self-Training Distributed Brain — UDP v2 (Optimized)
=====================================================
Fixes from v1:
- 10ms collection timeout (was 200ms) — neurons respond fast on localhost
- Non-blocking recv loop
- Batch all neurons at once, collect in one pass
- Thread-based neurons (faster spawn, still separate UDP sockets)
- Each neuron: just weights + relu. Dumb. Intelligence in connections.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import socket
import struct
import time
import random
import threading
import numpy as np

DIM = 32
VOCAB_SIZE = 128
BASE_PORT = 11000
N_NEURONS = 16
N_DEVICES = 4


def pack_signal(nid, arr):
    return struct.pack(f'!I{DIM}f', nid, *arr)

def unpack_signal(data):
    vals = struct.unpack(f'!I{DIM}f', data)
    return vals[0], list(vals[1:])


class NeuronThread(threading.Thread):
    """Dumb neuron: weights + relu. Runs as thread with own UDP socket."""

    def __init__(self, nid, dev_id, port, coord_port, downstream_ports):
        super().__init__(daemon=True)
        self.nid = nid
        self.dev = dev_id
        self.port = port
        self.coord_port = coord_port
        self.downstream = downstream_ports
        self.running = True

        # Private weights — only this neuron sees these
        self.w = np.random.randn(DIM, DIM).astype(np.float32) * 0.1
        self.bias = np.zeros(DIM, dtype=np.float32)
        self.threshold = np.float32(0.0)

    def run(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(0.05)
        sock.bind(('127.0.0.1', self.port))

        while self.running:
            try:
                data, addr = sock.recvfrom(4096)
                _, signal = unpack_signal(data)
                x = np.array(signal, dtype=np.float32)

                # THE NEURON: matmul + relu. That's it.
                out = np.maximum(x @ self.w + self.bias - self.threshold, 0)

                if np.mean(np.abs(out)) > 0.005:
                    out_data = pack_signal(self.nid, out.tolist())
                    # Forward to downstream
                    for dp in self.downstream:
                        try:
                            sock.sendto(out_data, ('127.0.0.1', dp))
                        except:
                            pass
                    # Report to coordinator
                    sock.sendto(out_data, ('127.0.0.1', self.coord_port))

                    # Hebbian: firing adjusts threshold
                    self.threshold = max(self.threshold - 0.001, -1.0)
            except socket.timeout:
                pass
            except:
                pass

        sock.close()


class FastBrain(nn.Module):
    def __init__(self, n_neurons=N_NEURONS, n_devices=N_DEVICES):
        super().__init__()
        self.n_neurons = n_neurons
        self.n_devices = n_devices
        self.coord_port = BASE_PORT + n_neurons

        self.embed = nn.Embedding(VOCAB_SIZE, DIM)
        self.pos_embed = nn.Embedding(16, DIM)
        self.output_proj = nn.Linear(DIM, VOCAB_SIZE)

        # Coordinator UDP socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(0.01)  # 10ms — fast!
        self.sock.bind(('127.0.0.1', self.coord_port))

        # Build connections
        downstream = {}
        for i in range(n_neurons):
            targets = random.sample(
                [j for j in range(n_neurons) if j != i],
                min(3, n_neurons - 1))
            downstream[i] = [BASE_PORT + t for t in targets]

        # Start neuron threads
        self.neurons = []
        for i in range(n_neurons):
            n = NeuronThread(i, i % n_devices, BASE_PORT + i,
                            self.coord_port, downstream[i])
            n.start()
            self.neurons.append(n)

        time.sleep(0.2)
        print(f"  {n_neurons} neurons online (threads + UDP)", flush=True)

    def forward(self, input_ids):
        B, S = input_ids.shape
        pos = torch.arange(S)
        x = self.embed(input_ids) + self.pos_embed(pos.unsqueeze(0))

        outputs = []
        for t in range(S):
            x_t = x[0, t, :].detach().numpy().tolist()
            data = pack_signal(999, x_t)

            # Broadcast to all neurons
            for i in range(self.n_neurons):
                if self.neurons[i].running:
                    try:
                        self.sock.sendto(data, ('127.0.0.1', BASE_PORT + i))
                    except:
                        pass

            # Collect (fast — 10ms timeout with drain loop)
            results = []
            deadline = time.time() + 0.015  # 15ms max
            while time.time() < deadline:
                try:
                    data, _ = self.sock.recvfrom(4096)
                    _, signal = unpack_signal(data)
                    results.append(torch.tensor(signal))
                except socket.timeout:
                    break
                except:
                    break

            if results:
                outputs.append(torch.stack(results).mean(dim=0))
            else:
                outputs.append(torch.zeros(DIM))

        return self.output_proj(torch.stack(outputs).unsqueeze(0))

    def kill_device(self, dev_id):
        killed = 0
        for n in self.neurons:
            if n.dev == dev_id:
                n.running = False
                killed += 1
        return killed

    def stop(self):
        for n in self.neurons:
            n.running = False
        self.sock.close()


def run():
    print("=== UDP BRAIN v2 (OPTIMIZED) ===\n", flush=True)

    brain = FastBrain(N_NEURONS, N_DEVICES)

    train_pairs = []
    for i in range(0, 26, 2):
        inp = [65 + i, 65 + i + 1]
        out = [65 + i + 2, 65 + i + 3]
        if max(out) < 128:
            train_pairs.append((inp, out))

    optimizer = torch.optim.Adam(brain.parameters(), lr=1e-3)

    n_steps = 3000
    correct_window = []
    t0 = time.time()

    print(f"  Training: {n_steps} steps, {len(train_pairs)} patterns\n",
          flush=True)

    for step in range(n_steps):
        inp, target = random.choice(train_pairs)
        input_ids = torch.tensor([inp])
        target_ids = torch.tensor([target])

        logits = brain(input_ids)
        loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), target_ids.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            pred = logits.argmax(dim=-1)
            correct = (pred == target_ids).all().item()
            correct_window.append(correct)
            if len(correct_window) > 200:
                correct_window.pop(0)

        if (step + 1) % 300 == 0:
            acc = sum(correct_window) / len(correct_window)
            elapsed = time.time() - t0
            sps = (step + 1) / elapsed
            print(f"    step {step+1}/{n_steps}  acc={acc:.3f}  "
                  f"loss={loss.item():.4f}  {sps:.1f} steps/s", flush=True)

    # Eval
    print("\n=== EVAL ===", flush=True)
    correct, total = 0, len(train_pairs)
    for inp, target in train_pairs:
        with torch.no_grad():
            logits = brain(torch.tensor([inp]))
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
    print("\n=== KILL DEVICE 0 ===", flush=True)
    killed = brain.kill_device(0)
    print(f"  Killed {killed} neurons", flush=True)
    time.sleep(0.3)

    correct_after = 0
    for inp, target in train_pairs:
        with torch.no_grad():
            logits = brain(torch.tensor([inp]))
            if (logits.argmax(dim=-1) == torch.tensor([target])).all().item():
                correct_after += 1

    print(f"  Before: {correct}/{total} = {correct/total:.0%}", flush=True)
    print(f"  After:  {correct_after}/{total} = {correct_after/total:.0%}",
          flush=True)
    print(f"\n  Time: {time.time()-t0:.0f}s, "
          f"Speed: {n_steps/(time.time()-t0):.1f} steps/s", flush=True)

    brain.stop()


if __name__ == "__main__":
    run()
