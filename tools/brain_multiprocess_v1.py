#!/usr/bin/env python3
"""
Self-Training Distributed Brain — Multi-Process with Real TCP
==============================================================
Each neuron is a SEPARATE PROCESS communicating over TCP sockets.
Architecturally identical to N phones on a LAN.

No shared memory. No single-GPU matmul cheating.
Each neuron only sees signals on the wire.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import socket
import pickle
import struct
import threading
import time
import random
import json
import sys
import os
import signal
from multiprocessing import Process, Queue, Event

DIM = 32
VOCAB_SIZE = 128  # ASCII — simple for proof of concept
BASE_PORT = 9000
N_NEURONS = 16
N_DEVICES = 4  # neurons grouped into devices


# ---------------------------------------------------------------------------
# Wire protocol: length-prefixed pickle
# ---------------------------------------------------------------------------
def send_tensor(sock, tensor_data):
    """Send a tensor over TCP."""
    data = pickle.dumps(tensor_data)
    sock.sendall(struct.pack('!I', len(data)) + data)


def recv_tensor(sock):
    """Receive a tensor from TCP."""
    raw_len = _recvall(sock, 4)
    if not raw_len:
        return None
    msg_len = struct.unpack('!I', raw_len)[0]
    data = _recvall(sock, msg_len)
    if not data:
        return None
    return pickle.loads(data)


def _recvall(sock, n):
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return bytes(data)


# ---------------------------------------------------------------------------
# Neuron Process — runs independently, communicates via TCP
# ---------------------------------------------------------------------------
class NeuronProcess:
    """One neuron running as a separate process with its own TCP server."""

    def __init__(self, neuron_id, device_id, port, dim=DIM):
        self.id = neuron_id
        self.device_id = device_id
        self.port = port
        self.dim = dim

        # This neuron's weights (private — no other neuron can see these)
        self.w = torch.randn(dim, dim) * 0.1
        self.bias = torch.zeros(dim)
        self.threshold = 0.0  # ReLU threshold (learned)

        # Connections to downstream neurons (port numbers)
        self.downstream = []  # list of (neuron_id, port)

        # Hebbian state
        self.last_activation = None
        self.activation_count = 0

    def activate(self, input_signal):
        """ReLU with learned threshold — fire or don't."""
        pre = input_signal @ self.w + self.bias
        post = F.relu(pre - self.threshold)
        self.last_activation = post.detach()
        self.activation_count += 1
        return post

    def send_to_downstream(self, signal):
        """Send activation to all downstream neurons via TCP."""
        sent = 0
        for dst_id, dst_port in self.downstream:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(0.5)  # 500ms timeout — simulates real network
                sock.connect(('127.0.0.1', dst_port))
                send_tensor(sock, {
                    'type': 'activation',
                    'src': self.id,
                    'signal': signal.detach().numpy().tolist()
                })
                sock.close()
                sent += 1
            except (ConnectionRefusedError, socket.timeout, OSError):
                pass  # neuron offline — graceful degradation
        return sent

    def hebbian_update(self, lr=0.01):
        """Strengthen weights based on activation magnitude."""
        if self.last_activation is not None:
            magnitude = self.last_activation.abs().mean().item()
            self.threshold = max(self.threshold - lr * magnitude, -1.0)


def neuron_server(neuron_id, device_id, port, input_queue, output_queue,
                  stop_event, downstream_ports):
    """Run a neuron as a TCP server in a separate process."""
    neuron = NeuronProcess(neuron_id, device_id, port)
    neuron.downstream = downstream_ports

    # Start TCP server
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.settimeout(1.0)
    server.bind(('127.0.0.1', port))
    server.listen(5)

    accumulated_signal = torch.zeros(DIM)
    n_received = 0

    while not stop_event.is_set():
        # Check for incoming TCP connections
        try:
            conn, addr = server.accept()
            msg = recv_tensor(conn)
            conn.close()
            if msg and msg['type'] == 'activation':
                signal = torch.tensor(msg['signal'])
                accumulated_signal += signal
                n_received += 1
        except socket.timeout:
            pass

        # Check for input from coordinator
        try:
            cmd = input_queue.get_nowait()
            if cmd['type'] == 'forward':
                input_signal = torch.tensor(cmd['signal'])
                # Combine with accumulated signals from other neurons
                if n_received > 0:
                    combined = input_signal + accumulated_signal / n_received
                else:
                    combined = input_signal
                accumulated_signal = torch.zeros(DIM)
                n_received = 0

                # Activate
                output = neuron.activate(combined)

                # Send to downstream neurons
                neuron.send_to_downstream(output)

                # Send result back to coordinator
                output_queue.put({
                    'neuron_id': neuron_id,
                    'output': output.detach().numpy().tolist(),
                    'fired': output.abs().mean().item() > 0.01
                })

                # Hebbian update
                neuron.hebbian_update()

            elif cmd['type'] == 'update_weights':
                # Gradient update from coordinator
                grad = torch.tensor(cmd['grad'])
                neuron.w -= cmd['lr'] * grad
                output_queue.put({'neuron_id': neuron_id, 'status': 'updated'})

            elif cmd['type'] == 'stop':
                break
        except Exception:
            pass

    server.close()


# ---------------------------------------------------------------------------
# Coordinator — orchestrates forward pass across neuron processes
# ---------------------------------------------------------------------------
class BrainCoordinator:
    """Coordinates distributed neurons for training."""

    def __init__(self, n_neurons=N_NEURONS, n_devices=N_DEVICES):
        self.n_neurons = n_neurons
        self.n_devices = n_devices
        self.processes = []
        self.input_queues = []
        self.output_queues = []
        self.stop_events = []

        # Fixed input/output layers (on coordinator)
        self.embed = nn.Embedding(VOCAB_SIZE, DIM)
        self.pos_embed = nn.Embedding(16, DIM)
        self.output_proj = nn.Linear(DIM, VOCAB_SIZE)

        # Build downstream connections (random sparse graph)
        self.downstream_map = {}
        for i in range(n_neurons):
            n_conns = random.randint(1, min(4, n_neurons - 1))
            targets = random.sample([j for j in range(n_neurons) if j != i],
                                   n_conns)
            self.downstream_map[i] = [(t, BASE_PORT + t) for t in targets]

    def start(self):
        """Start all neuron processes."""
        print(f"  Starting {self.n_neurons} neuron processes...", flush=True)
        for i in range(self.n_neurons):
            device_id = i % self.n_devices
            port = BASE_PORT + i
            iq = Queue()
            oq = Queue()
            se = Event()

            p = Process(target=neuron_server,
                       args=(i, device_id, port, iq, oq, se,
                             self.downstream_map[i]))
            p.daemon = True
            p.start()

            self.processes.append(p)
            self.input_queues.append(iq)
            self.output_queues.append(oq)
            self.stop_events.append(se)

        time.sleep(0.5)  # let servers bind
        print(f"  All {self.n_neurons} neurons online.", flush=True)

    def forward(self, input_ids, active_neurons=None):
        """Distributed forward pass across neuron processes."""
        if active_neurons is None:
            active_neurons = list(range(self.n_neurons))

        B, S = input_ids.shape
        pos = torch.arange(S)
        x = self.embed(input_ids) + self.pos_embed(pos.unsqueeze(0))

        outputs = []
        for t in range(S):
            x_t = x[0, t, :].tolist()

            # Send input to all active neurons
            for nid in active_neurons:
                self.input_queues[nid].put({
                    'type': 'forward',
                    'signal': x_t
                })

            # Collect outputs (with timeout — handles dropped neurons)
            results = []
            deadline = time.time() + 0.5  # 500ms deadline
            received = set()
            while len(received) < len(active_neurons) and time.time() < deadline:
                for nid in active_neurons:
                    if nid not in received:
                        try:
                            result = self.output_queues[nid].get_nowait()
                            results.append(result)
                            received.add(result['neuron_id'])
                        except Exception:
                            pass
                if len(received) < len(active_neurons):
                    time.sleep(0.01)

            # Aggregate outputs
            if results:
                fired = [r for r in results if r['fired']]
                if fired:
                    aggregated = torch.tensor(
                        [r['output'] for r in fired]).mean(dim=0)
                else:
                    aggregated = torch.zeros(DIM)
            else:
                aggregated = torch.zeros(DIM)

            outputs.append(aggregated)

        output = torch.stack(outputs).unsqueeze(0)  # [1, S, dim]
        return self.output_proj(output)

    def kill_device(self, device_id):
        """Kill all neurons on a device."""
        killed = 0
        for i in range(self.n_neurons):
            if i % self.n_devices == device_id:
                self.stop_events[i].set()
                killed += 1
        return killed

    def stop_all(self):
        for se in self.stop_events:
            se.set()
        for p in self.processes:
            p.join(timeout=2)


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------
def run():
    print("=== MULTI-PROCESS DISTRIBUTED BRAIN ===\n", flush=True)
    print(f"  {N_NEURONS} neurons as separate processes", flush=True)
    print(f"  {N_DEVICES} simulated devices", flush=True)
    print(f"  Communication: TCP over localhost", flush=True)

    brain = BrainCoordinator(N_NEURONS, N_DEVICES)
    brain.start()

    # Training data
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

    print(f"\n  Training: {n_steps} steps, {len(train_pairs)} patterns",
          flush=True)
    t0 = time.time()

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

    # Final eval
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
            print(f"  {'OK' if match else 'MISS'} {in_s}→{tgt_s} pred:{pred_s}",
                  flush=True)

    print(f"\n  All neurons: {correct}/{total} = {correct/total:.0%}", flush=True)

    # Resilience: kill device 0
    print("\n=== RESILIENCE: Kill device 0 ===", flush=True)
    killed = brain.kill_device(0)
    print(f"  Killed {killed} neurons on device 0", flush=True)
    time.sleep(0.5)

    active = [i for i in range(N_NEURONS) if i % N_DEVICES != 0]
    correct_after = 0
    for inp, target in train_pairs:
        with torch.no_grad():
            logits = brain.forward(torch.tensor([inp]), active_neurons=active)
            pred = logits.argmax(dim=-1)
            if (pred == torch.tensor([target])).all().item():
                correct_after += 1

    print(f"  Before: {correct}/{total} = {correct/total:.0%}", flush=True)
    print(f"  After:  {correct_after}/{total} = {correct_after/total:.0%}",
          flush=True)

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.0f}s", flush=True)

    brain.stop_all()


if __name__ == "__main__":
    run()
