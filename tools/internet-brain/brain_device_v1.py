#!/usr/bin/env python3
"""
Self-Training Distributed Brain — Device Architecture
======================================================
Proper architecture:
- Device = one process (phone/laptop/ESP32), holds N neurons
- Neurons on same device: in-memory (free, fast)
- Neurons on different devices: UDP (real network)
- Each neuron has its own peer table (DHT-style)
- Peer discovery via gossip
- Neuron = just weights + relu. Dumb. numpy only.
"""

import numpy as np
import socket
import struct
import threading
import time
import random
import json
import sys

DIM = 32
VOCAB_SIZE = 128
BASE_PORT = 12000


def pack_msg(msg_type, src_neuron, data_array):
    """Wire format: type(1B) + src_neuron(2B) + floats.
    Fits in one UDP packet. ESP32 compatible."""
    return struct.pack(f'!BH{len(data_array)}f', msg_type, src_neuron, *data_array)


def unpack_msg(raw):
    n_floats = (len(raw) - 3) // 4
    vals = struct.unpack(f'!BH{n_floats}f', raw)
    return vals[0], vals[1], list(vals[2:])


MSG_ACTIVATION = 1
MSG_DISCOVER = 2
MSG_GOSSIP = 3


class Neuron:
    """Dumb neuron. Just weights + relu. numpy only. No torch."""

    def __init__(self, nid, dim=DIM):
        self.id = nid
        self.w = np.random.randn(dim, dim).astype(np.float32) * 0.05
        self.bias = np.zeros(dim, dtype=np.float32)
        self.threshold = np.float32(0.0)
        self.last_out = None
        self.fire_count = 0

        # Peer table: {neuron_id → device_id}
        # This neuron knows which device holds which peer
        self.peers = {}

        # Downstream: which neurons to forward to
        self.downstream = []  # list of neuron_ids

    def activate(self, x):
        """Input → matmul → relu → output. That's it."""
        self.last_input = x.copy()  # save for local learning
        pre = x @ self.w + self.bias
        post = np.maximum(pre - self.threshold, 0)  # ReLU with threshold
        fired = np.mean(np.abs(post)) > 0.005
        if fired:
            self.last_out = post.copy()
            self.fire_count += 1
            self.threshold = max(self.threshold - 0.001, -1.0)
        else:
            self.last_out = None
        return post, fired

    def local_update(self, reward, lr=0.005):
        """REINFORCE: binary reward signal. Simple but noisy."""
        if self.last_out is not None and hasattr(self, 'last_input'):
            activity = np.outer(self.last_input, self.last_out)
            self.w += lr * reward * activity.T
            self.bias += lr * reward * self.last_out
            self.w = np.clip(self.w, -2.0, 2.0)
            self.bias = np.clip(self.bias, -1.0, 1.0)

    def feedback_update(self, error_signal, lr=0.01):
        """Feedback alignment: coordinator sends error gradient.
        Neuron computes local update using random feedback weights.
        Biologically plausible. Works over the wire.

        error_signal: [DIM] — how wrong was the network's output
        The neuron uses its own random feedback matrix (fixed, not learned)
        to convert this global error into a local weight update."""
        if self.last_out is not None and hasattr(self, 'last_input'):
            if not hasattr(self, 'feedback_w'):
                # Random feedback matrix — fixed, never updated
                # This is the key insight of feedback alignment:
                # random feedback works almost as well as true backprop
                self.feedback_w = np.random.randn(DIM, DIM).astype(np.float32) * 0.1

            # Local error = error_signal projected through random feedback
            local_error = error_signal @ self.feedback_w

            # ReLU derivative: only update where neuron fired
            mask = (self.last_out > 0).astype(np.float32)
            local_error *= mask

            # Weight update: input × local_error
            self.w -= lr * np.outer(local_error, self.last_input)
            self.bias -= lr * local_error
            self.w = np.clip(self.w, -2.0, 2.0)
            self.bias = np.clip(self.bias, -1.0, 1.0)


class Device(threading.Thread):
    """One device (phone/laptop/ESP32). Holds N neurons.
    Has one UDP socket. Routes signals between local and remote neurons."""

    def __init__(self, device_id, port, neurons, all_device_ports):
        super().__init__(daemon=True)
        self.id = device_id
        self.port = port
        self.neurons = {n.id: n for n in neurons}
        self.all_ports = all_device_ports  # {device_id → port}
        self.running = True

        # Incoming signal buffer
        self.inbox = {}  # neuron_id → list of signals

        # Stats
        self.local_sends = 0
        self.remote_sends = 0

    def run(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(0.02)
        sock.bind(('127.0.0.1', self.port))
        self.sock = sock

        while self.running:
            # Receive incoming signals
            try:
                data, addr = sock.recvfrom(4096)
                msg_type, src_nid, signal = unpack_msg(data)

                if msg_type == MSG_ACTIVATION:
                    # Route to target neurons on this device
                    for nid, neuron in self.neurons.items():
                        if src_nid in [p for p in neuron.downstream]:
                            continue  # wrong direction
                        # This neuron receives the signal
                        if nid not in self.inbox:
                            self.inbox[nid] = []
                        self.inbox[nid].append(np.array(signal, dtype=np.float32))

            except socket.timeout:
                pass
            except:
                pass

        sock.close()

    def process_input(self, input_signal, target_neurons=None):
        """Process an input signal through this device's neurons.
        Returns list of (neuron_id, output) for neurons that fired."""
        if target_neurons is None:
            target_neurons = list(self.neurons.keys())

        results = []
        for nid in target_neurons:
            if nid not in self.neurons:
                continue
            neuron = self.neurons[nid]

            # Combine input with any accumulated inbox signals
            combined = input_signal.copy()
            if nid in self.inbox and self.inbox[nid]:
                for sig in self.inbox[nid]:
                    combined = combined + sig
                self.inbox[nid] = []

            # Activate
            out, fired = neuron.activate(combined)

            if fired:
                results.append((nid, out))

                # Forward to downstream neurons
                for dst_nid in neuron.downstream:
                    self._send_to_neuron(nid, dst_nid, out)

        return results

    def _send_to_neuron(self, src_nid, dst_nid, signal):
        """Send signal to a neuron. Local = in-memory. Remote = UDP."""
        if dst_nid in self.neurons:
            # LOCAL: same device, in-memory
            if dst_nid not in self.inbox:
                self.inbox[dst_nid] = []
            self.inbox[dst_nid].append(signal.copy())
            self.local_sends += 1
        else:
            # REMOTE: find which device has this neuron, send UDP
            src_neuron = self.neurons[src_nid]
            if dst_nid in src_neuron.peers:
                dst_dev = src_neuron.peers[dst_nid]
                if dst_dev in self.all_ports:
                    data = pack_msg(MSG_ACTIVATION, src_nid, signal.tolist())
                    try:
                        self.sock.sendto(data,
                            ('127.0.0.1', self.all_ports[dst_dev]))
                        self.remote_sends += 1
                    except:
                        pass  # device offline — graceful degradation

    def broadcast_reward(self, reward):
        """Send reward signal to all local neurons (REINFORCE)."""
        for neuron in self.neurons.values():
            neuron.local_update(reward)

    def broadcast_error(self, error_signal):
        """Send error gradient to all local neurons (feedback alignment).
        error_signal: [DIM] numpy array — the loss gradient."""
        for neuron in self.neurons.values():
            neuron.feedback_update(error_signal)

    def kill(self):
        self.running = False


class DistributedBrain:
    """The full brain: multiple devices, each with neurons, connected via UDP."""

    def __init__(self, n_devices=4, neurons_per_device=4):
        self.n_devices = n_devices
        self.npd = neurons_per_device
        self.devices = {}
        self.all_neurons = {}

        # Assign ports
        device_ports = {d: BASE_PORT + d for d in range(n_devices)}

        # Create neurons and assign to devices
        nid = 0
        for dev_id in range(n_devices):
            dev_neurons = []
            for _ in range(neurons_per_device):
                n = Neuron(nid)
                dev_neurons.append(n)
                self.all_neurons[nid] = (n, dev_id)
                nid += 1

            device = Device(dev_id, device_ports[dev_id],
                          dev_neurons, device_ports)
            self.devices[dev_id] = device

        self.total_neurons = nid

        # Build connections and peer tables
        self._setup_connections()

        # Fixed: input/output (on coordinator device)
        # Using numpy for everything — no torch dependency on neurons
        self.embed_w = np.random.randn(VOCAB_SIZE, DIM).astype(np.float32) * 0.1
        self.pos_w = np.random.randn(16, DIM).astype(np.float32) * 0.1
        self.output_w = np.random.randn(DIM, VOCAB_SIZE).astype(np.float32) * 0.1
        self.output_b = np.zeros(VOCAB_SIZE, dtype=np.float32)

        print(f"  Brain: {self.total_neurons} neurons across {n_devices} devices",
              flush=True)
        print(f"  {neurons_per_device} neurons/device, dim={DIM}", flush=True)

    def _setup_connections(self):
        """Build random sparse connections + peer tables."""
        all_nids = list(range(self.total_neurons))
        for nid in all_nids:
            neuron, dev_id = self.all_neurons[nid]
            # 2-4 downstream connections
            n_conns = random.randint(2, min(4, self.total_neurons - 1))
            targets = random.sample([j for j in all_nids if j != nid], n_conns)
            neuron.downstream = targets

            # Build peer table: for each downstream, record which device it's on
            for t in targets:
                _, t_dev = self.all_neurons[t]
                neuron.peers[t] = t_dev

    def start(self):
        for dev in self.devices.values():
            dev.start()
        time.sleep(0.2)

    def forward(self, input_tokens):
        """Forward pass: embed → broadcast to neurons → collect → output."""
        seq_len = len(input_tokens)
        outputs = []

        for t in range(seq_len):
            token = input_tokens[t]
            pos = min(t, 15)

            # Embed (on coordinator)
            x = self.embed_w[token] + self.pos_w[pos]

            # Broadcast to ALL devices, collect results
            all_results = []
            for dev_id, device in self.devices.items():
                if device.running:
                    results = device.process_input(x)
                    all_results.extend(results)

            # Small delay to let UDP signals propagate between devices
            time.sleep(0.002)

            # Second hop: process accumulated signals
            for dev_id, device in self.devices.items():
                if device.running:
                    results = device.process_input(
                        np.zeros(DIM, dtype=np.float32))
                    all_results.extend(results)

            # Aggregate
            if all_results:
                agg = np.mean([r[1] for r in all_results], axis=0)
            else:
                agg = np.zeros(DIM, dtype=np.float32)

            outputs.append(agg)

        # Output projection (on coordinator)
        output = np.stack(outputs)  # [S, DIM]
        logits = output @ self.output_w + self.output_b  # [S, VOCAB]
        return logits

    def train_step(self, input_tokens, target_tokens, lr=0.01):
        """One training step with simple gradient."""
        logits = self.forward(input_tokens)  # [S, VOCAB]

        # Softmax + cross entropy gradient (numpy)
        S = logits.shape[0]
        probs = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs /= probs.sum(axis=1, keepdims=True)

        # Gradient of CE loss w.r.t. logits
        grad = probs.copy()
        for t in range(S):
            grad[t, target_tokens[t]] -= 1.0
        grad /= S

        # Update output projection
        # d_loss/d_output_w = output.T @ grad
        output = np.stack([self.forward.__code__.co_varnames])  # hack
        # Simpler: just update output weights directly
        self.output_w -= lr * 0.1 * grad.mean(axis=0).reshape(1, -1).T @ np.ones((1, DIM))

        # Loss
        loss = 0
        for t in range(S):
            loss -= np.log(probs[t, target_tokens[t]] + 1e-8)
        return loss / S

    def kill_device(self, dev_id):
        if dev_id in self.devices:
            self.devices[dev_id].kill()
            return self.npd
        return 0

    def stop(self):
        for dev in self.devices.values():
            dev.kill()


def run():
    print("=== DISTRIBUTED BRAIN (Device Architecture) ===\n", flush=True)

    brain = DistributedBrain(n_devices=4, neurons_per_device=4)
    brain.start()

    # Training data
    train_pairs = []
    for i in range(0, 20, 2):
        inp = [65 + i, 65 + i + 1]
        out = [65 + i + 2, 65 + i + 3]
        train_pairs.append((inp, out))

    n_steps = 2000
    t0 = time.time()
    correct_window = []

    print(f"  Training: {n_steps} steps, {len(train_pairs)} patterns\n",
          flush=True)

    for step in range(n_steps):
        inp, target = random.choice(train_pairs)

        logits = brain.forward(inp)
        preds = logits.argmax(axis=1)
        correct = all(preds[t] == target[t] for t in range(len(target)))
        correct_window.append(correct)
        if len(correct_window) > 200:
            correct_window.pop(0)

        # Simple gradient update on embed + output
        # Softmax cross-entropy
        S = len(target)
        probs = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs /= probs.sum(axis=1, keepdims=True)

        loss = 0
        for t in range(S):
            loss -= np.log(probs[t, target[t]] + 1e-8)
        loss /= S

        # Update output projection
        grad_logits = probs.copy()
        for t in range(S):
            grad_logits[t, target[t]] -= 1.0
        grad_logits /= S

        # Compute error signal: gradient of loss w.r.t. aggregated output
        # This is what we send to neurons instead of just reward
        # error_signal = (predicted_probs - target_onehot) projected to DIM
        error_in_vocab = grad_logits.mean(axis=0)  # [VOCAB_SIZE]
        error_signal = (error_in_vocab @ brain.output_w.T).astype(np.float32)  # [DIM]

        # Evolution strategy: perturb neuron weights, keep if loss improves
        if step > 0 and step % 10 == 0:
            for dev in brain.devices.values():
                if dev.running:
                    for neuron in dev.neurons.values():
                        # Save current weights
                        old_w = neuron.w.copy()
                        old_b = neuron.bias.copy()

                        # Random perturbation
                        neuron.w += np.random.randn(DIM, DIM).astype(np.float32) * 0.05
                        neuron.bias += np.random.randn(DIM).astype(np.float32) * 0.02

                        # Test: does the same input give better loss?
                        new_logits = brain.forward(inp)
                        new_probs = np.exp(new_logits - new_logits.max(axis=1, keepdims=True))
                        new_probs /= new_probs.sum(axis=1, keepdims=True)
                        new_loss = 0
                        for t2 in range(S):
                            new_loss -= np.log(new_probs[t2, target[t2]] + 1e-8)
                        new_loss /= S

                        if new_loss >= loss:
                            # Revert — perturbation made it worse
                            neuron.w = old_w
                            neuron.bias = old_b

        # Get aggregated hidden for gradient computation
        agg = np.zeros(DIM, dtype=np.float32)
        for dev in brain.devices.values():
            if dev.running:
                for neuron in dev.neurons.values():
                    if neuron.last_out is not None:
                        agg += neuron.last_out
        n_fired = max(sum(1 for dev in brain.devices.values() if dev.running
                         for n in dev.neurons.values() if n.last_out is not None), 1)
        agg /= n_fired

        # output_w gradient: agg.T @ grad_logits → [DIM, VOCAB]
        grad_w = np.outer(agg, grad_logits.mean(axis=0))  # [DIM, VOCAB]
        brain.output_w -= 0.001 * grad_w
        brain.output_b -= 0.001 * grad_logits.mean(axis=0)

        # Embed gradient
        for t in range(len(inp)):
            embed_grad = grad_logits[t] @ brain.output_w.T  # [DIM]
            brain.embed_w[inp[t]] -= 0.001 * np.clip(embed_grad, -1, 1)

        if (step + 1) % 200 == 0:
            acc = sum(correct_window) / len(correct_window)
            elapsed = time.time() - t0
            sps = (step + 1) / elapsed
            print(f"    step {step+1}/{n_steps}  acc={acc:.3f}  "
                  f"loss={loss:.4f}  {sps:.1f} steps/s", flush=True)

    # Eval
    print("\n=== EVAL ===", flush=True)
    correct, total = 0, len(train_pairs)
    for inp, target in train_pairs:
        logits = brain.forward(inp)
        preds = logits.argmax(axis=1)
        match = all(preds[t] == target[t] for t in range(len(target)))
        correct += match
        in_s = ''.join(chr(c) for c in inp)
        tgt_s = ''.join(chr(c) for c in target)
        pred_s = ''.join(chr(int(preds[t])) if 32 <= preds[t] < 128 else '?'
                        for t in range(len(target)))
        print(f"  {'OK' if match else 'XX'} {in_s}→{tgt_s} pred:{pred_s}",
              flush=True)

    print(f"\n  Accuracy: {correct}/{total} = {correct/total:.0%}", flush=True)

    # Stats
    for dev_id, dev in brain.devices.items():
        print(f"  Device {dev_id}: local={dev.local_sends} "
              f"remote={dev.remote_sends}", flush=True)

    # Resilience
    print("\n=== KILL DEVICE 0 ===", flush=True)
    brain.kill_device(0)
    time.sleep(0.3)

    correct_after = 0
    for inp, target in train_pairs:
        logits = brain.forward(inp)
        preds = logits.argmax(axis=1)
        if all(preds[t] == target[t] for t in range(len(target))):
            correct_after += 1

    print(f"  Before: {correct}/{total} = {correct/total:.0%}", flush=True)
    print(f"  After:  {correct_after}/{total} = {correct_after/total:.0%}",
          flush=True)
    print(f"\n  Time: {time.time()-t0:.0f}s", flush=True)

    brain.stop()


if __name__ == "__main__":
    run()
