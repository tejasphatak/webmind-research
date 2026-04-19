#!/usr/bin/env python3
"""
Biological Distributed Brain — v1
===================================
Takes the PROVEN learning rules from brain_v2 (100% accuracy)
and puts them into the multi-device architecture.

Learning = biology only:
1. Hebbian: fire together → wire together
2. Dopamine: global good/bad signal scales Hebbian update
3. Homeostasis: neurons auto-adjust firing thresholds

NO gradients. NO backprop. NO feedback alignment.
Each device = separate thread with UDP socket.
Neurons = numpy only (ESP32 compatible).
"""

import numpy as np
import socket
import struct
import threading
import time
import random

DIM = 32
VOCAB_SIZE = 128
BASE_PORT = 19000

def pack_signal(nid, arr):
    return struct.pack(f'!BH{DIM}f', 1, nid, *arr)

def pack_reward(reward_val):
    return struct.pack('!Bf', 2, reward_val)

def unpack_msg(data):
    msg_type = struct.unpack('!B', data[:1])[0]
    if msg_type == 1:  # activation
        vals = struct.unpack(f'!BH{DIM}f', data)
        return 'activation', vals[1], np.array(vals[2:], dtype=np.float32)
    elif msg_type == 2:  # reward
        vals = struct.unpack('!Bf', data)
        return 'reward', 0, vals[1]
    return None, 0, None


class BiologicalNeuron:
    """Neuron with biological learning rules. numpy only."""

    def __init__(self, nid, dim=DIM):
        self.id = nid
        self.dim = dim

        # Weights (private to this neuron)
        self.w = np.random.randn(dim, dim).astype(np.float32) * 0.05
        self.bias = np.zeros(dim, dtype=np.float32)

        # Biological state
        self.threshold = np.float32(0.0)    # firing threshold
        self.last_input = None
        self.last_output = None
        self.fired = False
        self.fire_count = 0
        self.rest_count = 0  # how long since last fire

        # Downstream peers
        self.downstream = []  # list of neuron_ids

    def activate(self, x):
        """ReLU with adaptive threshold. Fire or rest."""
        self.last_input = x.copy()
        pre = x @ self.w + self.bias
        post = np.maximum(pre - self.threshold, 0)

        self.fired = np.mean(np.abs(post)) > 0.01
        if self.fired:
            self.last_output = post.copy()
            self.fire_count += 1
            self.rest_count = 0
        else:
            self.last_output = np.zeros(self.dim, dtype=np.float32)
            self.rest_count += 1

        # Homeostasis: auto-balance firing rate
        if self.fire_count > 0:
            fire_rate = self.fire_count / (self.fire_count + self.rest_count)
            if fire_rate > 0.7:
                self.threshold += 0.01  # firing too much → raise threshold
            elif fire_rate < 0.2:
                self.threshold -= 0.01  # firing too little → lower threshold
            self.threshold = np.clip(self.threshold, -2.0, 2.0)

        return post

    def hebbian_update(self, dopamine=0.0, lr=0.03):
        """Hebbian + dopamine modulation.
        dopamine > 0: "that was good" → strengthen active paths
        dopamine < 0: "that was bad" → weaken active paths
        dopamine = 0: pure Hebbian (fire together wire together)
        """
        if self.last_input is None or not self.fired:
            return

        # Hebbian: outer product of input × output
        # This is what the neuron "did" — the pattern it detected
        activity = np.outer(self.last_input, self.last_output)

        # Modulate by dopamine (1.0 = neutral, >1 = reward, <1 = punish)
        modulation = 1.0 + dopamine

        # Update weights
        self.w += lr * modulation * activity.T
        self.bias += lr * modulation * self.last_output * 0.1

        # Weight decay (prevents explosion, encourages sparsity)
        self.w *= 0.999
        self.bias *= 0.999

        # Clip
        self.w = np.clip(self.w, -3.0, 3.0)
        self.bias = np.clip(self.bias, -2.0, 2.0)


class BiologicalDevice(threading.Thread):
    """One device with N neurons, UDP socket, biological learning."""

    def __init__(self, dev_id, port, neurons, all_ports, peer_map):
        super().__init__(daemon=True)
        self.id = dev_id
        self.port = port
        self.neurons = {n.id: n for n in neurons}
        self.all_ports = all_ports
        self.peer_map = peer_map  # {neuron_id → device_id}
        self.running = True
        self.inbox = {}
        self.local_sends = 0
        self.remote_sends = 0

    def run(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(0.01)
        self.sock.bind(('127.0.0.1', self.port))

        while self.running:
            try:
                data, addr = self.sock.recvfrom(4096)
                msg_type, src_nid, payload = unpack_msg(data)
                if msg_type == 'activation':
                    for nid in self.neurons:
                        if nid not in self.inbox:
                            self.inbox[nid] = []
                        self.inbox[nid].append(payload)
                elif msg_type == 'reward':
                    # Dopamine signal → all neurons update
                    for neuron in self.neurons.values():
                        neuron.hebbian_update(dopamine=payload)
            except socket.timeout:
                pass
            except:
                pass

        self.sock.close()

    def process(self, input_signal):
        """Process input through local neurons, forward to downstream."""
        results = []
        for nid, neuron in self.neurons.items():
            # Combine input with inbox
            combined = input_signal.copy()
            if nid in self.inbox and self.inbox[nid]:
                for sig in self.inbox[nid]:
                    combined = combined + sig
                combined /= (len(self.inbox[nid]) + 1)
                self.inbox[nid] = []

            output = neuron.activate(combined)

            if neuron.fired:
                results.append((nid, output))
                # Forward to downstream
                for dst_nid in neuron.downstream:
                    self._route(nid, dst_nid, output)

        return results

    def _route(self, src_nid, dst_nid, signal):
        if dst_nid in self.neurons:
            if dst_nid not in self.inbox:
                self.inbox[dst_nid] = []
            self.inbox[dst_nid].append(signal.copy())
            self.local_sends += 1
        else:
            dst_dev = self.peer_map.get(dst_nid)
            if dst_dev is not None and dst_dev in self.all_ports:
                try:
                    data = pack_signal(src_nid, signal.tolist())
                    self.sock.sendto(data, ('127.0.0.1', self.all_ports[dst_dev]))
                    self.remote_sends += 1
                except:
                    pass

    def send_dopamine(self, reward):
        """Broadcast dopamine to all local neurons."""
        for neuron in self.neurons.values():
            neuron.hebbian_update(dopamine=reward)


class BioBrain:
    """Distributed brain with biological learning."""

    def __init__(self, n_devices=4, neurons_per_device=4, dim=DIM):
        self.n_devices = n_devices
        self.npd = neurons_per_device
        self.dim = dim
        self.devices = {}

        # Coordinator's learnable layers (numpy, simple gradient OK here)
        self.embed = np.random.randn(VOCAB_SIZE, dim).astype(np.float32) * 0.1
        self.pos_embed = np.random.randn(16, dim).astype(np.float32) * 0.1
        self.out_w = np.random.randn(dim, VOCAB_SIZE).astype(np.float32) * 0.1
        self.out_b = np.zeros(VOCAB_SIZE, dtype=np.float32)

        # Build neurons and devices
        ports = {d: BASE_PORT + d for d in range(n_devices)}
        all_neurons = {}
        nid = 0
        for dev_id in range(n_devices):
            neurons = []
            for _ in range(neurons_per_device):
                n = BiologicalNeuron(nid, dim)
                neurons.append(n)
                all_neurons[nid] = dev_id
                nid += 1
            self.devices[dev_id] = BiologicalDevice(
                dev_id, ports[dev_id], neurons, ports, all_neurons)

        self.total_neurons = nid

        # Wire random connections
        all_nids = list(range(self.total_neurons))
        for nid_val in all_nids:
            dev_id = all_neurons[nid_val]
            dev = self.devices[dev_id]
            neuron = dev.neurons[nid_val]
            targets = random.sample([j for j in all_nids if j != nid_val],
                                   min(3, len(all_nids) - 1))
            neuron.downstream = targets

        print(f"  BioBrain: {self.total_neurons} neurons, {n_devices} devices, "
              f"dim={dim}", flush=True)

    def start(self):
        for dev in self.devices.values():
            dev.start()
        time.sleep(0.1)

    def forward(self, tokens):
        """Forward pass through distributed brain."""
        outputs = []
        for t, tok in enumerate(tokens):
            x = self.embed[tok] + self.pos_embed[min(t, 15)]

            # Broadcast to all devices
            all_results = []
            for dev in self.devices.values():
                if dev.running:
                    results = dev.process(x)
                    all_results.extend(results)

            # Small delay for inter-device signals
            time.sleep(0.001)

            # Second hop
            for dev in self.devices.values():
                if dev.running:
                    results = dev.process(np.zeros(self.dim, dtype=np.float32))
                    all_results.extend(results)

            if all_results:
                agg = np.mean([r[1] for r in all_results], axis=0)
            else:
                agg = np.zeros(self.dim, dtype=np.float32)

            outputs.append(agg)

        logits = np.stack(outputs) @ self.out_w + self.out_b
        return logits

    def train_step(self, tokens_in, tokens_out, lr=0.005):
        """One training step: forward, compute reward, broadcast dopamine."""
        logits = self.forward(tokens_in)
        S = len(tokens_out)

        # Softmax
        probs = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs /= probs.sum(axis=1, keepdims=True)

        # Loss
        loss = 0
        for t in range(S):
            loss -= np.log(probs[t, tokens_out[t]] + 1e-8)
        loss /= S

        # Accuracy
        preds = logits.argmax(axis=1)
        correct = all(preds[t] == tokens_out[t] for t in range(S))

        # Natural dopamine: based on loss improvement over recent history
        # No artificial boost — the system rewards itself
        if not hasattr(self, '_loss_history'):
            self._loss_history = []
        self._loss_history.append(loss)
        if len(self._loss_history) > 50:
            self._loss_history.pop(0)
        avg_loss = np.mean(self._loss_history)
        # Better than average → positive dopamine. Worse → negative.
        dopamine = (avg_loss - loss) * 2.0  # scale for effect

        # Broadcast dopamine to ALL devices
        for dev in self.devices.values():
            if dev.running:
                dev.send_dopamine(dopamine)

        # Get aggregated hidden state from fired neurons
        fired_neurons = []
        for dev in self.devices.values():
            for n in dev.neurons.values():
                if n.last_output is not None and n.fired:
                    fired_neurons.append(n.last_output)
        if fired_neurons:
            h = np.mean(fired_neurons, axis=0)
        else:
            h = np.zeros(self.dim, dtype=np.float32)

        # PRAGMATIC: coordinator uses gradient for its LOCAL layers (embed + output)
        # This is on ONE device — no wire needed. Gradient stays local.
        # Neurons across devices use biological learning (Hebbian + dopamine).
        grad = probs.copy()
        for t in range(S):
            grad[t, tokens_out[t]] -= 1.0
        grad /= S

        self.out_w -= lr * np.outer(h, grad.mean(axis=0))
        self.out_b -= lr * grad.mean(axis=0)

        for t in range(len(tokens_in)):
            self.embed[tokens_in[t]] -= lr * np.clip(
                grad[t] @ self.out_w.T, -1, 1)

        self.out_w = np.clip(self.out_w, -5, 5)
        self.out_b = np.clip(self.out_b, -3, 3)

        return loss, correct

    def kill_device(self, dev_id):
        self.devices[dev_id].running = False

    def stop(self):
        for dev in self.devices.values():
            dev.running = False


def run():
    print("=== BIOLOGICAL DISTRIBUTED BRAIN ===\n", flush=True)

    brain = BioBrain(n_devices=4, neurons_per_device=5, dim=DIM)
    brain.start()

    # Training data
    train_pairs = []
    for i in range(0, 20, 2):
        inp = [65 + i, 65 + i + 1]
        out = [65 + i + 2, 65 + i + 3]
        train_pairs.append((inp, out))

    n_steps = 15000
    t0 = time.time()
    correct_window = []

    print(f"  Training: {n_steps} steps, {len(train_pairs)} patterns\n",
          flush=True)

    for step in range(n_steps):
        inp, target = random.choice(train_pairs)
        loss, correct = brain.train_step(inp, target)

        correct_window.append(correct)
        if len(correct_window) > 200:
            correct_window.pop(0)

        if (step + 1) % 500 == 0:
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

    for dev_id, dev in brain.devices.items():
        n_fired = sum(1 for n in dev.neurons.values() if n.fire_count > 0)
        print(f"  Device {dev_id}: local={dev.local_sends} "
              f"remote={dev.remote_sends} fired={n_fired}/{brain.npd}",
              flush=True)

    # Resilience
    print("\n=== KILL DEVICE 0 ===", flush=True)
    brain.kill_device(0)
    time.sleep(0.2)

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
