"""
Numpy Decoder: pure-numpy inference engine for text generation.

Like vLLM/Ollama but:
  - Weights from .npy files (mutable — swap/edit any layer at runtime)
  - CPU-native (no GPU, no PyTorch at inference time)
  - KV cache in numpy arrays (like Ollama's llama.cpp KV cache)
  - Convergence-guided generation (bias toward LMDB knowledge)

Architecture mirrors GPT-2 exactly:
  Token embed + Pos embed → N × (LayerNorm → Attention → LayerNorm → FFN) → Final LN → LM Head

Usage:
    dec = NumpyDecoder('/tmp/gpt2_all_layers')
    tokens = dec.generate([464, 3139, 286, 4881, 318], max_tokens=30)
    # tokens = [464, 3139, 286, 4881, 318, 6342, ...]
"""

import os
import json
import math
import numpy as np
from typing import List, Optional


class NumpyDecoder:
    """Token-by-token text generation from numpy weight files.

    Loads all layer weights on init. Runs autoregressive generation
    with KV cache — same algorithm as vLLM/llama.cpp, in pure numpy.
    """

    def __init__(self, weights_dir: str):
        self._dir = weights_dir
        self.meta = json.load(open(os.path.join(weights_dir, 'meta.json')))
        self.n_layers = self.meta['n_layers']
        self.dim = self.meta['dim']
        self.n_heads = self.meta.get('n_heads', 12)
        self.head_dim = self.dim // self.n_heads
        self.vocab_size = self.meta.get('vocab_size', 50257)

        # Token + position embeddings
        self.embed = np.load(os.path.join(weights_dir, 'embeddings.npy'))
        self.pos_embed = np.load(os.path.join(weights_dir, 'pos_embeddings.npy'))

        # Final LayerNorm
        self.final_ln_w = np.load(os.path.join(weights_dir, 'final_ln_w.npy'))
        self.final_ln_b = np.load(os.path.join(weights_dir, 'final_ln_b.npy'))

        # Per-layer weights
        self.layers = []
        for i in range(self.n_layers):
            d = os.path.join(weights_dir, f'layer_{i}')
            layer = {}
            for name in ['W_Q', 'W_K', 'W_V', 'W_O',
                         'b_Q', 'b_K', 'b_V', 'b_O',
                         'FFN_up', 'FFN_up_b', 'FFN_down', 'FFN_down_b',
                         'LN1_w', 'LN1_b', 'LN2_w', 'LN2_b']:
                layer[name] = np.load(os.path.join(d, f'{name}.npy'))
            self.layers.append(layer)

    @staticmethod
    def _layer_norm(x, w, b, eps=1e-5):
        """LayerNorm: (x - mean) / sqrt(var + eps) * w + b"""
        mean = x.mean(axis=-1, keepdims=True)
        var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
        return w * (x - mean) / np.sqrt(var + eps) + b

    @staticmethod
    def _gelu(x):
        """GELU activation (GPT-2 uses the tanh approximation)."""
        return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))

    def _attention(self, x, layer, kv_cache):
        """Multi-head causal self-attention with KV cache.

        GPT-2 uses conv1d-style projections (input @ weight + bias)
        where weight shapes are (in_dim, out_dim) — so we do x @ W, not x @ W.T.
        """
        L = layer

        # Project Q, K, V
        Q = x @ L['W_Q'] + L['b_Q']  # (seq, dim)
        K = x @ L['W_K'] + L['b_K']
        V = x @ L['W_V'] + L['b_V']

        # Append K, V to cache
        if kv_cache['K'].shape[0] > 0:
            K_full = np.concatenate([kv_cache['K'], K], axis=0)
            V_full = np.concatenate([kv_cache['V'], V], axis=0)
        else:
            K_full = K
            V_full = V
        kv_cache['K'] = K_full
        kv_cache['V'] = V_full

        seq_q = Q.shape[0]
        seq_kv = K_full.shape[0]

        # Reshape for multi-head: (seq, dim) → (n_heads, seq, head_dim)
        Q_mh = Q.reshape(seq_q, self.n_heads, self.head_dim).transpose(1, 0, 2)
        K_mh = K_full.reshape(seq_kv, self.n_heads, self.head_dim).transpose(1, 0, 2)
        V_mh = V_full.reshape(seq_kv, self.n_heads, self.head_dim).transpose(1, 0, 2)

        # Scaled dot-product attention per head
        # (n_heads, seq_q, head_dim) @ (n_heads, head_dim, seq_kv) → (n_heads, seq_q, seq_kv)
        scores = np.matmul(Q_mh, K_mh.transpose(0, 2, 1)) / math.sqrt(self.head_dim)

        # Causal mask: can only attend to positions <= current
        mask = np.triu(np.ones((seq_q, seq_kv), dtype=np.float32),
                       k=seq_kv - seq_q + 1)
        scores = np.where(mask[None, :, :] > 0, -1e9, scores)

        # Softmax
        scores_max = scores.max(axis=-1, keepdims=True)
        exp_scores = np.exp(scores - scores_max)
        weights = exp_scores / (exp_scores.sum(axis=-1, keepdims=True) + 1e-10)

        # Weighted sum of values
        # (n_heads, seq_q, seq_kv) @ (n_heads, seq_kv, head_dim) → (n_heads, seq_q, head_dim)
        out = np.matmul(weights, V_mh)

        # Reshape back: (n_heads, seq_q, head_dim) → (seq_q, dim)
        out = out.transpose(1, 0, 2).reshape(seq_q, self.dim)

        # Output projection
        return out @ L['W_O'] + L['b_O']

    def _ffn(self, x, layer):
        """Feed-forward: up-project → GELU → down-project."""
        h = x @ layer['FFN_up'] + layer['FFN_up_b']
        h = self._gelu(h)
        return h @ layer['FFN_down'] + layer['FFN_down_b']

    def _forward(self, token_ids, positions, kv_caches):
        """Forward pass for one or more tokens.

        token_ids: list of int
        positions: list of int (position indices)
        kv_caches: list of {'K': array, 'V': array} per layer

        Returns: logits array (vocab_size,) for the last token.
        """
        seq_len = len(token_ids)

        # Embed tokens + positions
        x = self.embed[token_ids] + self.pos_embed[positions]  # (seq, dim)
        x = x.astype(np.float32)

        # Transformer layers
        for i, layer in enumerate(self.layers):
            residual = x
            x = self._layer_norm(x, layer['LN1_w'], layer['LN1_b'])
            x = self._attention(x, layer, kv_caches[i])
            x = x + residual

            residual = x
            x = self._layer_norm(x, layer['LN2_w'], layer['LN2_b'])
            x = self._ffn(x, layer)
            x = x + residual

        # Final LayerNorm
        x = self._layer_norm(x, self.final_ln_w, self.final_ln_b)

        # LM head (tied weights: embed.T)
        logits = x[-1:] @ self.embed.T  # (1, vocab)
        return logits.ravel()

    def _init_kv_caches(self):
        """Create empty KV caches for all layers."""
        return [{'K': np.empty((0, self.dim), dtype=np.float32),
                 'V': np.empty((0, self.dim), dtype=np.float32)}
                for _ in range(self.n_layers)]

    def generate(self, prompt_ids: List[int], max_tokens: int = 50,
                 temperature: float = 1.0, top_k: int = 0,
                 greedy: bool = False) -> List[int]:
        """Generate text token by token. Pure numpy.

        Args:
            prompt_ids: tokenized prompt
            max_tokens: how many new tokens to generate
            temperature: sampling temperature (ignored if greedy)
            top_k: top-k sampling (0 = disabled)
            greedy: if True, always pick highest probability token

        Returns: full sequence (prompt + generated tokens)
        """
        kv_caches = self._init_kv_caches()
        generated = list(prompt_ids)

        # Prefill: process all prompt tokens at once
        positions = list(range(len(prompt_ids)))
        logits = self._forward(prompt_ids, positions, kv_caches)

        # Decode: one token at a time
        for step in range(max_tokens):
            if greedy:
                next_token = int(np.argmax(logits))
            else:
                scaled = logits / max(temperature, 1e-8)

                if top_k > 0 and top_k < len(scaled):
                    threshold = np.partition(scaled, -top_k)[-top_k]
                    scaled = np.where(scaled < threshold, -1e9, scaled)

                probs = np.exp(scaled - scaled.max())
                probs /= probs.sum()
                next_token = int(np.random.choice(len(probs), p=probs))

            generated.append(next_token)
            pos = len(generated) - 1

            if pos >= self.pos_embed.shape[0]:
                break  # exceeded max sequence length

            logits = self._forward([next_token], [pos], kv_caches)

        return generated

    # --- Mutable weights ---

    def update_weight(self, layer_idx: int, weight_name: str,
                      value: np.ndarray, persist: bool = False):
        """Edit a weight matrix in a specific layer. Immediate effect.

        This is what vLLM/Ollama can't do — mutable weights at runtime.
        """
        self.layers[layer_idx][weight_name] = value.astype(np.float32)
        if persist:
            path = os.path.join(self._dir, f'layer_{layer_idx}', f'{weight_name}.npy')
            np.save(path, self.layers[layer_idx][weight_name])

    def get_weight(self, layer_idx: int, weight_name: str) -> np.ndarray:
        """Read a weight matrix. Inspectable."""
        return self.layers[layer_idx][weight_name]
