"""
Multi-Head Phase Interference
==============================

The capacity bottleneck: single-head = one attention pattern = one measurement basis.
Language and vision need multiple simultaneous patterns.

From quantum physics: to fully characterize a quantum state, you measure in
MULTIPLE bases simultaneously. Each measurement (head) applies different
rotation operators (q_rot, k_rot, v_rot) and produces a different interference
pattern. The results are concatenated — like quantum state tomography.

This is NOT standard multi-head attention:
- Standard MHA: H sets of (W_q, W_k, W_v) ∈ R^{d×d/H} = H × 3 × d²/H params
- Phase MHA:    H sets of (θ_q, θ_k, θ_v) ∈ R^{d/H}   = H × 3 × d/H params
- Ratio: d²/H vs d/H = factor of d fewer parameters

For embed_dim=256, that's 256x fewer attention parameters.
"""

import torch
import torch.nn as nn
import math


class ComplexNorm(nn.Module):
    """Normalize magnitude, preserve phase exactly."""
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        mag = torch.abs(x)
        mean_mag = mag.mean(dim=-1, keepdim=True)
        std_mag = mag.std(dim=-1, keepdim=True)
        norm_mag = (mag - mean_mag) / (std_mag + self.eps)
        scale = torch.tanh(norm_mag) / (mag + self.eps)
        return x * scale.to(torch.complex64)


class PhaseMultiHeadAttention(nn.Module):
    """Multi-head phase interference.

    Each head operates on a slice of the embedding dimension with its own
    rotation angles. Heads interfere independently, outputs are concatenated.

    Physics analogy: H independent measurements of the same wavefunction,
    each from a different orientation on the Bloch sphere.
    """
    def __init__(self, embed_dim, num_heads, causal=False):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.causal = causal

        # Each head gets its own rotation angles
        # Shape: (num_heads, head_dim) — independent unitary operators per head
        self.q_rot = nn.Parameter(torch.empty(num_heads, self.head_dim).uniform_(-math.pi, math.pi))
        self.k_rot = nn.Parameter(torch.empty(num_heads, self.head_dim).uniform_(-math.pi, math.pi))
        self.v_rot = nn.Parameter(torch.empty(num_heads, self.head_dim).uniform_(-math.pi, math.pi))

    def forward(self, state):
        """
        state: (batch, seq_len, embed_dim) complex64
        returns: (batch, seq_len, embed_dim) complex64
        """
        B, S, D = state.shape

        # Reshape to (batch, seq_len, num_heads, head_dim)
        state_heads = state.view(B, S, self.num_heads, self.head_dim)

        # Apply per-head phase rotations
        # q_rot: (num_heads, head_dim) → broadcast over batch and seq
        q = state_heads * torch.exp(1j * self.q_rot)  # (B, S, H, d_h)
        k = state_heads * torch.exp(1j * self.k_rot)
        v = state_heads * torch.exp(1j * self.v_rot)

        # Rearrange to (batch, num_heads, seq_len, head_dim) for batched matmul
        q = q.permute(0, 2, 1, 3)  # (B, H, S, d_h)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # Per-head interference: Q_h · K_h†
        interference = torch.matmul(q, k.conj().transpose(-1, -2))  # (B, H, S, S)
        attn_logits = interference.real / math.sqrt(self.head_dim)

        # Causal mask (if decoder)
        if self.causal:
            mask = torch.tril(torch.ones(S, S, device=state.device))
            attn_logits = attn_logits.masked_fill(mask.unsqueeze(0).unsqueeze(0) == 0, float('-inf'))

        attn_weights = torch.softmax(attn_logits * 8.0, dim=-1)

        # Weighted superposition per head
        attn_out = torch.matmul(attn_weights.to(torch.complex64), v)  # (B, H, S, d_h)

        # Concatenate heads back: (B, H, S, d_h) → (B, S, H, d_h) → (B, S, D)
        attn_out = attn_out.permute(0, 2, 1, 3).contiguous().view(B, S, D)

        return attn_out


class PhaseMultiHeadModel(nn.Module):
    """Complete model with multi-head phase interference.

    Single step: encode → multi-head interfere → resonate → normalize → measure

    Can be used for classification (pool + classify) or
    generation (per-position readout).
    """
    def __init__(self, vocab_size, embed_dim=256, num_heads=8,
                 num_classes=None, max_seq_len=128, causal=False,
                 task="classify", patch_pixels=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.task = task

        # Input projection
        if patch_pixels is not None:
            # Vision: patch pixels → embedding
            self.input_proj = nn.Linear(patch_pixels, embed_dim)
        else:
            # Language: token → embedding
            self.input_proj = nn.Embedding(vocab_size, embed_dim)

        self.patch_pixels = patch_pixels
        self.vocab_size = vocab_size

        # Positional phase — multi-frequency sinusoidal
        pos = torch.arange(max_seq_len).unsqueeze(1).float()
        dim = torch.arange(embed_dim).unsqueeze(0).float()
        freq = 1.0 / (10000 ** (dim / embed_dim))
        self.register_buffer('pos_phase', pos * freq * math.pi)

        # Multi-head phase interference
        self.attention = PhaseMultiHeadAttention(embed_dim, num_heads, causal=causal)

        # Complex resonance (feed-forward)
        w_init = torch.empty(embed_dim, embed_dim)
        nn.init.orthogonal_(w_init)
        self.ff_real = nn.Parameter(w_init)
        self.ff_imag = nn.Parameter(torch.empty(embed_dim, embed_dim).uniform_(-0.01, 0.01))

        self.norm = ComplexNorm()

        # Readout depends on task
        if task == "classify":
            self.readout = nn.Sequential(
                nn.Linear(embed_dim * 2, num_classes)  # real + imag concatenated
            )
        else:  # generate
            self.readout_real = nn.Linear(embed_dim, vocab_size)
            self.readout_imag = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        if self.patch_pixels is not None:
            B, S, _ = x.shape
            mag = torch.tanh(self.input_proj(x))
        else:
            B, S = x.shape
            mag = torch.tanh(self.input_proj(x))

        # Wavefunction encoding
        state = mag.to(torch.complex64) * torch.exp(1j * self.pos_phase[:S].unsqueeze(0))

        # Multi-head phase interference (single step)
        attn_out = self.attention(state)
        state = state + attn_out  # residual superposition

        # Complex resonance
        ff_weights = torch.complex(self.ff_real, self.ff_imag)
        state = torch.matmul(state, ff_weights)
        state = self.norm(state)

        # Measurement
        if self.task == "classify":
            pooled = state.mean(dim=1)  # (B, embed_dim) complex
            features = torch.cat([pooled.real, pooled.imag], dim=-1)
            return self.readout(features)
        else:
            return self.readout_real(state.real) + self.readout_imag(state.imag)

    @torch.no_grad()
    def generate(self, seed, max_new_tokens, temperature=0.8, top_p=0.9):
        for _ in range(max_new_tokens):
            logits = self.forward(seed)
            next_logits = logits[:, -1, :] / temperature

            sorted_logits, sorted_idx = torch.sort(next_logits, descending=True)
            cumprobs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            remove = cumprobs > top_p
            remove[..., 1:] = remove[..., :-1].clone()
            remove[..., 0] = False
            sorted_logits[remove] = float('-inf')

            probs = torch.softmax(sorted_logits, dim=-1)
            sampled = torch.multinomial(probs, 1)
            next_token = sorted_idx.gather(-1, sampled)
            seed = torch.cat([seed, next_token], dim=1)
        return seed
