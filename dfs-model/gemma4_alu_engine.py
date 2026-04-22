import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from safetensors import safe_open
import os
import json
import numpy as np
import time
import sys


class Gemma4ALUEngine:
    """
    Carrier + Fraction inference engine for Gemma-4-27B.

    Weights are stored as:
      carrier (block mean, FP16) + 1-bit sign delta × amplitude

    Reconstruction: W = carrier + (sign ? +amp : -amp)
    """

    def __init__(self, model_dir):
        self.model_dir = model_dir
        with open(os.path.join(model_dir, "config.json")) as f:
            full_config = json.load(f)

        # Text config
        tc = full_config["text_config"]
        self.hidden_size = tc["hidden_size"]
        self.num_layers = tc["num_hidden_layers"]
        self.num_heads = tc["num_attention_heads"]
        self.head_dim = tc["head_dim"]                          # 256 (sliding)
        self.global_head_dim = tc["global_head_dim"]            # 512 (full)
        self.num_kv_heads = tc["num_key_value_heads"]           # 16  (sliding)
        self.num_global_kv_heads = tc["num_global_key_value_heads"]  # 4 (full)
        self.intermediate_size = tc["intermediate_size"]
        self.rms_norm_eps = tc["rms_norm_eps"]
        self.layer_types = tc["layer_types"]                    # "sliding_attention" or "full_attention"
        self.k_eq_v = tc.get("attention_k_eq_v", False)
        self.logit_softcapping = tc.get("final_logit_softcapping", None)

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)

        # Load all compressed shards
        self.weights = {}
        for fname in sorted(os.listdir(model_dir)):
            if fname.endswith("_compressed.safetensors"):
                print(f"Loading {fname}...")
                with safe_open(os.path.join(model_dir, fname), framework="pt", device="cpu") as sf:
                    for k in sf.keys():
                        self.weights[k] = sf.get_tensor(k)
        print(f"Loaded {len(self.weights)} tensors. Gemma-4-27B ALU engine ready.")

    def _rms_norm(self, x, weight_key):
        w = self.weights[weight_key].float()
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.rms_norm_eps) * w

    def _alu_linear(self, x, weight_prefix):
        """Reconstruct weight from carrier+fraction and apply linear."""
        carriers = self.weights[f"{weight_prefix}.carriers"].float()
        packed_signs = self.weights[f"{weight_prefix}.signs"]
        amp = self.weights[f"{weight_prefix}.amplitude"].float()

        total_params = carriers.numel() * 32
        in_dim = x.shape[-1]
        out_dim = total_params // in_dim

        signs = torch.from_numpy(np.unpackbits(packed_signs.numpy())).bool()
        signs = signs[:total_params]

        W = carriers.repeat_interleave(32)[:total_params] + torch.where(signs, amp, -amp)
        return F.linear(x, W.view(out_dim, in_dim))

    def _layer_prefix(self, i):
        return f"model.language_model.layers.{i}"

    def _weight_prefix(self, i, name):
        return f"model.language_model.layers.{i}.{name}.weight"

    def forward(self, input_ids):
        """Single forward pass through all layers. Returns logits for last token."""
        # Embed
        h = F.embedding(input_ids, self.weights["model.language_model.embed_tokens.weight"].float())
        # Gemma scales embeddings
        h = h * (self.hidden_size ** 0.5)

        t0 = time.time()
        for i in range(self.num_layers):
            lp = self._layer_prefix(i)
            is_full = self.layer_types[i] == "full_attention"

            # Per-layer dims
            if is_full:
                hd = self.global_head_dim
                n_kv = self.num_global_kv_heads
            else:
                hd = self.head_dim
                n_kv = self.num_kv_heads
            n_h = self.num_heads
            reps = n_h // n_kv

            # --- Attention block ---
            normed = self._rms_norm(h, f"{lp}.input_layernorm.weight")

            q = self._alu_linear(normed, f"{lp}.self_attn.q_proj.weight")
            k = self._alu_linear(normed, f"{lp}.self_attn.k_proj.weight")

            # k_eq_v: full attention layers share K and V
            if is_full and self.k_eq_v:
                v = k.clone()
            else:
                v = self._alu_linear(normed, f"{lp}.self_attn.v_proj.weight")

            seq_len = q.shape[1]
            q = q.view(1, seq_len, n_h, hd).transpose(1, 2)
            k = k.view(1, seq_len, n_kv, hd).transpose(1, 2)
            v = v.view(1, seq_len, n_kv, hd).transpose(1, 2)

            # GQA expand
            if reps > 1:
                k = k.repeat_interleave(reps, dim=1)
                v = v.repeat_interleave(reps, dim=1)

            # QK norm
            qn_key = f"{lp}.self_attn.q_norm.weight"
            if qn_key in self.weights:
                q = q * self.weights[qn_key].view(1, 1, 1, -1).float()
                k = k * self.weights[f"{lp}.self_attn.k_norm.weight"].view(1, 1, 1, -1).float()

            # Attention
            attn = torch.matmul(q, k.transpose(-2, -1)) / (hd ** 0.5)
            attn = F.softmax(attn, dim=-1)
            out = torch.matmul(attn, v)
            out = out.transpose(1, 2).reshape(1, seq_len, n_h * hd)
            attn_out = self._alu_linear(out, f"{lp}.self_attn.o_proj.weight")

            # Post-attention norm + residual
            attn_out = self._rms_norm(attn_out, f"{lp}.post_attention_layernorm.weight")
            h = h + attn_out

            # --- MLP block ---
            normed2 = self._rms_norm(h, f"{lp}.pre_feedforward_layernorm.weight")

            gate = F.gelu(self._alu_linear(normed2, f"{lp}.mlp.gate_proj.weight"), approximate='tanh')
            up = self._alu_linear(normed2, f"{lp}.mlp.up_proj.weight")
            mlp_out = self._alu_linear(gate * up, f"{lp}.mlp.down_proj.weight")

            # Post-feedforward norm + residual
            mlp_out = self._rms_norm(mlp_out, f"{lp}.post_feedforward_layernorm.weight")

            # Layer scalar
            scalar_key = f"{lp}.layer_scalar"
            if scalar_key in self.weights:
                mlp_out = mlp_out * self.weights[scalar_key].float()

            h = h + mlp_out

            elapsed = time.time() - t0
            print(f"  Layer {i:2d}/{self.num_layers} ({'full' if is_full else 'slid'}) — {elapsed:.1f}s", end='\r')

        print()

        # Final norm
        h = self._rms_norm(h, "model.language_model.norm.weight")

        # LM head (tied embeddings)
        logits = F.linear(h[:, -1:, :], self.weights["model.language_model.embed_tokens.weight"].float())

        # Logit softcapping
        if self.logit_softcapping:
            cap = self.logit_softcapping
            logits = cap * torch.tanh(logits / cap)

        # Show top predictions
        probs = F.softmax(logits[0, -1], dim=-1)
        topk = torch.topk(probs, 10)
        print("\n  Top 10 predictions:")
        for prob, idx in zip(topk.values, topk.indices):
            tok = self.tokenizer.decode([idx.item()])
            print(f"    {prob.item():.4f}  '{tok}'")

        return logits

    def generate(self, prompt, max_tokens=10):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids
        generated = []

        print(f"Prompt: {prompt}")
        print(f"Tokens: {input_ids.shape[1]}, generating up to {max_tokens}...")

        for step in range(max_tokens):
            print(f"\n--- Step {step+1}/{max_tokens} ---")
            logits = self.forward(input_ids)
            next_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            token_str = self.tokenizer.decode(next_id[0])
            generated.append(token_str)
            print(f"  -> '{token_str}'")

            if next_id.item() in (self.tokenizer.eos_token_id if isinstance(self.tokenizer.eos_token_id, list) else [self.tokenizer.eos_token_id]):
                break

            input_ids = torch.cat([input_ids, next_id], dim=1)

        return "".join(generated)


if __name__ == "__main__":
    max_tokens = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    engine = Gemma4ALUEngine(".")
    prompt = "The capital of France is"
    print(f"\nUser: {prompt}")
    t0 = time.time()
    result = engine.generate(prompt, max_tokens=max_tokens)
    print(f"\nGemma-4-ALU: {prompt}{result}")
    print(f"Total time: {time.time()-t0:.1f}s")
