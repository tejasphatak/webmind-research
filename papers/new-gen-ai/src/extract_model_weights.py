"""
Extract weights from any HuggingFace model → .npy files for LMDB integration.

Knowledge transfer: pretrained models store language understanding in weight
matrices. This script cracks them open, extracts embeddings + attention weights,
and saves them in a format our convergence loop can use.

Supports:
  - Encoder models (BERT, MiniLM, mpnet): separate Q/K/V per layer
  - Decoder models (GPT-2): fused c_attn, split into Q/K/V
  - Decoder models (Gemma, Llama): separate q_proj/k_proj/v_proj per layer

Usage:
    # Extract from MiniLM (already cached)
    python3 extract_model_weights.py --model sentence-transformers/all-MiniLM-L6-v2

    # Extract from GPT-2 (already cached)
    python3 extract_model_weights.py --model gpt2

    # Extract from specific layer
    python3 extract_model_weights.py --model gpt2 --layer 10

    # Custom output path
    python3 extract_model_weights.py --model gpt2 --output ~/nexus-brain/model_weights/gpt2

    # List what's in a model (no extraction)
    python3 extract_model_weights.py --model gpt2 --list
"""

import os
import sys
import json
import time
import argparse
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Tuple


# ---------------------------------------------------------------------------
# Architecture detection + weight key mapping
# ---------------------------------------------------------------------------

# Each architecture family maps to:
#   embedding_key: path to token embedding matrix
#   layer_pattern: how to find attention weights per layer
#   qkv_style: 'separate' (BERT/Llama) or 'fused' (GPT-2)
#   n_layers_key: config.json key for layer count
#   dim_key: config.json key for hidden dim

ARCHITECTURES = {
    'bert': {
        'embedding_key': 'embeddings.word_embeddings.weight',
        'layer_prefix': 'encoder.layer.{N}',
        'q_key': 'attention.self.query.weight',
        'k_key': 'attention.self.key.weight',
        'v_key': 'attention.self.value.weight',
        'q_bias': 'attention.self.query.bias',
        'k_bias': 'attention.self.key.bias',
        'v_bias': 'attention.self.value.bias',
        'ffn_up_key': 'intermediate.dense.weight',
        'ffn_down_key': 'output.dense.weight',
        'qkv_style': 'separate',
        'n_layers_key': 'num_hidden_layers',
        'dim_key': 'hidden_size',
    },
    'gpt2': {
        'embedding_key': 'wte.weight',
        'layer_prefix': 'h.{N}',
        'fused_qkv_key': 'attn.c_attn.weight',
        'fused_qkv_bias': 'attn.c_attn.bias',
        'ffn_up_key': 'mlp.c_fc.weight',
        'ffn_down_key': 'mlp.c_proj.weight',
        'qkv_style': 'fused',
        'n_layers_key': 'n_layer',
        'dim_key': 'n_embd',
    },
    'llama': {
        'embedding_key': 'model.embed_tokens.weight',
        'layer_prefix': 'model.layers.{N}',
        'q_key': 'self_attn.q_proj.weight',
        'k_key': 'self_attn.k_proj.weight',
        'v_key': 'self_attn.v_proj.weight',
        'ffn_up_key': 'mlp.up_proj.weight',
        'ffn_down_key': 'mlp.down_proj.weight',
        'qkv_style': 'separate',
        'n_layers_key': 'num_hidden_layers',
        'dim_key': 'hidden_size',
    },
}

# Map model_type from config.json → architecture family
MODEL_TYPE_MAP = {
    'bert': 'bert',
    'distilbert': 'bert',
    'roberta': 'bert',
    'gpt2': 'gpt2',
    'llama': 'llama',
    'gemma': 'llama',   # Gemma uses same key patterns as Llama
    'gemma2': 'llama',
    'mistral': 'llama',
    'phi': 'llama',
    'qwen2': 'llama',
}


def find_model_path(model_name: str) -> Tuple[Path, dict]:
    """Find model files in HuggingFace cache. Returns (snapshot_dir, config)."""
    cache_dir = Path.home() / '.cache' / 'huggingface' / 'hub'

    # Normalize model name to cache directory format
    cache_name = 'models--' + model_name.replace('/', '--')
    model_dir = cache_dir / cache_name

    if not model_dir.exists():
        raise FileNotFoundError(
            f"Model not found in HF cache: {model_dir}\n"
            f"Download it first: python3 -c \"from transformers import AutoModel; "
            f"AutoModel.from_pretrained('{model_name}')\"")

    # Find the latest snapshot
    snapshots_dir = model_dir / 'snapshots'
    if not snapshots_dir.exists():
        raise FileNotFoundError(f"No snapshots dir: {snapshots_dir}")

    snapshot_dirs = sorted(snapshots_dir.iterdir(), key=lambda p: p.stat().st_mtime)
    if not snapshot_dirs:
        raise FileNotFoundError(f"No snapshots in {snapshots_dir}")

    snapshot = snapshot_dirs[-1]  # most recent

    # Load config
    config_path = snapshot / 'config.json'
    if not config_path.exists():
        raise FileNotFoundError(f"No config.json in {snapshot}")

    with open(config_path) as f:
        config = json.load(f)

    return snapshot, config


def detect_architecture(config: dict) -> str:
    """Detect architecture family from config.json."""
    model_type = config.get('model_type', '').lower()

    if model_type in MODEL_TYPE_MAP:
        return MODEL_TYPE_MAP[model_type]

    # Fallback: check for architecture hints
    architectures = config.get('architectures', [])
    for arch_name in architectures:
        arch_lower = arch_name.lower()
        if 'bert' in arch_lower:
            return 'bert'
        if 'gpt2' in arch_lower:
            return 'gpt2'
        if 'llama' in arch_lower or 'gemma' in arch_lower or 'mistral' in arch_lower:
            return 'llama'

    raise ValueError(
        f"Unknown model_type '{model_type}'. "
        f"Supported: {list(MODEL_TYPE_MAP.keys())}")


def find_safetensors(snapshot_dir: Path) -> list:
    """Find safetensors weight file(s). Returns list of paths (1 for single, N for sharded)."""
    st = snapshot_dir / 'model.safetensors'
    if st.exists():
        return [st]

    # Check for sharded weights
    shards = sorted(snapshot_dir.glob('model-*.safetensors'))
    if shards:
        return shards

    # Fallback to pytorch_model.bin
    pt = snapshot_dir / 'pytorch_model.bin'
    if pt.exists():
        raise NotImplementedError(
            f"Only safetensors format supported. Found pytorch_model.bin at {pt}. "
            "Re-download with safetensors support.")

    raise FileNotFoundError(f"No weight files found in {snapshot_dir}")


def _load_tensor(shard_files: list, key: str) -> np.ndarray:
    """Load a tensor by key from potentially sharded safetensors.

    Handles bfloat16 by loading via PyTorch and converting to float32.
    """
    for shard_path in shard_files:
        try:
            # Try numpy first (fast, works for float16/float32)
            from safetensors import safe_open
            with safe_open(str(shard_path), framework='numpy') as f:
                if key in f.keys():
                    return f.get_tensor(key).astype(np.float32)
        except (KeyError, TypeError):
            pass

        try:
            # Fallback to PyTorch (handles bfloat16)
            from safetensors import safe_open
            with safe_open(str(shard_path), framework='pt') as f:
                if key in f.keys():
                    import torch
                    return f.get_tensor(key).float().numpy()
        except KeyError:
            continue

    raise KeyError(f"Tensor '{key}' not found in any shard")


def _list_all_keys(shard_files: list) -> list:
    """List all tensor keys across all shards."""
    from safetensors import safe_open
    keys = []
    for shard_path in shard_files:
        try:
            with safe_open(str(shard_path), framework='pt') as f:
                keys.extend(f.keys())
        except Exception:
            pass
    return sorted(set(keys))


# ---------------------------------------------------------------------------
# Weight extraction
# ---------------------------------------------------------------------------

def list_weights(model_name: str):
    """List all weight tensors in a model."""
    snapshot, config = find_model_path(model_name)
    shard_files = find_safetensors(snapshot)
    arch_family = detect_architecture(config)
    arch = ARCHITECTURES[arch_family]

    n_layers = config.get(arch['n_layers_key'], '?')
    dim = config.get(arch['dim_key'], '?')
    n_kv_heads = config.get('num_key_value_heads', config.get('num_attention_heads', '?'))

    print(f"Model:    {model_name}")
    print(f"Type:     {config.get('model_type', '?')} → {arch_family}")
    print(f"Layers:   {n_layers}")
    print(f"Dim:      {dim}")
    print(f"Shards:   {len(shard_files)}")
    print(f"QKV:      {arch['qkv_style']}")
    if n_kv_heads != config.get('num_attention_heads'):
        print(f"GQA:      {config.get('num_attention_heads')} Q heads, {n_kv_heads} KV heads")
    print()

    from safetensors import safe_open
    total_params = 0
    for shard_path in shard_files:
        with safe_open(str(shard_path), framework='pt') as f:
            for key in sorted(f.keys()):
                tensor = f.get_tensor(key)
                total_params += tensor.numel()
                print(f"  {key:60s} {str(list(tensor.shape)):20s} {tensor.dtype}")
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Total size: {total_params * 4 / 1e6:.1f} MB (float32)")


def extract_weights(model_name: str, output_path: str,
                    layer: Optional[int] = None) -> dict:
    """Extract embeddings + Q/K/V weights from a HuggingFace model.

    Handles: single/sharded safetensors, bfloat16, GQA (grouped query attention).
    Returns metadata dict with extraction details.
    """
    t0 = time.time()
    snapshot, config = find_model_path(model_name)
    shard_files = find_safetensors(snapshot)
    arch_family = detect_architecture(config)
    arch = ARCHITECTURES[arch_family]

    n_layers = config[arch['n_layers_key']]
    dim = config[arch['dim_key']]

    # GQA detection: num_key_value_heads < num_attention_heads
    n_q_heads = config.get('num_attention_heads', 1)
    n_kv_heads = config.get('num_key_value_heads', n_q_heads)
    is_gqa = (n_kv_heads < n_q_heads)

    # Default: second-to-last layer (semantic + task-aware)
    if layer is None:
        layer = max(0, n_layers - 2)
    if layer < 0 or layer >= n_layers:
        raise ValueError(f"Layer {layer} out of range [0, {n_layers-1}]")

    print(f"Extracting from {model_name}")
    print(f"  Architecture: {arch_family}, {n_layers} layers, dim={dim}")
    if is_gqa:
        print(f"  GQA: {n_q_heads} Q heads, {n_kv_heads} KV heads "
              f"(will tile K/V {n_q_heads // n_kv_heads}x)")
    print(f"  Shards: {len(shard_files)}")
    print(f"  Using layer: {layer} (of 0-{n_layers-1})")

    os.makedirs(output_path, exist_ok=True)

    # 1. Token embeddings
    emb_key = arch['embedding_key']
    embeddings = _load_tensor(shard_files, emb_key)
    np.save(os.path.join(output_path, 'embeddings.npy'), embeddings)
    print(f"  Embeddings: {embeddings.shape} → embeddings.npy "
          f"({embeddings.nbytes / 1e6:.1f} MB)")

    # 2. Q/K/V weights for selected layer
    layer_prefix = arch['layer_prefix'].replace('{N}', str(layer))
    all_keys = _list_all_keys(shard_files)

    if arch['qkv_style'] == 'fused':
        # GPT-2: fused c_attn (dim, 3*dim) → split into Q, K, V
        fused_key = f'{layer_prefix}.{arch["fused_qkv_key"]}'
        fused_w = _load_tensor(shard_files, fused_key)
        W_Q = fused_w[:, :dim].copy()
        W_K = fused_w[:, dim:2*dim].copy()
        W_V = fused_w[:, 2*dim:].copy()

        fused_bias_key = f'{layer_prefix}.{arch["fused_qkv_bias"]}'
        if fused_bias_key in all_keys:
            fused_b = _load_tensor(shard_files, fused_bias_key)
            b_Q = fused_b[:dim].copy()
            b_K = fused_b[dim:2*dim].copy()
            b_V = fused_b[2*dim:].copy()
        else:
            b_Q = np.zeros(dim, dtype=np.float32)
            b_K = np.zeros(dim, dtype=np.float32)
            b_V = np.zeros(dim, dtype=np.float32)

    elif arch['qkv_style'] == 'separate':
        # BERT/Llama/Gemma: separate Q, K, V matrices
        W_Q = _load_tensor(shard_files, f'{layer_prefix}.{arch["q_key"]}')
        W_K = _load_tensor(shard_files, f'{layer_prefix}.{arch["k_key"]}')
        W_V = _load_tensor(shard_files, f'{layer_prefix}.{arch["v_key"]}')

        # GQA: tile K/V to match Q dimensions
        # e.g., Gemma 2B: Q=(2048,2048), K=(256,2048) → tile K to (2048,2048)
        if is_gqa and W_K.shape[0] < W_Q.shape[0]:
            tile_factor = W_Q.shape[0] // W_K.shape[0]
            print(f"  GQA tiling: K {W_K.shape} × {tile_factor} → ", end='')
            W_K = np.tile(W_K, (tile_factor, 1))
            W_V = np.tile(W_V, (tile_factor, 1))
            print(f"{W_K.shape}")

        # Biases (optional — Llama/Gemma don't have them)
        q_bias_key = f'{layer_prefix}.{arch.get("q_bias", "NONE")}'
        if arch.get('q_bias') and q_bias_key in all_keys:
            b_Q = _load_tensor(shard_files, q_bias_key)
            b_K = _load_tensor(shard_files, f'{layer_prefix}.{arch["k_bias"]}')
            b_V = _load_tensor(shard_files, f'{layer_prefix}.{arch["v_bias"]}')
            # Tile biases too if GQA
            if is_gqa and b_K.shape[0] < b_Q.shape[0]:
                tile_factor = b_Q.shape[0] // b_K.shape[0]
                b_K = np.tile(b_K, tile_factor)
                b_V = np.tile(b_V, tile_factor)
        else:
            b_Q = np.zeros(W_Q.shape[0], dtype=np.float32)
            b_K = np.zeros(W_K.shape[0], dtype=np.float32)
            b_V = np.zeros(W_V.shape[0], dtype=np.float32)

    # Save Q/K/V
    for name, arr in [('W_Q', W_Q), ('W_K', W_K), ('W_V', W_V),
                      ('b_Q', b_Q), ('b_K', b_K), ('b_V', b_V)]:
        np.save(os.path.join(output_path, f'{name}.npy'), arr)

    print(f"  W_Q: {W_Q.shape}, W_K: {W_K.shape}, W_V: {W_V.shape}")

    # 3. FFN weights (optional, for future use)
    ffn_up_key = f'{layer_prefix}.{arch["ffn_up_key"]}'
    ffn_down_key = f'{layer_prefix}.{arch["ffn_down_key"]}'
    if ffn_up_key in all_keys:
        ffn_up = _load_tensor(shard_files, ffn_up_key)
        ffn_down = _load_tensor(shard_files, ffn_down_key)
        np.save(os.path.join(output_path, 'FFN_up.npy'), ffn_up)
        np.save(os.path.join(output_path, 'FFN_down.npy'), ffn_down)
        print(f"  FFN: up={ffn_up.shape}, down={ffn_down.shape}")

    # 4. Save metadata
    elapsed = time.time() - t0
    meta = {
        'model_name': model_name,
        'model_type': config.get('model_type', ''),
        'architecture': arch_family,
        'n_layers': n_layers,
        'dim': dim,
        'layer_used': layer,
        'qkv_style': arch['qkv_style'],
        'gqa': is_gqa,
        'n_q_heads': n_q_heads,
        'n_kv_heads': n_kv_heads,
        'embedding_shape': list(embeddings.shape),
        'W_Q_shape': list(W_Q.shape),
        'extraction_time_s': round(elapsed, 2),
        'source_path': str(shard_files[0]),
    }
    with open(os.path.join(output_path, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"  Metadata → meta.json")
    print(f"  Done in {elapsed:.1f}s")

    return meta


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Extract weights from HuggingFace models for LMDB integration')
    parser.add_argument('--model', required=True,
                        help='HF model name (e.g., gpt2, sentence-transformers/all-MiniLM-L6-v2)')
    parser.add_argument('--output', default=None,
                        help='Output directory (default: ~/nexus-brain/model_weights/<model_short_name>)')
    parser.add_argument('--layer', type=int, default=None,
                        help='Layer to extract attention weights from (default: second-to-last)')
    parser.add_argument('--list', action='store_true',
                        help='List all weights in the model without extracting')

    args = parser.parse_args()

    if args.list:
        list_weights(args.model)
        return

    if args.output is None:
        # Derive short name: "sentence-transformers/all-MiniLM-L6-v2" → "all-MiniLM-L6-v2"
        short_name = args.model.split('/')[-1]
        args.output = os.path.expanduser(f'~/nexus-brain/model_weights/{short_name}')

    meta = extract_weights(args.model, args.output, layer=args.layer)
    print(f"\nExtracted to: {args.output}")
    print(f"Files: embeddings.npy, W_Q.npy, W_K.npy, W_V.npy, b_Q.npy, b_K.npy, b_V.npy, meta.json")


if __name__ == '__main__':
    main()
