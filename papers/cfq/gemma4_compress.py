import torch
import torch.nn as nn
from safetensors import safe_open
from safetensors.torch import save_file
import os
import json
import numpy as np
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

# --- COMPRESSION LOGIC ---
def compress_tensor(tensor):
    """
    Compresses a single tensor using 'Carrier + Fraction'.
    Uses GPU if available for the math.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    W = tensor.to(device).float()
    n = W.numel()
    
    # Architecture: 32-Block Carrier + 1-bit Directional Fraction
    block_size = 32
    pad = 0
    if n % block_size != 0:
        pad = block_size - (n % block_size)
        W_flat = torch.cat([W.flatten(), torch.zeros(pad, device=device)])
    else:
        W_flat = W.flatten()
        
    W_blocks = W_flat.view(-1, block_size)
    carriers = W_blocks.mean(dim=1).half().cpu()
    
    # Delta Layer: 1-bit Fraction
    deltas = W_blocks - carriers.to(device).unsqueeze(1).float()
    # We use a 50% sparsity for the 'Fractional Pieces' to stay ultra-compact
    # But for this demo, we'll store a 1-bit sign for ALL weights (1 bit/param)
    amp = deltas.abs().mean().half().cpu()
    signs_bits = np.packbits((deltas > 0).cpu().numpy())
    
    return {
        'carriers': carriers,
        'signs': signs_bits,
        'amplitude': amp,
        'pad': pad,
        'shape': list(tensor.shape)
    }

def process_shard(shard_info):
    shard_path, keys, output_dir = shard_info
    results = {}
    with safe_open(shard_path, framework="pt", device="cpu") as f:
        for key in keys:
            tensor = f.get_tensor(key)
            if "weight" in key and len(tensor.shape) == 2 and any(x in key for x in ["mlp", "self_attn"]):
                results[key] = compress_tensor(tensor)
            else:
                # Keep small tensors and embeddings as is
                results[key] = tensor.half()
    
    # Save the compressed shard
    shard_name = os.path.basename(shard_path).replace(".safetensors", "_compressed.safetensors")
    # Note: safetensors doesn't support nested dicts, so we'll flatten or save as separate files
    # For simplicity in this demo, we'll save each compressed component with a suffix
    flat_results = {}
    for k, v in results.items():
        if isinstance(v, dict):
            flat_results[f"{k}.carriers"] = v['carriers']
            flat_results[f"{k}.signs"] = torch.from_numpy(v['signs'])
            flat_results[f"{k}.amplitude"] = v['amplitude'].view(1)
            # metadata will be stored in a separate json
        else:
            flat_results[k] = v
            
    out_path = os.path.join(output_dir, shard_name)
    save_file(flat_results, out_path)
    return out_path

if __name__ == "__main__":
    input_dir = "/workspace/gemma-4"
    output_dir = "/workspace/gemma-4-compressed"
    os.makedirs(output_dir, exist_ok=True)
    
    index_path = os.path.join(input_dir, "model.safetensors.index.json")
    with open(index_path) as f:
        weight_map = json.load(f)["weight_map"]
        
    shard_to_keys = {}
    for k, v in weight_map.items():
        shard_to_keys.setdefault(v, []).append(k)
        
    tasks = []
    for shard_name, keys in shard_to_keys.items():
        tasks.append((os.path.join(input_dir, shard_name), keys, output_dir))
        
    print(f"Starting parallel compression of {len(tasks)} shards...")
    t0 = time.time()
    
    # Use multiple processes to speed up CPU-bound packing
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_shard, t) for t in tasks]
        for future in as_completed(futures):
            print(f"  Finished: {future.result()}")
            
    print(f"Total compression time: {time.time()-t0:.1f}s")
