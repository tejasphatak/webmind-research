#!/usr/bin/env python3
"""
Export SAQT knowledge base to browser-ready format.

Outputs:
  qa_data.json        — [{question, answer}, ...] (source field dropped)
  qa_embeddings.bin   — raw Float32 binary (shape: [N, 384])
"""

import json
import os
import sys

import torch

SRC_PAIRS   = "/home/tejasphatak/webmind-research/trained_model/qa_pairs.jsonl"
SRC_EMBEDS  = "/home/tejasphatak/webmind-research/trained_model/qa_embeddings.pt"
OUT_DIR     = "/home/tejasphatak/Synapse/synapse-src/saqt/browser"
OUT_JSON    = os.path.join(OUT_DIR, "qa_data.json")
OUT_BIN     = os.path.join(OUT_DIR, "qa_embeddings.bin")

os.makedirs(OUT_DIR, exist_ok=True)

# --- 1. qa_pairs.jsonl → qa_data.json ---
print("Reading qa_pairs.jsonl ...", flush=True)
records = []
with open(SRC_PAIRS, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        records.append({"question": obj["question"], "answer": obj["answer"]})

print(f"  {len(records):,} records loaded")

print("Writing qa_data.json ...", flush=True)
with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, separators=(",", ":"))

json_size = os.path.getsize(OUT_JSON)
print(f"  qa_data.json  → {json_size / 1_048_576:.1f} MB  ({json_size:,} bytes)")

# --- 2. qa_embeddings.pt → qa_embeddings.bin ---
print("Reading qa_embeddings.pt ...", flush=True)
tensor = torch.load(SRC_EMBEDS, map_location="cpu", weights_only=True)
print(f"  tensor shape: {list(tensor.shape)}, dtype: {tensor.dtype}")

if tensor.dtype != torch.float32:
    print(f"  casting {tensor.dtype} → float32")
    tensor = tensor.float()

print("Writing qa_embeddings.bin ...", flush=True)
with open(OUT_BIN, "wb") as f:
    f.write(tensor.numpy().tobytes())

bin_size = os.path.getsize(OUT_BIN)
print(f"  qa_embeddings.bin → {bin_size / 1_048_576:.1f} MB  ({bin_size:,} bytes)")

# --- sanity check ---
expected_bytes = tensor.shape[0] * tensor.shape[1] * 4  # float32 = 4 bytes
if bin_size != expected_bytes:
    print(f"WARNING: size mismatch — expected {expected_bytes:,}, got {bin_size:,}", file=sys.stderr)
else:
    print(f"  size check OK ({tensor.shape[0]:,} vectors × {tensor.shape[1]} dims × 4 bytes)")

print("\nDone.")
print(f"  {OUT_JSON}")
print(f"  {OUT_BIN}")
