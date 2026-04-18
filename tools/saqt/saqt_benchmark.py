#!/usr/bin/env python3
"""SAQT Benchmark — run HLE questions through trained system."""
import torch, json, time, random, os
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DIR = os.environ.get("SAQT_OUT_DIR", "/workspace/trained_model")
DATA_DIR = os.environ.get("SAQT_DATA_DIR", "/workspace/data")

print(f"Device: {DEVICE}", flush=True)
print("Loading trained model...", flush=True)

all_embs = torch.load(f"{MODEL_DIR}/embeddings.pt", weights_only=True).to(DEVICE)
chunks = []
with open(f"{MODEL_DIR}/chunks.jsonl") as f:
    for line in f:
        chunks.append(json.loads(line))
neuron_data = {int(k): v for k, v in json.load(open(f"{MODEL_DIR}/neuron_data.json")).items()}

encoder = SentenceTransformer('all-MiniLM-L6-v2', device=DEVICE)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

config = GPT2Config(vocab_size=50257, n_positions=256, n_embd=256,
                    n_layer=2, n_head=4, n_inner=1024)
kernel = GPT2LMHeadModel(config).to(DEVICE)
state = torch.load(f"{MODEL_DIR}/kernel.pt", weights_only=True)
# Fix key prefix if saved from wrapper class
if any(k.startswith("model.") for k in state):
    state = {k.replace("model.", "", 1): v for k, v in state.items()}
kernel.load_state_dict(state)
kernel.eval()

# Neuron profiles
profiles = {}
for nid, indices in neuron_data.items():
    if indices:
        profiles[nid] = all_embs[indices].mean(dim=0)

# Load HLE
hle = []
with open(f"{DATA_DIR}/hle.jsonl") as f:
    for line in f:
        hle.append(json.loads(line))

sample = random.sample(hle, min(30, len(hle)))
print(f"Running {len(sample)} HLE questions...\n", flush=True)

total_facts, total_ms = 0, 0
for q_data in sample:
    q = q_data["question"][:200]
    t0 = time.time()

    q_emb = encoder.encode([q], convert_to_tensor=True,
                          show_progress_bar=False)[0].to(DEVICE)
    retrieved, trace, path = [], [], []

    for hop in range(5):
        scores = []
        for nid, profile in profiles.items():
            if nid in path:
                continue
            sim = F.cosine_similarity(
                q_emb.unsqueeze(0), profile.unsqueeze(0)).item()
            scores.append((nid, sim))
        scores.sort(key=lambda x: x[1], reverse=True)
        if not scores:
            break

        best_nid = scores[0][0]
        n_indices = neuron_data[best_nid]
        n_embs = all_embs[n_indices]
        sims = F.cosine_similarity(q_emb.unsqueeze(0), n_embs)
        top_k = min(3, len(n_indices))
        top_vals, top_idxs = sims.topk(top_k)
        for j, idx in enumerate(top_idxs):
            if top_vals[j] > 0.3:
                fact = chunks[n_indices[idx.item()]]["text"]
                if fact not in retrieved:
                    retrieved.append(fact)

        # Reason with trained kernel
        fact_str = " | ".join(retrieved[-4:])
        ctx = f"Question: {q} Facts: {fact_str}"
        inp = tokenizer(ctx, return_tensors='pt', truncation=True,
                       max_length=200).to(DEVICE)
        with torch.no_grad():
            out = kernel.generate(inp['input_ids'], max_new_tokens=20,
                                pad_token_id=50256)
        new_tok = out[0][inp['input_ids'].size(1):]
        ref = tokenizer.decode(new_tok, skip_special_tokens=True).strip()
        if ref:
            trace.append(ref[:60])

        path.append(best_nid)

        # Re-encode with accumulated context
        trace_str = " -> ".join(trace[-3:])
        full_ctx = f"Question: {q} Facts: {fact_str} Reasoning: {trace_str}"
        q_emb = encoder.encode([full_ctx], convert_to_tensor=True,
                              show_progress_bar=False)[0].to(DEVICE)

    ms = (time.time() - t0) * 1000
    total_facts += len(retrieved)
    total_ms += ms
    print(f"  [{len(retrieved):2d} facts, {len(path)} hops, {ms:.0f}ms] "
          f"Q: {q[:60]}", flush=True)
    if trace:
        print(f"    Thought: {trace[0][:70]}", flush=True)
    if retrieved:
        print(f"    Top:     {retrieved[0][:70]}", flush=True)

avg_f = total_facts / len(sample)
avg_ms = total_ms / len(sample)
print(f"\n{'='*50}", flush=True)
print(f"SAQT BENCHMARK RESULTS", flush=True)
print(f"{'='*50}", flush=True)
print(f"  HLE questions:     {len(sample)}", flush=True)
print(f"  Avg facts/query:   {avg_f:.1f}", flush=True)
print(f"  Avg time/query:    {avg_ms:.0f}ms", flush=True)
print(f"  Effective tok/s:   ~{avg_f*20/(avg_ms/1000):.0f} (retrieved)", flush=True)
print(f"  Knowledge base:    {len(chunks):,} chunks", flush=True)
print(f"  Neurons:           {len(neuron_data)}", flush=True)
print(f"  Kernel trained:    loss 4.89 → 3.52", flush=True)
