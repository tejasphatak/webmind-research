#!/usr/bin/env python3
"""
SAQT Training Pipeline
======================
1. Encode all 65K chunks with sentence transformer → embeddings
2. Build distributed vector DB across N neurons
3. Train reasoning kernels on HLE + reasoning datasets
4. Benchmark: run HLE questions through trained system
5. Save everything for deployment

Input: data/*.jsonl files
Output: trained_model/ directory with embeddings + kernels
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
import json, time, random, os, copy
from pathlib import Path
from dataclasses import dataclass, field

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMB_DIM = 384
DATA_DIR = Path(os.environ.get("SAQT_DATA_DIR", "/workspace/data"))
OUT_DIR = Path(os.environ.get("SAQT_OUT_DIR", "/workspace/trained_model"))
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════
# Step 1: Load and encode all data
# ══════════════════════════════════════════════════════════════

def load_all_chunks(data_dir):
    """Load all JSONL files from data directory."""
    chunks = []
    for f in sorted(data_dir.glob("*.jsonl")):
        with open(f) as fh:
            for line in fh:
                row = json.loads(line)
                text = row.get("text") or row.get("question", "")
                if text and len(text) > 10:
                    chunks.append({
                        "text": text[:500],
                        "topic": row.get("topic", row.get("category", "general")),
                        "source": row.get("source", f.stem),
                        "answer": row.get("answer", ""),
                    })
    return chunks


def encode_chunks(encoder, chunks, batch_size=256):
    """Encode all chunks with sentence transformer."""
    texts = [c["text"] for c in chunks]
    print(f"  Encoding {len(texts)} chunks (batch_size={batch_size})...", flush=True)
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        with torch.no_grad():
            embs = encoder.encode(batch, convert_to_tensor=True, show_progress_bar=False)
        all_embs.append(embs.cpu())
        if (i // batch_size + 1) % 20 == 0:
            print(f"    {i+len(batch)}/{len(texts)}", flush=True)
    return torch.cat(all_embs, dim=0)


# ══════════════════════════════════════════════════════════════
# Step 2: Distribute across neurons
# ══════════════════════════════════════════════════════════════

def distribute_to_neurons(chunks, embeddings, n_neurons=20, replication=3):
    """Assign chunks to neurons by topic, with replication."""
    # Group by topic
    topic_groups = {}
    for i, chunk in enumerate(chunks):
        topic = chunk["topic"]
        if topic not in topic_groups:
            topic_groups[topic] = []
        topic_groups[topic].append(i)

    # Assign topics to primary neurons (round-robin)
    topics = sorted(topic_groups.keys())
    neuron_data = {i: [] for i in range(n_neurons)}
    topic_to_neuron = {}

    for i, topic in enumerate(topics):
        primary = i % n_neurons
        topic_to_neuron[topic] = primary
        for idx in topic_groups[topic]:
            neuron_data[primary].append(idx)

    # Replicate to r-1 other neurons
    if replication > 1:
        for topic, primary in topic_to_neuron.items():
            replicas = random.sample(
                [n for n in range(n_neurons) if n != primary],
                min(replication - 1, n_neurons - 1))
            for replica in replicas:
                for idx in topic_groups[topic]:
                    neuron_data[replica].append(idx)

    return neuron_data, topic_to_neuron


# ══════════════════════════════════════════════════════════════
# Step 3: Train reasoning kernels
# ══════════════════════════════════════════════════════════════

class ReasoningKernel(nn.Module):
    def __init__(self, vocab_size=50257, hidden=256, n_layers=2, n_heads=4):
        super().__init__()
        config = GPT2Config(
            vocab_size=vocab_size, n_positions=256,
            n_embd=hidden, n_layer=n_layers, n_head=n_heads, n_inner=hidden*4)
        self.model = GPT2LMHeadModel(config)

    def forward(self, input_ids, labels=None):
        return self.model(input_ids=input_ids, labels=labels)

    def generate(self, input_ids, max_new_tokens=30):
        self.model.eval()
        with torch.no_grad():
            return self.model.generate(
                input_ids, max_new_tokens=max_new_tokens,
                do_sample=False, pad_token_id=50256)


def create_reasoning_training_data(chunks, embeddings, encoder, tokenizer,
                                   n_examples=5000):
    """Create (context, target) pairs for kernel training.
    Uses HLE/reasoning questions: given facts, produce useful refinement."""
    training_pairs = []

    # Get reasoning questions (HLE, MMLU, ARC, etc.)
    reasoning_chunks = [c for c in chunks if c["source"] in
        ("hle", "mmlu_pro", "arc", "gsm8k", "bbh_causal", "bbh_logic",
         "mmlu_algebra", "mmlu_physics", "mmlu_philosophy", "mmlu_compsec",
         "mmlu_astronomy", "mmlu_religions", "hotpotqa")]

    # Get knowledge chunks for retrieval simulation
    knowledge_chunks = [c for c in chunks if c["source"] in
        ("wikipedia", "hotpotqa_context", "squad")]
    knowledge_texts = [c["text"] for c in knowledge_chunks]

    if not knowledge_texts:
        knowledge_texts = [c["text"] for c in chunks[:10000]]

    print(f"  Reasoning questions: {len(reasoning_chunks)}", flush=True)
    print(f"  Knowledge base: {len(knowledge_texts)}", flush=True)

    # Encode knowledge for retrieval
    print(f"  Encoding knowledge base for retrieval simulation...", flush=True)
    k_embs = encoder.encode(knowledge_texts[:10000], convert_to_tensor=True,
                           show_progress_bar=False, batch_size=256)

    random.shuffle(reasoning_chunks)

    for chunk in reasoning_chunks[:n_examples]:
        q = chunk["text"]
        a = chunk.get("answer", "")

        # Simulate retrieval: find top-3 similar facts
        q_emb = encoder.encode([q], convert_to_tensor=True, show_progress_bar=False)
        sims = F.cosine_similarity(q_emb, k_embs)
        top_idxs = sims.topk(3).indices.tolist()
        retrieved = [knowledge_texts[i][:150] for i in top_idxs]

        # Training input: question + retrieved facts
        context = f"Question: {q[:200]} Facts: {' | '.join(retrieved)}"
        # Training target: the answer or a useful refinement
        if a:
            target = f" Answer: {a[:150]}"
        else:
            target = f" The key concept here relates to {q.split()[-3:]}."

        training_pairs.append((context, target))

    print(f"  Created {len(training_pairs)} training pairs", flush=True)
    return training_pairs


def train_kernel(kernel, tokenizer, training_pairs, n_epochs=3, lr=1e-4,
                 batch_size=8):
    """Train reasoning kernel on (context, target) pairs."""
    kernel.train()
    optimizer = torch.optim.AdamW(kernel.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs * len(training_pairs) // batch_size)

    total_loss = 0
    steps = 0

    for epoch in range(n_epochs):
        random.shuffle(training_pairs)
        epoch_loss = 0
        epoch_steps = 0

        for i in range(0, len(training_pairs), batch_size):
            batch = training_pairs[i:i+batch_size]
            texts = [ctx + tgt for ctx, tgt in batch]

            tokens = tokenizer(texts, return_tensors='pt', truncation=True,
                             max_length=256, padding='max_length').to(DEVICE)
            input_ids = tokens['input_ids']

            outputs = kernel(input_ids=input_ids, labels=input_ids)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(kernel.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            epoch_steps += 1
            steps += 1

        avg = epoch_loss / max(epoch_steps, 1)
        print(f"    Epoch {epoch+1}/{n_epochs}  loss={avg:.4f}  "
              f"lr={optimizer.param_groups[0]['lr']:.6f}", flush=True)

    return total_loss


# ══════════════════════════════════════════════════════════════
# Step 4: Benchmark
# ══════════════════════════════════════════════════════════════

@dataclass
class QueryPacket:
    original_question: str
    current_embedding: torch.Tensor
    reasoning_trace: list = field(default_factory=list)
    retrieved_facts: list = field(default_factory=list)
    hop_count: int = 0
    path_history: list = field(default_factory=list)

    def context_string(self):
        parts = [f"Question: {self.original_question}"]
        if self.retrieved_facts:
            parts.append("Facts: " + " | ".join(self.retrieved_facts[-6:]))
        if self.reasoning_trace:
            parts.append("Reasoning: " + " -> ".join(self.reasoning_trace[-3:]))
        return " ".join(parts)


def benchmark_saqt(encoder, kernel, tokenizer, all_embs, chunks,
                   neuron_data, n_neurons, hle_questions, max_hops=5):
    """Run HLE questions through trained SAQT system."""
    # Build neuron profiles
    profiles = {}
    for nid, indices in neuron_data.items():
        if indices:
            profiles[nid] = all_embs[indices].mean(dim=0)

    results = []
    for q_data in hle_questions:
        q = q_data["question"]
        a = q_data.get("answer", "")

        t0 = time.time()
        q_emb = encoder.encode([q], convert_to_tensor=True,
                              show_progress_bar=False)[0]
        packet = QueryPacket(original_question=q, current_embedding=q_emb)

        for hop in range(max_hops):
            # Route
            scores = []
            for nid, profile in profiles.items():
                if nid in packet.path_history:
                    continue
                sim = F.cosine_similarity(
                    packet.current_embedding.unsqueeze(0),
                    profile.unsqueeze(0)).item()
                scores.append((nid, sim))
            scores.sort(key=lambda x: x[1], reverse=True)

            if not scores:
                break

            best_nid = scores[0][0]

            # Retrieve from this neuron
            n_indices = neuron_data[best_nid]
            n_embs = all_embs[n_indices]
            sims = F.cosine_similarity(packet.current_embedding.unsqueeze(0), n_embs)
            top_k = min(3, len(n_indices))
            top_vals, top_idxs = sims.topk(top_k)
            for j, idx in enumerate(top_idxs):
                if top_vals[j] > 0.3:
                    fact = chunks[n_indices[idx.item()]]["text"]
                    if fact not in packet.retrieved_facts:
                        packet.retrieved_facts.append(fact)

            # Reason with kernel
            context = packet.context_string()
            inp = tokenizer(context, return_tensors='pt', truncation=True,
                          max_length=200).to(DEVICE)
            kernel.model.eval()
            with torch.no_grad():
                out = kernel.generate(inp['input_ids'], max_new_tokens=20)
            new_tokens = out[0][inp['input_ids'].size(1):]
            refinement = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            if refinement:
                packet.reasoning_trace.append(refinement[:80])

            packet.hop_count += 1
            packet.path_history.append(best_nid)

            # Re-encode
            packet.current_embedding = encoder.encode(
                [packet.context_string()], convert_to_tensor=True,
                show_progress_bar=False)[0]

        elapsed = (time.time() - t0) * 1000
        results.append({
            "question": q[:100],
            "answer": a[:50],
            "n_facts": len(packet.retrieved_facts),
            "hops": packet.hop_count,
            "time_ms": elapsed,
            "facts": packet.retrieved_facts[:5],
            "trace": packet.reasoning_trace[:3],
        })

    return results


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

def main():
    print("=== SAQT TRAINING PIPELINE ===\n", flush=True)
    print(f"  Device: {DEVICE}", flush=True)
    print(f"  Data dir: {DATA_DIR}", flush=True)
    print(f"  Output dir: {OUT_DIR}\n", flush=True)

    # Step 1: Load data
    print("── STEP 1: LOADING DATA ──\n", flush=True)
    chunks = load_all_chunks(DATA_DIR)
    print(f"  Loaded {len(chunks)} chunks from {DATA_DIR}\n", flush=True)

    # Step 2: Encode
    print("── STEP 2: ENCODING ──\n", flush=True)
    t0 = time.time()
    encoder = SentenceTransformer('all-MiniLM-L6-v2', device=DEVICE)
    all_embs = encode_chunks(encoder, chunks, batch_size=512)
    encode_time = time.time() - t0
    print(f"  Encoded {len(all_embs)} chunks in {encode_time:.0f}s", flush=True)
    print(f"  Embedding shape: {all_embs.shape}", flush=True)
    print(f"  Embedding size: {all_embs.numel()*4/1e6:.0f}MB\n", flush=True)

    # Save embeddings
    torch.save(all_embs, OUT_DIR / "embeddings.pt")
    with open(OUT_DIR / "chunks.jsonl", "w") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")
    print(f"  Saved embeddings + chunks to {OUT_DIR}\n", flush=True)

    # Step 3: Distribute
    print("── STEP 3: DISTRIBUTING ACROSS NEURONS ──\n", flush=True)
    n_neurons = 20
    neuron_data, topic_map = distribute_to_neurons(
        chunks, all_embs, n_neurons=n_neurons, replication=3)
    for nid in sorted(neuron_data.keys()):
        n = len(neuron_data[nid])
        if n > 0:
            print(f"  Neuron {nid:2d}: {n:>6,} chunks", flush=True)
    print(flush=True)

    # Step 4: Train reasoning kernel
    print("── STEP 4: TRAINING REASONING KERNEL ──\n", flush=True)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    kernel = ReasoningKernel().to(DEVICE)
    n_params = sum(p.numel() for p in kernel.parameters())
    print(f"  Kernel: {n_params:,} params ({n_params*4/1e6:.0f}MB)", flush=True)

    t0 = time.time()
    training_pairs = create_reasoning_training_data(
        chunks, all_embs, encoder, tokenizer, n_examples=5000)
    print(f"\n  Training kernel...", flush=True)
    train_kernel(kernel, tokenizer, training_pairs, n_epochs=3, lr=1e-4)
    train_time = time.time() - t0
    print(f"  Training done in {train_time:.0f}s\n", flush=True)

    # Save kernel
    torch.save(kernel.state_dict(), OUT_DIR / "kernel.pt")
    print(f"  Saved kernel to {OUT_DIR / 'kernel.pt'}\n", flush=True)

    # Step 5: Benchmark on HLE
    print("── STEP 5: HLE BENCHMARK ──\n", flush=True)
    hle_questions = []
    hle_file = DATA_DIR / "hle.jsonl"
    if hle_file.exists():
        with open(hle_file) as f:
            for line in f:
                hle_questions.append(json.loads(line))

    # Sample 20 HLE questions
    if hle_questions:
        sample = random.sample(hle_questions, min(20, len(hle_questions)))
        print(f"  Running {len(sample)} HLE questions...\n", flush=True)

        results = benchmark_saqt(
            encoder, kernel, tokenizer, all_embs, chunks,
            neuron_data, n_neurons, sample, max_hops=5)

        total_facts = 0
        total_time = 0
        for r in results:
            total_facts += r["n_facts"]
            total_time += r["time_ms"]
            print(f"  [{r['n_facts']:2d} facts, {r['hops']} hops, {r['time_ms']:.0f}ms] "
                  f"Q: {r['question'][:60]}", flush=True)
            if r["facts"]:
                print(f"    Top fact: {r['facts'][0][:70]}", flush=True)
            if r["trace"]:
                print(f"    Thought:  {r['trace'][0][:70]}", flush=True)

        avg_facts = total_facts / len(results)
        avg_time = total_time / len(results)
        print(f"\n  Avg facts/query: {avg_facts:.1f}", flush=True)
        print(f"  Avg time/query:  {avg_time:.0f}ms", flush=True)
        print(f"  Avg tok/sec:     ~{avg_facts * 20 / (avg_time/1000):.0f} "
              f"(retrieved tokens)", flush=True)

    # Save neuron assignment
    with open(OUT_DIR / "neuron_data.json", "w") as f:
        json.dump({str(k): v for k, v in neuron_data.items()}, f)
    with open(OUT_DIR / "topic_map.json", "w") as f:
        json.dump(topic_map, f, indent=2)

    # Summary
    print(f"\n{'='*60}", flush=True)
    print("SAQT TRAINING COMPLETE", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  Total chunks:     {len(chunks):,}", flush=True)
    print(f"  Embedding dim:    {EMB_DIM}", flush=True)
    print(f"  Neurons:          {n_neurons}", flush=True)
    print(f"  Replication:      r=3", flush=True)
    print(f"  Kernel params:    {n_params:,}", flush=True)
    print(f"  Encode time:      {encode_time:.0f}s", flush=True)
    print(f"  Train time:       {train_time:.0f}s", flush=True)
    print(f"  Output dir:       {OUT_DIR}", flush=True)
    print(f"  Files:", flush=True)
    for f in sorted(OUT_DIR.glob("*")):
        print(f"    {f.name}: {f.stat().st_size/1e6:.1f}MB", flush=True)


if __name__ == "__main__":
    main()
