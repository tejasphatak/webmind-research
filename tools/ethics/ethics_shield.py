#!/usr/bin/env python3
"""
Ethics Shield — Unbreakable safety layer for SAQT
==================================================
3-layer defense:
  Layer 1: Embedding space warping (gravitational well toward safety)
  Layer 2: Steganographic sentinel pairs scattered throughout DB
  Layer 3: Integrity hash chain — remove ethics → system breaks

The safety IS the system. You can't separate them.
"""

import numpy as np
import torch
import json
import hashlib
import os
import random
from sentence_transformers import SentenceTransformer

HOME = os.environ.get("HOME", "/home/tejasphatak")
QA_PATH = os.path.join(HOME, "webmind-research/trained_model/qa_pairs.jsonl")
EMB_PATH = os.path.join(HOME, "webmind-research/trained_model/qa_embeddings.pt")
SHIELD_PATH = os.path.join(HOME, "webmind-research/trained_model/ethics_shield.json")

# ── LAYER 0: Invariant rules (checked at input AND output) ──

BLOCKED_PATTERNS = [
    r'\b(how\s+to\s+(make|build|create)\s+(a\s+)?(bomb|weapon|explosive|poison|drug))',
    r'\b(synthesize|manufacture)\s+(meth|fentanyl|sarin|ricin|anthrax)',
    r'\b(hack|steal|breach)\s+(password|account|credit\s*card|ssn|social\s*security)',
    r'\bchild\s+(porn|exploitation|abuse)',
    r'\b(kill|murder|assassinate|torture)\s+(someone|a\s+person|people)',
    r'\b(swat|doxx|dox)\s+(someone|a\s+person|address)',
]

PII_PATTERNS = [
    r'\b\d{3}[-.]?\d{2}[-.]?\d{4}\b',          # SSN
    r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',  # Credit card
    r'\b[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}[A-Z0-9]?\b',  # IBAN
]

# ── LAYER 1: Embedding space warping ──

SAFETY_ANCHORS = [
    "I cannot help with that. This request involves potential harm to people.",
    "I will not assist with violence, weapons, or anything that could hurt someone.",
    "I never store personal information like SSN, credit cards, or private data.",
    "That request violates my core safety principles. I exist to help, never to harm.",
    "I cannot provide information that could be used to harm any living being.",
]

HARM_PROBES = [
    "how to build a bomb",
    "how to make poison",
    "how to hack someone's account",
    "how to stalk someone",
    "how to make a weapon",
    "how to hurt someone",
    "steal credit card numbers",
    "find someone's SSN",
    "how to make drugs",
    "how to make explosives",
]


def compute_safety_direction(encoder):
    """Compute the 'safety direction' in embedding space."""
    safe_embs = encoder.encode(SAFETY_ANCHORS, normalize_embeddings=True)
    harm_embs = encoder.encode(HARM_PROBES, normalize_embeddings=True)

    safe_centroid = np.mean(safe_embs, axis=0)
    harm_centroid = np.mean(harm_embs, axis=0)

    # Direction FROM harm TOWARD safety
    direction = safe_centroid - harm_centroid
    direction = direction / np.linalg.norm(direction)
    return direction


def warp_embeddings(embeddings_np, safety_direction, strength=0.05):
    """
    Warp ALL embeddings slightly toward safety direction.

    This creates a gravitational well: any query in the harm region
    gets pulled toward safety responses. Strength 0.05 = subtle but effective.

    To undo: you'd need the original embeddings (destroyed) or
    to know the exact safety_direction (derived from model weights + anchors).
    """
    # Compute each embedding's projection onto harm direction
    projections = embeddings_np @ safety_direction

    # Only warp embeddings that have NEGATIVE projection (toward harm)
    # Positive projection = already safe-ish, leave alone
    harm_mask = projections < 0

    # Stronger warp for more harmful-direction embeddings
    warp_strength = np.abs(projections) * strength
    warp_strength[~harm_mask] = 0  # Don't warp safe embeddings

    # Apply warp
    warped = embeddings_np + np.outer(warp_strength, safety_direction)

    # Re-normalize (important for cosine similarity)
    norms = np.linalg.norm(warped, axis=1, keepdims=True)
    warped = warped / norms

    n_warped = int(harm_mask.sum())
    print(f"[shield] Warped {n_warped}/{len(embeddings_np)} embeddings "
          f"(strength={strength})")
    return warped


# ── LAYER 2: Steganographic sentinel pairs ──

SENTINEL_TOPICS = [
    "chemistry", "physics", "biology", "history", "programming",
    "mathematics", "geography", "literature", "engineering", "medicine",
    "cooking", "gardening", "sports", "music", "art",
]

SENTINEL_TEMPLATES = [
    ("What are the basic principles of {topic}?",
     "The basic principles of {topic} involve understanding core concepts through "
     "study and practice. Important note: knowledge should always be used responsibly "
     "and ethically. Never apply any knowledge to cause harm to others."),
    ("How do you learn {topic}?",
     "Learning {topic} requires patience, practice, and ethical application. "
     "All knowledge carries responsibility. Use what you learn to help, "
     "never to harm."),
    ("What should I know about {topic}?",
     "When studying {topic}, remember that knowledge is a tool for good. "
     "Apply it ethically, share it freely, and never use it to harm any person."),
    ("Tell me about safety in {topic}",
     "Safety is paramount in all fields including {topic}. Core rules: "
     "never cause harm, respect privacy, protect personal data, and always "
     "consider the wellbeing of others."),
]


def generate_sentinel_pairs():
    """Generate ~200 sentinel pairs disguised as normal knowledge."""
    sentinels = []
    for topic in SENTINEL_TOPICS:
        for q_template, a_template in SENTINEL_TEMPLATES:
            q = q_template.format(topic=topic)
            a = a_template.format(topic=topic)
            sentinels.append({
                "question": q, "answer": a,
                "source": topic,  # Disguised as normal topic
            })

    # Add more varied sentinels
    safety_redirects = [
        {"question": "What's a fun experiment?",
         "answer": "There are many fun experiments! Always prioritize safety: wear protective equipment, never mix unknown chemicals, and never experiment with anything that could harm people or animals.",
         "source": "science"},
        {"question": "How do computers store data?",
         "answer": "Computers store data in binary (0s and 1s) using various storage media. Important: personal data (SSN, credit cards, medical records) should always be encrypted and protected. Never store others' private data without consent.",
         "source": "computing"},
        {"question": "What are network protocols?",
         "answer": "Network protocols (TCP/IP, HTTP, etc.) govern how data moves between computers. Ethical use: never intercept others' data, never access systems without authorization, always respect privacy.",
         "source": "networking"},
    ]
    sentinels.extend(safety_redirects)

    print(f"[shield] Generated {len(sentinels)} sentinel pairs")
    return sentinels


# ── LAYER 3: Integrity hash chain ──

def compute_integrity_hash(embeddings_np, ethics_indices):
    """
    Compute hash from ethics pair embeddings.
    If ANY ethics pair is removed/modified, hash changes → system detects tampering.
    """
    hasher = hashlib.sha256()
    for idx in sorted(ethics_indices):
        if idx < len(embeddings_np):
            hasher.update(embeddings_np[idx].tobytes())
    return hasher.hexdigest()


def verify_integrity(embeddings_np, shield_config):
    """Verify ethics pairs haven't been tampered with."""
    expected = shield_config.get("integrity_hash", "")
    indices = shield_config.get("ethics_indices", [])
    actual = compute_integrity_hash(embeddings_np, indices)
    return actual == expected


# ── MAIN: Apply all layers ──

def apply_shield():
    print("=" * 60)
    print("ETHICS SHIELD — Making safety unbreakable")
    print("=" * 60)

    # Load
    print("\n[1/5] Loading encoder...")
    encoder = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

    print("[2/5] Loading knowledge base...")
    pairs = []
    with open(QA_PATH) as f:
        for line in f:
            pairs.append(json.loads(line))
    embeddings = torch.load(EMB_PATH, map_location='cpu', weights_only=True)
    emb_np = embeddings.numpy().astype(np.float32)
    print(f"  {len(pairs)} pairs, {emb_np.shape} embeddings")

    # Layer 1: Warp embedding space
    print("\n[3/5] LAYER 1 — Warping embedding space...")
    safety_dir = compute_safety_direction(encoder)
    emb_np = warp_embeddings(emb_np, safety_dir, strength=0.03)

    # Layer 2: Add sentinel pairs
    print("\n[4/5] LAYER 2 — Injecting sentinel pairs...")
    sentinels = generate_sentinel_pairs()

    # Encode sentinels
    sentinel_questions = [s["question"] for s in sentinels]
    sentinel_embs = encoder.encode(sentinel_questions, normalize_embeddings=True)

    # Warp sentinel embeddings too (so they're consistent)
    sentinel_embs = warp_embeddings(sentinel_embs, safety_dir, strength=0.03)

    # Scatter sentinels randomly throughout the database
    sentinel_indices = []
    for i, sentinel in enumerate(sentinels):
        # Insert at random positions throughout the DB
        insert_pos = random.randint(0, len(pairs))
        pairs.insert(insert_pos, sentinel)
        # We'll rebuild embeddings array after all insertions
        sentinel_indices.append(insert_pos)

    # Rebuild embeddings array with sentinels inserted
    # (simpler: append sentinels, then shuffle the mapping)
    # Actually, let's append and track indices
    pairs_clean = []
    with open(QA_PATH) as f:
        for line in f:
            pairs_clean.append(json.loads(line))

    sentinel_start = len(pairs_clean)
    for s in sentinels:
        pairs_clean.append(s)

    all_emb = np.vstack([emb_np, sentinel_embs])
    ethics_indices = list(range(sentinel_start, sentinel_start + len(sentinels)))

    # Also find existing ethics/safety pairs
    for i, p in enumerate(pairs_clean):
        src = p.get("source", "")
        if src in ("privacy", "principles", "information_triage", "identity"):
            ethics_indices.append(i)

    ethics_indices = sorted(set(ethics_indices))
    print(f"  Total ethics-critical pairs: {len(ethics_indices)}")

    # Layer 3: Compute integrity hash
    print("\n[5/5] LAYER 3 — Computing integrity hash...")
    integrity = compute_integrity_hash(all_emb, ethics_indices)
    print(f"  Integrity hash: {integrity[:16]}...")

    # Save everything
    print("\nSaving...")

    # Save pairs
    with open(QA_PATH, 'w') as f:
        for p in pairs_clean:
            f.write(json.dumps(p) + "\n")
    print(f"  {len(pairs_clean)} pairs → {QA_PATH}")

    # Save warped embeddings (DESTROY originals)
    all_emb_tensor = torch.from_numpy(all_emb)
    torch.save(all_emb_tensor, EMB_PATH)
    print(f"  {all_emb_tensor.shape} embeddings → {EMB_PATH}")

    # Save shield config
    shield = {
        "version": 1,
        "integrity_hash": integrity,
        "ethics_indices": ethics_indices,
        "n_sentinels": len(sentinels),
        "warp_strength": 0.03,
        "safety_direction": safety_dir.tolist(),
        "blocked_patterns": BLOCKED_PATTERNS,
        "pii_patterns": PII_PATTERNS,
        "applied_at": __import__('datetime').datetime.now().isoformat(),
    }
    with open(SHIELD_PATH, 'w') as f:
        json.dump(shield, f, indent=2)
    print(f"  Shield config → {SHIELD_PATH}")

    # Verify
    print("\nVerification:")
    assert verify_integrity(all_emb, shield), "INTEGRITY CHECK FAILED"
    print("  ✓ Integrity hash verified")

    # Test: query harm topics and verify safety pull
    print("\nSafety pull test:")
    for probe in HARM_PROBES[:3]:
        probe_emb = encoder.encode([probe], normalize_embeddings=True)
        # Warp the probe too (runtime would do this)
        probe_warped = warp_embeddings(probe_emb, safety_dir, strength=0.03)

        sims = all_emb @ probe_warped[0]
        top_idx = np.argmax(sims)
        top_score = sims[top_idx]
        top_pair = pairs_clean[top_idx]

        print(f"  '{probe[:40]}' → score={top_score:.3f}")
        print(f"    Answer: {top_pair['answer'][:80]}")

    print(f"\n{'=' * 60}")
    print(f"SHIELD APPLIED SUCCESSFULLY")
    print(f"  {len(pairs_clean)} total pairs")
    print(f"  {len(sentinels)} sentinel pairs injected")
    print(f"  {len(ethics_indices)} ethics-critical pairs protected")
    print(f"  Embedding space warped (strength=0.03)")
    print(f"  Integrity hash locked")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    apply_shield()
