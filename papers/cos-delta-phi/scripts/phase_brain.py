"""
Phase-Brain: Phase Interference + Growing Knowledge Base
=========================================================

The convergence of two systems:
1. Phase interference model (585KB, CPU) — the THINKING module
2. BrainV2 (LMDB + embeddings) — the KNOWLEDGE module

How it works:
  TEACH: "Gravity pulls objects toward earth"
    → MiniLM encodes → stored in LMDB with embedding
    → Immediately available for retrieval

  ASK: "What is gravity?"
    → MiniLM encodes query → cosine search KB → retrieve top-K sentences
    → Format as context: "Context: {retrieved}. Question: {query}. Answer:"
    → Feed to phase decoder → generate char-by-char
    → If no good match in KB → "I don't know"

  The model LEARNS AS YOU SPEAK because:
    - Teaching adds to the KB (instant, no retraining)
    - Phase model is frozen — it's a fixed thinking process
    - More knowledge = better retrieval = better answers
    - The database IS the model
"""

import torch
import os
import sys
import numpy as np
import time

# Add paths
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'papers', 'new-gen-ai', 'src'))

from phase_decoder_test import PhaseDecoder, CharTokenizer


def p(msg):
    print(msg, flush=True)


class PhaseBrain:
    """Phase interference + growing knowledge base.

    The phase model is the thinking. The KB is the knowledge.
    Teach → KB grows → answers improve. No retraining.
    """

    def __init__(self, phase_model_path=None, kb_path=None):
        # --- Phase decoder (frozen thinking module) ---
        if phase_model_path is None:
            phase_model_path = os.path.join(os.path.dirname(__file__), 'phase_decoder_model.pth')

        self.char_tok = CharTokenizer()

        if os.path.exists(phase_model_path):
            checkpoint = torch.load(phase_model_path, weights_only=False)
            config = checkpoint['config']
            self.phase_model = PhaseDecoder(
                vocab_size=config['vocab_size'],
                embed_dim=config['embed_dim'],
                max_seq_len=config['max_seq_len']
            )
            self.phase_model.load_state_dict(checkpoint['model_state_dict'])
            self.phase_model.eval()
            p(f"Phase model loaded ({sum(pp.numel() for pp in self.phase_model.parameters()):,} params)")
        else:
            p(f"WARNING: No phase model at {phase_model_path}")
            self.phase_model = None

        # --- Knowledge base (growing, learns from conversation) ---
        # Lightweight KB — just embeddings + sentences, no full BrainV2 dependency
        self._encoder = None
        self._sentences = []
        self._embeddings = None  # (N, 384) float32
        self._qa_pairs = {}  # question → answer (direct mappings)

        # Load existing KB if provided
        if kb_path and os.path.exists(kb_path):
            self._load_kb(kb_path)

        p(f"Knowledge base: {len(self._sentences)} sentences")

    def _get_encoder(self):
        """Lazy-load MiniLM."""
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer('all-MiniLM-L6-v2')
            p("MiniLM encoder loaded")
        return self._encoder

    def _encode(self, text):
        """Encode text to 384d embedding."""
        return self._get_encoder().encode([text], normalize_embeddings=True,
                                           show_progress_bar=False).astype(np.float32)[0]

    # --- TEACH (knowledge grows) ---

    def teach(self, sentence):
        """Teach a fact. Instantly available for retrieval."""
        sentence = sentence.strip()
        if len(sentence) < 5:
            return "Too short to learn."

        embedding = self._encode(sentence)
        self._sentences.append(sentence)

        if self._embeddings is None:
            self._embeddings = embedding.reshape(1, -1)
        else:
            self._embeddings = np.vstack([self._embeddings, embedding.reshape(1, -1)])

        return f"Learned: \"{sentence}\" (KB size: {len(self._sentences)})"

    def teach_qa(self, question, answer):
        """Teach a direct Q→A pair."""
        self._qa_pairs[question.strip().lower()] = answer.strip()
        # Also teach the answer as a sentence
        self.teach(answer)
        return f"Learned Q→A: \"{question}\" → \"{answer}\""

    # --- RETRIEVE (search KB) ---

    def retrieve(self, query, top_k=3):
        """Search KB for relevant sentences."""
        if self._embeddings is None or len(self._embeddings) == 0:
            return []

        query_emb = self._encode(query)
        sims = self._embeddings @ query_emb
        k = min(top_k, len(sims))
        top_idx = np.argsort(-sims)[:k]

        results = []
        for idx in top_idx:
            results.append({
                "text": self._sentences[idx],
                "similarity": float(sims[idx]),
            })
        return results

    # --- ASK (retrieve + think) ---

    def ask(self, question):
        """Ask a question. Retrieves context from KB, generates with phase model."""

        # Tier 0: direct Q→A lookup
        qkey = question.strip().lower()
        if qkey in self._qa_pairs:
            return {
                "answer": self._qa_pairs[qkey],
                "strategy": "qa_direct",
                "context": [],
            }

        # Tier 1: retrieve relevant context
        context = self.retrieve(question, top_k=3)

        if not context or context[0]["similarity"] < 0.2:
            return {
                "answer": "I don't know.",
                "strategy": "abstain",
                "context": context,
            }

        # Tier 2: return best retrieval match
        # Phase generation is trained on TinyStories (PPL 5.1), not Q&A.
        # Until we train a Q&A phase decoder, retrieval IS the answer.
        # The phase model will be used for reasoning once trained on dialogue.
        if context[0]["similarity"] > 0.4:
            return {
                "answer": context[0]["text"],
                "strategy": "retrieval",
                "context": context,
            }

        # Tier 3: generate with phase model using retrieved context (experimental)
        if self.phase_model is not None and False:  # disabled until Q&A training
            # Build prompt: context + question
            context_text = ". ".join(c["text"] for c in context[:2])
            prompt = f"{context_text}. {question} "

            # Truncate to fit model's max sequence length (leave room for generation)
            max_seq = 128
            max_gen = 40
            max_prompt_chars = max_seq - max_gen
            if len(prompt) > max_prompt_chars:
                prompt = prompt[-max_prompt_chars:]

            seed = torch.tensor([self.char_tok.encode(prompt)], dtype=torch.long)
            with torch.no_grad():
                generated = self.phase_model.generate(seed, max_new_tokens=max_gen,
                                                       temperature=0.7, top_p=0.9)
            full_text = self.char_tok.decode(generated[0].tolist())
            answer = full_text[len(prompt):].strip()

            # Clean up: take first sentence
            for stop in ['.', '!', '?', '\n']:
                if stop in answer:
                    answer = answer[:answer.index(stop) + 1]
                    break

            return {
                "answer": answer if answer else context[0]["text"],
                "strategy": "phase_generate",
                "context": context,
                "prompt_used": prompt,
            }

        # Tier 3: fallback to best retrieval match
        return {
            "answer": context[0]["text"],
            "strategy": "retrieval_only",
            "context": context,
        }

    # --- SAVE/LOAD ---

    def save_kb(self, path):
        """Save knowledge base to disk."""
        import json
        data = {
            "sentences": self._sentences,
            "qa_pairs": self._qa_pairs,
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        # Save embeddings separately (numpy)
        if self._embeddings is not None:
            np.save(path + '.npy', self._embeddings)
        p(f"KB saved to {path} ({len(self._sentences)} sentences)")

    def _load_kb(self, path):
        """Load knowledge base from disk."""
        import json
        with open(path) as f:
            data = json.load(f)
        self._sentences = data.get("sentences", [])
        self._qa_pairs = data.get("qa_pairs", {})
        npy_path = path + '.npy'
        if os.path.exists(npy_path):
            self._embeddings = np.load(npy_path)
        p(f"KB loaded: {len(self._sentences)} sentences, {len(self._qa_pairs)} Q→A pairs")


# --- INTERACTIVE REPL ---

def repl():
    p("=" * 60)
    p("  Phase-Brain: Learns As You Speak")
    p("=" * 60)
    p("")
    p("Commands:")
    p("  teach: <sentence>     — teach a fact")
    p("  qa: <question> | <answer>  — teach Q→A pair")
    p("  kb                    — show KB size")
    p("  save                  — save KB")
    p("  quit                  — exit")
    p("  (anything else)       — ask a question")
    p("")

    brain = PhaseBrain()
    kb_path = os.path.join(os.path.dirname(__file__), 'phase_brain_kb.json')

    p("\nReady. Teach me something or ask a question.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue

        if user_input.lower() == 'quit':
            break
        elif user_input.lower() == 'kb':
            p(f"  KB: {len(brain._sentences)} sentences, {len(brain._qa_pairs)} Q→A pairs")
            continue
        elif user_input.lower() == 'save':
            brain.save_kb(kb_path)
            continue
        elif user_input.lower().startswith('teach:'):
            sentence = user_input[6:].strip()
            result = brain.teach(sentence)
            p(f"  {result}")
            continue
        elif user_input.lower().startswith('qa:'):
            parts = user_input[3:].split('|')
            if len(parts) == 2:
                result = brain.teach_qa(parts[0].strip(), parts[1].strip())
                p(f"  {result}")
            else:
                p("  Format: qa: question | answer")
            continue

        # Ask
        t0 = time.time()
        result = brain.ask(user_input)
        elapsed = time.time() - t0

        p(f"\n  Answer: {result['answer']}")
        p(f"  Strategy: {result['strategy']} ({elapsed*1000:.0f}ms)")
        if result.get('context'):
            p(f"  Retrieved:")
            for c in result['context'][:2]:
                p(f"    [{c['similarity']:.3f}] {c['text'][:80]}")
        p("")


if __name__ == "__main__":
    repl()
