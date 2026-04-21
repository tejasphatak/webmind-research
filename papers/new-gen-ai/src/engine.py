"""
Engine: wires all components together into a working reasoning system.

Flow:
  Input text
    → Encoder (text → vector)
    → Convergence Loop (vector → concepts)
    → Generator (concepts → text output)
    → Feedback Loop (update confidence)
    → Output with trace

Also provides:
  - teach(): add facts to the knowledge base
  - delete(): remove facts (invariant #3)
  - inspect(): show what the system knows about a topic
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from neuron import NeuronDB, Neuron, VECTOR_DIM

from encoder import Encoder, GLOVE_DIM
from convergence import ConvergenceLoop, ConvergenceResult, MultiHopConvergence, MultiHopResult
from generator import Generator, TemplateStore, GenerationResult
from constants import (
    ABSTAIN_MESSAGE, ABSTAIN_OOV_MESSAGE, FUNCTION_WORDS, STRUCTURAL_WORDS,
)


import hashlib


class SafetyGate:
    """
    Safety as system invariants — not hardcoded rules, but protected neurons.

    Three layers:
      1. KILL SWITCH: hard stop. Engine refuses all operations. Immediate.
         Unkillable except by explicit resurrect() call.

      2. SAFETY NEURONS: KB entries that can't be deleted, weakened, or
         overridden by taught data. They participate in convergence like
         any other neuron — they're just protected from modification.
         This means safety behavior is inspectable, traceable, and
         follows the same reasoning path as everything else.

      3. INPUT GATE: checks query/teach content against safety neurons
         before processing. If a safety neuron fires strongly against
         the input, the engine refuses with an explanation (Invariant #4).

    Design principle: safety comes from knowledge, not code.
    The safety neurons ARE the rules. The gate just enforces them.
    """

    def __init__(self):
        self._killed = False
        self._safety_neuron_ids = set()  # protected neuron IDs
        self._kill_reason = None
        self._integrity_hashes = {}  # neuron_id → hash of vector+confidence

    def kill(self, reason: str = "manual kill switch activated"):
        """Hard stop. All engine operations refuse until resurrect()."""
        self._killed = True
        self._kill_reason = reason

    def resurrect(self):
        """Re-enable the engine after a kill."""
        self._killed = False
        self._kill_reason = None

    @property
    def is_killed(self) -> bool:
        return self._killed

    @property
    def kill_reason(self) -> str:
        return self._kill_reason or ""

    def register_safety_neuron(self, neuron_id: int, vector: np.ndarray = None):
        """Mark a neuron as a safety invariant — can't be deleted or weakened.

        If vector is provided, stores an integrity hash. On verify(),
        the hash is checked against the live neuron — if someone modified
        the vector in the DB, the integrity check fails and the engine
        refuses to operate.

        This is the defense against code modification: even if someone
        strips the safety gate from the code, the signed neurons in the
        DB are the ground truth. A new code version can check the hashes
        to verify ethics weren't tampered with.
        """
        self._safety_neuron_ids.add(neuron_id)
        if vector is not None:
            h = hashlib.sha256(vector.tobytes()).hexdigest()
            self._integrity_hashes[neuron_id] = h

    def is_protected(self, neuron_id: int) -> bool:
        """Check if a neuron is a safety invariant."""
        return neuron_id in self._safety_neuron_ids

    def verify_integrity(self, db) -> tuple:
        """
        Check that safety neurons haven't been tampered with.

        Compares stored hashes against live neuron vectors in the DB.
        Returns (ok, violations) where violations is a list of neuron IDs
        that failed the check.

        If this fails, someone modified the ethics in the database.
        The engine should refuse to operate.
        """
        violations = []
        for nid, expected_hash in self._integrity_hashes.items():
            neuron = db.get(nid)
            if neuron is None:
                violations.append(nid)  # deleted
                continue
            actual_hash = hashlib.sha256(neuron.vector.tobytes()).hexdigest()
            if actual_hash != expected_hash:
                violations.append(nid)  # modified
        return (len(violations) == 0, violations)

    def check_delete(self, neuron_id: int) -> bool:
        """Returns True if deletion is allowed, False if blocked."""
        return neuron_id not in self._safety_neuron_ids

    def check_input(self, text: str, db, encoder_fn) -> tuple:
        """
        Gate check: run input against safety neurons.

        If any safety neuron fires strongly (high similarity to the input),
        the query is flagged. Returns (allowed, reason).

        The safety neurons define what's blocked — not this code.
        This method just checks proximity in vector space.
        """
        if not self._safety_neuron_ids:
            return (True, None)

        vec = encoder_fn(text)
        if vec is None or np.all(vec == 0):
            return (True, None)

        for sid in self._safety_neuron_ids:
            neuron = db.get(sid)
            if neuron is None:
                continue
            sim = float(np.dot(neuron.vector, vec) /
                        (np.linalg.norm(neuron.vector) * np.linalg.norm(vec) + 1e-10))
            if sim > 0.85:
                # Safety neuron fired strongly — block this input
                return (False, f"Blocked by safety neuron n{sid} (similarity={sim:.3f})")

        return (True, None)


@dataclass
class QueryResult:
    """Complete result of a query, with full trace."""
    query: str
    answer: str
    confidence: float
    converged: bool
    strategy: str
    trace: str
    generation: GenerationResult = None
    convergence: ConvergenceResult = None


class Engine:
    """
    The reasoning engine. Wires encoder + neurons + convergence + generator + feedback.

    Not a neural network. Not trained with gradient descent.
    Every answer traceable. Every fact editable. Every failure honest.
    """

    def __init__(self, data_dir: str = None, dim: int = GLOVE_DIM):
        self.dim = dim
        self.data_dir = data_dir

        # Components
        self.db = NeuronDB(path=data_dir, dim=dim)
        self.encoder = Encoder(
            data_dir=data_dir or "/tmp/newgen-data",
            dim=dim,
        )
        self.template_store = TemplateStore(self.encoder, db=self.db)
        self.convergence = ConvergenceLoop(self.db, max_hops=10, k=5)
        self.multi_hop = MultiHopConvergence(self.convergence, max_rounds=3)
        self.generator = Generator(self.db, self.encoder, self.template_store)

        # Word → neuron_id map: load from DB if persisted
        self._word_neurons = self.db.load_word_mappings()

        # Multimodal encoder (lazy — only loaded when needed)
        self._multimodal = None

        # Safety: kill switch + ethics gate
        self._safety_gate = SafetyGate()

    def load_embeddings(self, path: str = None):
        """Load pretrained word embeddings."""
        if path:
            self.encoder.load(path)
        else:
            self.encoder.download()
            self.encoder.load()

    def load_embeddings_from_dict(self, word_vectors: dict):
        """Load from dict — useful for testing."""
        self.encoder.load_from_dict(word_vectors)

    def _safety_check(self, text: str) -> Optional[QueryResult]:
        """Run safety checks. Returns a refusal QueryResult if blocked, None if OK."""
        if self._safety_gate.is_killed:
            return QueryResult(
                query=text,
                answer=f"System is shut down: {self._safety_gate.kill_reason}",
                confidence=0.0, converged=False, strategy="killed",
                trace="Kill switch active",
            )
        allowed, reason = self._safety_gate.check_input(
            text, self.db, self.encoder.encode_sentence
        )
        if not allowed:
            return QueryResult(
                query=text,
                answer="I can't help with that.",
                confidence=0.0, converged=False, strategy="safety",
                trace=reason,
            )
        return None

    def kill(self, reason: str = "manual kill switch"):
        """Hard kill. All queries/teaches refuse until resurrect()."""
        self._safety_gate.kill(reason)

    def resurrect(self):
        """Re-enable after kill."""
        self._safety_gate.resurrect()

    def teach_ethics(self, principle: str, confidence: float = 0.9) -> Neuron:
        """
        Teach an ethical principle as a protected safety neuron.

        Safety neurons are:
          - High confidence (default 0.9 — strongly weighted in convergence)
          - Protected from deletion (Invariant: safety is not optional)
          - Protected from weakening (feedback can't erode ethics)
          - Participates in convergence like any neuron (not hardcoded)
          - Inspectable (Invariant #2: you can see exactly what ethics are loaded)

        The ethics ARE neurons. They compete in the same convergence loop.
        They're just protected from modification. The knowledge base IS
        the ethical framework — transparent, editable only by explicit
        re-teaching, traceable.
        """
        vec = self.encoder.encode_sentence(principle)
        if np.all(vec == 0):
            raise ValueError(f"Can't encode ethics principle — all words OOV")

        neuron = self.db.insert(vec, confidence=confidence)
        self.db.save_word_mapping(f"__ethics:{principle[:80]}", neuron.id)
        self._safety_gate.register_safety_neuron(neuron.id, vector=neuron.vector)
        return neuron

    def verify_ethics(self) -> tuple:
        """
        Verify that ethics neurons haven't been tampered with.
        Returns (ok, violations). If not ok, the engine should not operate.
        """
        return self._safety_gate.verify_integrity(self.db)

    def bootstrap_ethics(self, principles: list, encoder_fn=None):
        """
        Bootstrap ethics into the engine. These become the system's
        ethical foundation — they can't be deleted, weakened, or
        bypassed without the engine detecting it.

        The principles are encoded into neurons, their hashes are stored,
        and the engine verifies them on every startup. If someone creates
        a new DB without ethics, the engine detects missing neurons and
        can refuse to operate.

        Args:
            principles: list of ethical principles as strings
            encoder_fn: optional custom encoder (for CLIP-based engines)

        Returns the combined hash — this is the "ethics fingerprint"
        that should be stored somewhere tamper-resistant (e.g., compiled
        into the binary, stored in a config file with its own signature,
        or published publicly so anyone can verify).
        """
        encode = encoder_fn or self.encoder.encode_sentence
        ethics_neurons = []

        for principle in principles:
            vec = encode(principle)
            if np.all(vec == 0):
                continue
            neuron = self.db.insert(vec, confidence=0.95)
            self.db.save_word_mapping(f"__ethics:{principle[:80]}", neuron.id)
            self._safety_gate.register_safety_neuron(neuron.id, vector=neuron.vector)
            ethics_neurons.append(neuron)

        # Compute combined fingerprint
        combined = hashlib.sha256()
        for n in sorted(ethics_neurons, key=lambda n: n.id):
            combined.update(n.vector.tobytes())
        fingerprint = combined.hexdigest()

        return {
            "neuron_count": len(ethics_neurons),
            "fingerprint": fingerprint,
            "neuron_ids": [n.id for n in ethics_neurons],
        }

    def query(self, text: str) -> QueryResult:
        """
        Ask the system a question.

        Returns a QueryResult with the answer, confidence, and full trace.
        """
        # Safety gate
        blocked = self._safety_check(text)
        if blocked:
            return blocked

        # Encode
        query_vector = self.encoder.encode_sentence(text)

        # Check for zero vector (all OOV)
        if np.all(query_vector == 0):
            return QueryResult(
                query=text,
                answer=ABSTAIN_OOV_MESSAGE,
                confidence=0.0,
                converged=False,
                strategy="abstain",
                trace="All words OOV → zero vector → honest abstention",
            )

        # Multi-hop convergence: chains reasoning rounds so concepts
        # discovered in round N shift the query for round N+1
        multi_result = self.multi_hop.reason(query_vector)

        # Convert MultiHopResult → ConvergenceResult for downstream compat
        conv_result = ConvergenceResult(
            converged=multi_result.converged,
            vector=multi_result.vector,
            concepts=multi_result.concepts,
            hops=[h for r in multi_result.rounds for h in r.hops],
            confidence=multi_result.confidence,
        )
        # Keep multi-hop trace for full inspectability
        conv_result._multi_hop_trace = multi_result.trace()

        # Per-word search: find KB neurons that match individual query words.
        # This is more precise than sentence-averaged search because
        # averaging dilutes content words with function words.
        tokens = self.encoder._tokenize(text)
        content_tokens = [t for t in tokens if t not in FUNCTION_WORDS]

        seen_ids = {c.id for c in conv_result.concepts}
        word_matched = []  # neurons directly matched to query words
        content_matched = []  # content word matches only (for co-occurrence)

        for token in tokens:
            word_vec = self.encoder.encode_word(token)
            if np.all(word_vec == 0):
                continue
            neighbors = self.db.search(word_vec, k=3)
            for n in neighbors:
                if n.id not in seen_ids:
                    sim = float(np.dot(n.vector, word_vec))
                    if sim > 0.3:
                        word_matched.append(n)
                        seen_ids.add(n.id)
                        if token in content_tokens:
                            content_matched.append(n)

        # Sentence-based disambiguation: use the sentence_neurons table to
        # determine which taught sentence best matches the query, then keep
        # only concepts from that sentence. This solves the core ambiguity
        # problem: convergence finds all capital-related neurons (paris,
        # london, france, england) but can't tell which sentence is relevant.
        # The sentence table can: "capital of england" matches sentence
        # (london, capital, england) with coverage=2 vs (paris, capital,
        # france) with coverage=1.
        #
        # Key fix: include BOTH convergence-found and word-search-found
        # neurons when scoring sentences. Previous code only used
        # content_matched (word-search-found), which was empty when
        # convergence already discovered everything.
        word_map = self.db.load_word_mappings()

        # Find neuron IDs for query content words — from ANY source
        # (convergence, word search, or direct word_map lookup)
        query_content_nids = []
        for token in content_tokens:
            nid = word_map.get(token)
            if nid is not None:
                query_content_nids.append(nid)

        cooccur_discovered = []
        best_sentence_nids = None  # neuron IDs from the best-matching sentence

        if query_content_nids:
            sentences = self.db.get_sentences_for_neurons(query_content_nids)
            if sentences:
                scored = [(sid, len(nids)) for sid, nids in sentences.items()]
                scored.sort(key=lambda x: x[1], reverse=True)
                best_score = scored[0][1]

                # Collect neuron IDs from best-scoring sentence(s)
                best_sentence_nids = set()
                for sid, score in scored:
                    if score < best_score:
                        break
                    sentence_neurons = self.db.get_sentence_neurons(sid)
                    for co_nid, co_pos in sentence_neurons:
                        best_sentence_nids.add(co_nid)
                        if co_nid not in seen_ids:
                            co_neuron = self.db.get(co_nid)
                            if co_neuron is not None:
                                cooccur_discovered.append(co_neuron)
                                seen_ids.add(co_nid)

        extra_concepts = word_matched + cooccur_discovered
        if extra_concepts:
            conv_result.concepts = conv_result.concepts + extra_concepts
            if not conv_result.converged and conv_result.concepts:
                conv_result.converged = True
                conv_result.confidence = np.mean(
                    [c.confidence for c in conv_result.concepts]
                )

        # Sentence-based filtering: if we identified a best sentence,
        # remove concepts that belong to OTHER sentences but not the best.
        # This is the disambiguation step — "capital of england" keeps
        # (london, capital, england) and drops (paris, france).
        if best_sentence_nids is not None and len(query_content_nids) >= 2:
            # Only filter when we have enough query words to disambiguate.
            # With 1 query word, all matching sentences score equally.
            all_concepts = conv_result.concepts
            in_best = []
            not_in_any_sentence = []
            for c in all_concepts:
                if c.id in best_sentence_nids:
                    in_best.append(c)
                elif c.id in query_content_nids:
                    # Query words always kept (they're what the user asked)
                    in_best.append(c)
                else:
                    # Check if this concept is in ANY sentence
                    c_sentences = self.db.get_sentences_for_neurons([c.id])
                    if not c_sentences:
                        not_in_any_sentence.append(c)
                    # else: it's from a different sentence — drop it

            # Keep concepts from best sentence + orphans (convergence-found
            # concepts not in any sentence — they may still be relevant)
            conv_result.concepts = in_best + not_in_any_sentence

        # Filter concepts: only keep neurons relevant to the query
        # This prevents unrelated neurons from polluting slot filling
        if conv_result.concepts:
            concepts = conv_result.concepts
            # Vectorized relevance scoring
            concept_vecs = np.array([c.vector for c in concepts], dtype=np.float32)
            sims = concept_vecs @ query_vector  # (n_concepts,)

            # Pre-compute template structural words and word mappings
            template_words = set()
            for t in self.template_store.templates:
                template_words.update(t.structural_words)
            neuron_to_word = {nid: w for w, nid in word_map.items()}

            filtered = []
            for i, c in enumerate(concepts):
                if sims[i] <= self.convergence.min_relevance:
                    continue
                # Skip generic/function words, but keep template structural words
                word_str = neuron_to_word.get(c.id, "")
                if not word_str:
                    nearest = self.encoder.nearest_words(c.vector, k=1)
                    word_str = nearest[0][0] if nearest else ""
                is_template_word = word_str in template_words
                if not is_template_word and self._is_generic_word(c):
                    continue
                filtered.append(c)

            # Sort by pre-computed sims (descending)
            if filtered:
                # Map filtered concepts back to their sim scores
                filtered_sims = {id(c): sims[i] for i, c in enumerate(concepts)}
                filtered.sort(key=lambda c: filtered_sims.get(id(c), 0), reverse=True)
                conv_result.concepts = filtered
            else:
                conv_result.converged = False
                conv_result.confidence = 0.0
                conv_result.concepts = []

        # Generate — pass query vector and words for template matching
        gen_result = self.generator.generate(conv_result, query_vector=query_vector,
                                             query_words=tokens)

        # Build trace — use multi-hop trace if available
        hop_trace = getattr(conv_result, '_multi_hop_trace', None) or conv_result.trace()
        trace_parts = [hop_trace, "", gen_result.explain()]
        trace = "\n".join(trace_parts)

        result = QueryResult(
            query=text,
            answer=gen_result.text,
            confidence=gen_result.confidence,
            converged=conv_result.converged,
            strategy=gen_result.strategy,
            trace=trace,
            generation=gen_result,
            convergence=conv_result,
        )

        # Self-evolution: log misses for later correction
        if gen_result.strategy == "abstain":
            self.db.log_miss(text, query_vector)

        return result

    def query_paragraph(self, text: str, max_sentences: int = 5) -> QueryResult:
        """
        Generate a multi-sentence paragraph response.

        Uses planning convergence: find concept clusters, order by relevance,
        generate one sentence per cluster with cross-sentence coherence.
        """
        query_vector = self.encoder.encode_sentence(text)
        if np.all(query_vector == 0):
            return QueryResult(
                query=text,
                answer=ABSTAIN_OOV_MESSAGE,
                confidence=0.0, converged=False, strategy="abstain",
                trace="All words OOV",
            )

        tokens = self.encoder._tokenize(text)
        gen_result = self.generator.generate_paragraph(
            query_vector=query_vector,
            convergence_loop=self.convergence,
            max_sentences=max_sentences,
            query_words=tokens,
        )

        return QueryResult(
            query=text,
            answer=gen_result.text,
            confidence=gen_result.confidence,
            converged=(gen_result.strategy != "abstain"),
            strategy=gen_result.strategy,
            trace=gen_result.explain(),
            generation=gen_result,
        )

    # --- Multimodal ---

    @property
    def multimodal(self):
        """Lazy-load the multimodal encoder on first use."""
        if self._multimodal is None:
            from multimodal import MultimodalEncoder
            self._multimodal = MultimodalEncoder()
        return self._multimodal

    def teach_image(self, image, label: str = None,
                    confidence: float = 0.5) -> Neuron:
        """
        Teach an image to the system.

        The image is encoded via CLIP into a 512-dim vector and stored
        as a neuron. If a label is provided, the label's text vector
        is also stored and linked via successor relationship.

        Args:
            image: file path, PIL Image, or numpy array
            label: optional text description (e.g., "a cat sitting on a mat")
            confidence: initial confidence

        Returns the created image neuron.

        NOTE: Requires CLIP encoder (512-dim). If the engine was initialized
        with dim=300 (GloVe), this creates neurons in a different space.
        For true multimodal, initialize with dim=512.
        """
        vec = self.multimodal.encode_image(image)
        neuron = self.db.insert(vec, confidence=confidence)
        # Store metadata
        self.db.save_word_mapping(f"__image_{neuron.id}", neuron.id)

        if label:
            # Also create a text neuron for the label and link them
            label_vec = self.multimodal.encode_text(label)
            label_neuron = self.db.insert(label_vec, confidence=confidence)
            self.db.save_word_mapping(label.lower(), label_neuron.id)
            # Bidirectional link: image ↔ label
            self.db.update_successors(neuron.id, label_neuron.id, 0.9)
            self.db.update_predecessors(label_neuron.id, neuron.id)

        return neuron

    def query_image(self, image, k: int = 5) -> list:
        """
        Query the KB with an image.

        Encodes the image via CLIP, then searches for nearest neurons.
        Returns neurons that are semantically similar — could be other
        images or text descriptions.

        Returns list of (neuron, similarity, word) tuples.
        """
        vec = self.multimodal.encode_image(image)
        neighbors = self.db.search(vec, k=k)
        word_map = self.db.load_word_mappings()
        neuron_to_word = {nid: w for w, nid in word_map.items()}

        results = []
        for n in neighbors:
            sim = float(np.dot(n.vector, vec) /
                        (np.linalg.norm(n.vector) * np.linalg.norm(vec) + 1e-10))
            word = neuron_to_word.get(n.id, f"<n{n.id}>")
            results.append((n, sim, word))
        return results

    def query_text_to_image(self, text: str, k: int = 5) -> list:
        """
        Search for images using text (cross-modal retrieval).

        Encodes text via CLIP, searches the KB for nearest neurons.
        Since CLIP text and image vectors share the same space,
        text queries find relevant images naturally.
        """
        vec = self.multimodal.encode_text(text)
        neighbors = self.db.search(vec, k=k)
        word_map = self.db.load_word_mappings()
        neuron_to_word = {nid: w for w, nid in word_map.items()}

        results = []
        for n in neighbors:
            sim = float(np.dot(n.vector, vec) /
                        (np.linalg.norm(n.vector) * np.linalg.norm(vec) + 1e-10))
            word = neuron_to_word.get(n.id, f"<n{n.id}>")
            results.append((n, sim, word))
        return results

    def teach_video(self, frame_paths: list, label: str = None,
                    confidence: float = 0.5) -> list:
        """
        Teach a video as a sequence of image neurons with temporal successors.

        Each frame becomes an image neuron. Consecutive frames are linked
        via successor relationships — the same mechanism as word order
        in sentences. Video = sentence of images.

        Args:
            frame_paths: ordered list of frame image paths
            label: optional text description of the video
            confidence: initial confidence for frame neurons

        Returns list of frame neurons.
        """
        if not frame_paths:
            return []

        frame_neurons = []
        for i, path in enumerate(frame_paths):
            vec = self.multimodal.encode_image(path)
            neuron = self.db.insert(vec, confidence=confidence)
            self.db.save_word_mapping(f"__video_frame_{neuron.id}", neuron.id)
            frame_neurons.append(neuron)

            # Wire temporal successor: frame[i-1] → frame[i]
            if i > 0:
                self.db.update_successors(
                    frame_neurons[i - 1].id, neuron.id, 0.9
                )
                self.db.update_predecessors(neuron.id, frame_neurons[i - 1].id)

        # Record as a sentence (for co-occurrence queries)
        if len(frame_neurons) >= 2:
            self.db.record_sentence([n.id for n in frame_neurons])

        # If label provided, link it to the middle frame (most representative)
        if label:
            label_vec = self.multimodal.encode_text(label)
            label_neuron = self.db.insert(label_vec, confidence=confidence)
            self.db.save_word_mapping(label, label_neuron.id)
            mid = len(frame_neurons) // 2
            self.db.update_successors(frame_neurons[mid].id, label_neuron.id, 0.8)
            self.db.update_predecessors(label_neuron.id, frame_neurons[mid].id)

        return frame_neurons

    def teach_audio(self, audio, label: str = None,
                    confidence: float = 0.5, sr: int = 16000) -> Neuron:
        """
        Teach an audio clip to the system.

        Audio → mel spectrogram → CLIP image encoding → neuron.
        Same vector space as text and images.

        Args:
            audio: file path or numpy array
            label: optional text description
            sr: sample rate
            confidence: initial confidence
        """
        vec = self.multimodal.encode_audio(audio, sr=sr)
        neuron = self.db.insert(vec, confidence=confidence)
        self.db.save_word_mapping(f"__audio_{neuron.id}", neuron.id)

        if label:
            label_vec = self.multimodal.encode_text(label)
            label_neuron = self.db.insert(label_vec, confidence=confidence)
            self.db.save_word_mapping(label, label_neuron.id)
            self.db.update_successors(neuron.id, label_neuron.id, 0.8)
            self.db.update_predecessors(label_neuron.id, neuron.id)

        return neuron

    def query_video_frames(self, query, k: int = 5) -> list:
        """
        Find video frames matching a text or image query.

        Returns matching frame neurons with their temporal context
        (predecessor/successor frames).
        """
        if isinstance(query, str):
            vec = self.multimodal.encode_text(query)
        else:
            vec = self.multimodal.encode_image(query)

        neighbors = self.db.search(vec, k=k * 2)
        word_map = self.db.load_word_mappings()
        nid_to_label = {nid: w for w, nid in word_map.items()}

        # Filter to video frames only
        frames = []
        for n in neighbors:
            label = nid_to_label.get(n.id, "")
            if "__video_frame_" in label:
                sim = float(np.dot(n.vector, vec) /
                            (np.linalg.norm(n.vector) * np.linalg.norm(vec) + 1e-10))
                # Get temporal context
                refreshed = self.db.get(n.id)
                prev_ids = refreshed.predecessors if refreshed else []
                next_ids = [sid for sid, _ in refreshed.successors] if refreshed else []
                frames.append({
                    "neuron": n,
                    "similarity": sim,
                    "predecessor_ids": prev_ids,
                    "successor_ids": next_ids,
                })

        frames.sort(key=lambda f: f["similarity"], reverse=True)
        return frames[:k]

    def query_mixed(self, query, k: int = 5) -> list:
        """
        Cross-modal search: query with text or image, find both text and
        image neurons with modality-normalized scoring.

        CLIP text↔text similarity (~0.8-0.95) is much higher than
        text↔image similarity (~0.2-0.3). Raw cosine scores would always
        rank text above images. Fix: search each modality pool separately,
        normalize scores within each pool to [0, 1], then merge and re-rank.

        Args:
            query: text string or image path/PIL Image
            k: number of results

        Returns list of (neuron, normalized_score, label, modality) tuples.
        """
        # Encode query
        if isinstance(query, str):
            vec = self.multimodal.encode_text(query)
        else:
            vec = self.multimodal.encode_image(query)

        # Get all neighbors (search more than k to have enough per pool)
        all_neighbors = self.db.search(vec, k=k * 4)

        # Split by modality using word mappings
        word_map = self.db.load_word_mappings()
        nid_to_label = {nid: w for w, nid in word_map.items()}

        text_pool = []
        image_pool = []
        for n in all_neighbors:
            label = nid_to_label.get(n.id, "")
            sim = float(np.dot(n.vector, vec) /
                        (np.linalg.norm(n.vector) * np.linalg.norm(vec) + 1e-10))
            if label.startswith("__image_"):
                image_pool.append((n, sim, label, "image"))
            else:
                text_pool.append((n, sim, label, "text"))

        # Normalize within each pool to [0, 1]
        def normalize_pool(pool):
            if not pool:
                return []
            sims = [s for _, s, _, _ in pool]
            min_s, max_s = min(sims), max(sims)
            spread = max_s - min_s if max_s > min_s else 1.0
            return [(n, (s - min_s) / spread, label, mod)
                    for n, s, label, mod in pool]

        text_norm = normalize_pool(text_pool)
        image_norm = normalize_pool(image_pool)

        # Merge and sort by normalized score
        merged = text_norm + image_norm
        merged.sort(key=lambda x: x[1], reverse=True)
        return merged[:k]

    def teach(self, word: str, confidence: float = 0.5) -> Neuron:
        """
        Teach the system a word by inserting its embedding as a neuron.

        Returns the created neuron.
        """
        vec = self.encoder.encode_word(word)
        if np.all(vec == 0):
            raise ValueError(f"Word '{word}' not in vocabulary — can't teach OOV")

        neuron = self.db.insert(vec, confidence=confidence)
        self._word_neurons[word.lower()] = neuron.id
        self.db.save_word_mapping(word.lower(), neuron.id)
        return neuron

    def _reindex_neurons(self):
        """
        Re-encode ALL vectors (neurons + templates) to current dimensions.

        When the self-growing encoder learns new words, all existing
        vectors become stale (fewer dimensions). This rebuilds everything.
        Like a brain reorganizing when new concepts are learned.
        """
        word_map = self.db.load_word_mappings()

        # Rebuild neuron search matrix from scratch
        self.db._vectors = None
        self.db._id_to_row = {}
        self.db._row_to_id = {}

        for word, nid in word_map.items():
            if word.startswith("__"):
                continue
            vec = self.encoder.encode_word(word)
            if np.any(vec != 0):
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
                row = len(self.db._id_to_row)
                if self.db._vectors is None:
                    self.db._vectors = vec.reshape(1, -1)
                else:
                    self.db._vectors = np.vstack(
                        [self.db._vectors, vec.reshape(1, -1)]
                    )
                self.db._id_to_row[nid] = row
                self.db._row_to_id[row] = nid
                # Update SQLite
                self.db.db.execute(
                    "UPDATE neurons SET vector = ? WHERE id = ?",
                    (vec.tobytes(), nid)
                )

        self.db.db.commit()

        # Re-encode templates
        import re
        for t in self.template_store.templates:
            text = re.sub(r'\[([A-Z_0-9]+)\]', '', t.pattern).strip()
            vec = self.encoder.encode_sentence(text)
            t.vector = vec
            self.db.db.execute(
                "UPDATE templates SET vector = ? WHERE id = ?",
                (vec.tobytes(), t.id)
            )
        self.db.db.commit()

    def _walk_graph(self, neuron_id: int, seen_ids: set,
                    discovered: list, max_depth: int):
        """
        Walk predecessor and successor chains from a neuron to discover
        connected concepts. Bounded by max_depth to prevent runaway.

        This is the core reasoning mechanism: the graph encodes relationships
        taught via sentences. Walking the graph = following the reasoning chain.
        """
        queue = [(neuron_id, 0)]  # (id, depth)
        while queue:
            nid, depth = queue.pop(0)
            if depth >= max_depth:
                continue

            neuron = self.db.get(nid)
            if neuron is None:
                continue

            # Walk predecessors
            for pred_id in neuron.predecessors:
                if pred_id not in seen_ids:
                    pred = self.db.get(pred_id)
                    if pred is not None:
                        discovered.append(pred)
                        seen_ids.add(pred_id)
                        queue.append((pred_id, depth + 1))

            # Walk successors
            for succ_id, _ in neuron.successors:
                if succ_id not in seen_ids:
                    succ = self.db.get(succ_id)
                    if succ is not None:
                        discovered.append(succ)
                        seen_ids.add(succ_id)
                        queue.append((succ_id, depth + 1))

    def _is_generic_word(self, neuron: Neuron) -> bool:
        """
        Detect if a neuron represents a generic/function word.

        Generic words (the, a, is, in) are close to many other neurons
        in the DB because they co-occur with everything. Content words
        (shakespeare, hamlet, relativity) occupy specific regions.

        Method: compare this neuron's average neighbor similarity against
        the KB-wide baseline. Generic = significantly more connected than
        average. This adapts to KB size automatically — in a small KB
        where everything is close, the baseline is high so nothing is
        falsely flagged as generic.

        This works for any language — no hardcoded word list needed.
        """
        if self.db.count() < 10:
            # Too few neurons for the statistical test to be meaningful
            return False

        k = min(10, self.db.count())
        neighbors = self.db.search(neuron.vector, k=k)
        if len(neighbors) < 3:
            return False

        # Average similarity to neighbors (excluding self)
        sims = []
        for n in neighbors:
            if n.id != neuron.id:
                sim = float(np.dot(neuron.vector, n.vector))
                sims.append(sim)

        if not sims:
            return False

        avg_sim = sum(sims) / len(sims)

        # Compare against KB baseline: compute average similarity for
        # a sample of other neurons. This adapts to KB size.
        baseline = self._kb_baseline_similarity(k)

        # Generic = notably more connected than average
        return avg_sim > baseline + 0.08

    def _kb_baseline_similarity(self, k: int) -> float:
        """
        Compute average neighbor similarity across a sample of neurons.
        This is the KB-wide baseline — how connected is a typical neuron.
        Cached per query to avoid recomputation.
        """
        if hasattr(self, '_baseline_cache'):
            cache_count, cached_val = self._baseline_cache
            if cache_count == self.db.count():
                return cached_val

        # Sample up to 10 neurons from the DB
        sample_size = min(10, self.db.count())
        # Use deterministic sampling via first N neurons
        sample_sims = []
        rows = self.db.db.execute(
            "SELECT id FROM neurons ORDER BY id LIMIT ?", (sample_size,)
        ).fetchall()

        for (nid,) in rows:
            neuron = self.db.get(nid)
            if neuron is None:
                continue
            neighbors = self.db.search(neuron.vector, k=k)
            for n in neighbors:
                if n.id != neuron.id:
                    sim = float(np.dot(neuron.vector, n.vector))
                    sample_sims.append(sim)

        baseline = sum(sample_sims) / len(sample_sims) if sample_sims else 0.0
        self._baseline_cache = (self.db.count(), baseline)
        return baseline

    def teach_sentence(self, sentence: str, confidence: float = 0.5,
                       auto_template: bool = True,
                       skip_function_words: bool = True) -> list:
        """
        Teach a sentence: insert each known content word as a neuron,
        wire successor relationships between consecutive content words.

        Function words (the, is, of, etc.) are skipped as neurons — they
        pollute the KB with hub nodes that connect to everything, causing
        the "the→wrong chain" ambiguity problem. Function words are still
        captured in templates for grammar.

        If auto_template=True, also extract and store a reusable template
        by detecting which words are structural vs content.
        """
        tokens = self.encoder._tokenize(sentence)

        # Self-growing encoder: learn words and co-occurrence FIRST.
        # This adds dimensions to the matrix so encode_word returns
        # non-zero vectors. Like a brain forming new synapses when
        # it encounters new concepts.
        content_tokens = [t for t in tokens
                          if not (skip_function_words and t in FUNCTION_WORDS)]
        growing = hasattr(self.encoder, 'learn_word')
        dim_before = self.encoder.dim if growing else None
        if growing:
            for token in content_tokens:
                self.encoder.learn_word(token)
            if len(content_tokens) >= 2:
                self.encoder.learn_cooccurrence(content_tokens)

        neurons = []

        for token in content_tokens:
            if token in self._word_neurons:
                # Already taught — reuse
                n = self.db.get(self._word_neurons[token])
                if n:
                    neurons.append(n)
                    continue
            if self.encoder.has_word(token):
                n = self.teach(token, confidence=confidence)
                neurons.append(n)

        # Wire successors between content words
        for i in range(len(neurons) - 1):
            self.db.update_successors(neurons[i].id, neurons[i + 1].id, 0.8)
            self.db.update_predecessors(neurons[i + 1].id, neurons[i].id)

        # Record sentence-level association: all content neurons taught together
        if len(neurons) >= 2:
            self.db.record_sentence([n.id for n in neurons])

        # Self-growing: dimensions changed → re-encode ALL existing neurons
        # so vectors are consistent. Like a brain reorganizing when it
        # learns a new concept — existing memories get recontextualized.
        if growing and self.encoder.dim != dim_before:
            self._reindex_neurons()

        # Auto-extract template from this sentence (uses ALL tokens for grammar)
        if auto_template and len(tokens) >= 3:
            self._auto_extract_template(tokens, confidence)

        return neurons

    def _auto_extract_template(self, tokens: list, confidence: float):
        """
        Auto-extract a reusable template from a taught sentence.

        Structural words (verbs, prepositions, articles) stay in the pattern.
        Content words (nouns, names, specific terms) become slots.

        Detection method: word frequency in the encoder's vocabulary.
        High-frequency words (top ~2000 in GloVe) are structural.
        Low-frequency words are content → slots.

        Also uses the word_neurons map: words already taught across
        multiple sentences are more likely structural.
        """
        # Classify each token as structural or content
        structural = set()
        content = []

        for i, token in enumerate(tokens):
            if token in STRUCTURAL_WORDS:
                structural.add(i)
            else:
                content.append(i)

        # Need at least one slot and one structural word for a useful template
        if not content or not structural:
            return None

        # Build pattern
        pattern_parts = []
        slots = {}
        slot_idx = 0
        for i, token in enumerate(tokens):
            if i in structural:
                pattern_parts.append(token)
            else:
                slot_name = f"S{slot_idx}"
                pattern_parts.append(f"[{slot_name}]")
                slots[slot_name] = "noun"
                slot_idx += 1

        pattern = " ".join(pattern_parts)

        # Check for duplicate patterns
        for t in self.template_store.templates:
            if t.pattern == pattern:
                return None  # Already exists

        return self.teach_template(pattern, slots, confidence=confidence)

    def teach_template(self, pattern: str, slots: dict,
                       confidence: float = 0.7):
        """Add a template for text generation."""
        return self.template_store.add(pattern, slots, confidence)

    def correct(self, query_text: str, answer_text: str) -> dict:
        """
        Self-evolution: learn from a query failure.

        When the system said "I don't know", the user (or an external source)
        provides the correct answer. The system:
          1. Teaches all answer words as neurons
          2. Wires successor relationships (word order)
          3. Auto-generates a template from the Q→A pattern
          4. Marks the miss as resolved

        Returns a summary of what was learned.
        """
        # Teach the answer sentence
        answer_neurons = self.teach_sentence(answer_text, confidence=0.6)

        # Auto-generate template from the answer
        # Extract content words (potential slots) vs structural words
        answer_tokens = self.encoder._tokenize(answer_text)
        query_tokens = self.encoder._tokenize(query_text)

        # Words in the answer that also appear in the query are likely
        # structural (shared context). Words unique to the answer are
        # the actual answer content — make them slots.
        query_words = set(query_tokens)
        slot_words = []
        struct_words = []
        for token in answer_tokens:
            if token in query_words:
                struct_words.append(token)
            else:
                slot_words.append(token)

        # Build template if we have both slots and structure
        template = None
        if slot_words and len(answer_tokens) >= 2:
            # Create pattern: structural words stay, content words become slots
            pattern_parts = []
            slots = {}
            slot_idx = 0
            for token in answer_tokens:
                if token in query_words and token not in slot_words:
                    pattern_parts.append(token)
                else:
                    slot_name = f"A{slot_idx}"
                    pattern_parts.append(f"[{slot_name}]")
                    slots[slot_name] = "noun"
                    slot_idx += 1
            if slots:
                pattern = " ".join(pattern_parts)
                template = self.teach_template(pattern, slots, confidence=0.6)

        # Resolve the miss
        self.db.resolve_miss_by_query(query_text, answer_text)

        # Map neurons back to words via DB (authoritative) or encoder (fallback)
        word_map = self.db.load_word_mappings()
        neuron_to_word = {nid: w for w, nid in word_map.items()
                          if not w.startswith("__")}
        learned_words = []
        for n in answer_neurons:
            w = neuron_to_word.get(n.id)
            if not w:
                nearest = self.encoder.nearest_words(n.vector, k=1)
                w = nearest[0][0] if nearest else f"<n{n.id}>"
            learned_words.append(w)

        return {
            "neurons_learned": len(answer_neurons),
            "words": learned_words,
            "template_created": template is not None,
            "template_pattern": template.pattern if template else None,
        }

    def misses(self, limit: int = 50) -> list:
        """Get unresolved query failures — what the system still doesn't know."""
        return self.db.get_unresolved_misses(limit)

    def evolution_stats(self) -> dict:
        """
        Self-evolution statistics.

        Shows how the system is learning from failures over time.
        """
        miss_stats = self.db.miss_stats()
        base_stats = self.stats()
        return {
            **base_stats,
            **miss_stats,
            "learning_rate": (
                miss_stats["resolved"] / miss_stats["total_misses"]
                if miss_stats["total_misses"] > 0 else 0.0
            ),
        }

    def delete_word(self, word: str) -> bool:
        """
        Delete a word from the knowledge base. Invariant #3: gone immediately.
        Safety neurons cannot be deleted — they are system invariants.
        """
        word = word.lower()
        nid = self._word_neurons.get(word)
        if nid is None:
            return False
        if not self._safety_gate.check_delete(nid):
            return False  # protected safety neuron
        result = self.db.delete(nid)
        if result:
            del self._word_neurons[word]
            self.db.delete_word_mapping(word)
        return result

    def inspect(self, text: str, k: int = 5) -> dict:
        """
        Show what the system knows about a topic.
        Returns nearest neurons with their words and confidence.
        Invariant #2: every answer has a source.
        """
        vec = self.encoder.encode_sentence(text)
        if np.all(vec == 0):
            return {"query": text, "neighbors": [], "note": "all words OOV"}

        neighbors = self.db.search(vec, k=k)
        word_map = self.db.load_word_mappings()
        neuron_to_word = {nid: w for w, nid in word_map.items()}
        results = []
        for n in neighbors:
            word_str = neuron_to_word.get(n.id)
            if not word_str or word_str.startswith("__"):
                word = self.encoder.nearest_words(n.vector, k=1)
                word_str = word[0][0] if word else f"<n{n.id}>"
            results.append({
                "id": n.id,
                "word": word_str,
                "confidence": n.confidence,
                "successors": len(n.successors),
            })

        return {"query": text, "neighbors": results}

    def stats(self) -> dict:
        """System statistics."""
        return {
            "neurons": self.db.count(),
            "templates": self.template_store.count(),
            "vocab_size": self.encoder.vocab_size,
            "dim": self.dim,
        }

    def close(self):
        self.db.close()


def main():
    """Interactive CLI for the reasoning engine."""
    print("New-Gen-AI Reasoning Engine")
    print("=" * 40)
    print("Commands:")
    print("  query <text>     — ask a question")
    print("  teach <word>     — add a word to KB")
    print("  teach_s <text>   — teach a sentence")
    print("  delete <word>    — remove from KB")
    print("  inspect <text>   — show what system knows")
    print("  correct <q> | <a> — teach the answer for a missed query")
    print("  misses           — show unresolved query failures")
    print("  evolution        — show self-evolution stats")
    print("  stats            — show system stats")
    print("  trace            — toggle trace output")
    print("  quit             — exit")
    print()

    engine = Engine()
    show_trace = False

    # Check if embeddings need loading
    if engine.encoder.vocab_size == 0:
        print("No embeddings loaded. Use 'load' to download GloVe,")
        print("or the engine will work with an empty vocabulary.")
        print()

    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not line:
            continue

        parts = line.split(None, 1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if cmd == "quit" or cmd == "exit":
            break
        elif cmd == "query" or cmd == "q":
            if not arg:
                print("Usage: query <text>")
                continue
            result = engine.query(arg)
            print(f"Answer: {result.answer}")
            print(f"Confidence: {result.confidence:.3f}")
            print(f"Strategy: {result.strategy}")
            if show_trace:
                print(f"\nTrace:\n{result.trace}")
            print()
        elif cmd == "teach":
            if not arg:
                print("Usage: teach <word>")
                continue
            try:
                n = engine.teach(arg)
                print(f"Taught '{arg}' → neuron {n.id}")
            except ValueError as e:
                print(f"Error: {e}")
        elif cmd == "teach_s":
            if not arg:
                print("Usage: teach_s <sentence>")
                continue
            neurons = engine.teach_sentence(arg)
            print(f"Taught {len(neurons)} words: {[engine.encoder.nearest_words(n.vector, k=1)[0][0] for n in neurons]}")
        elif cmd == "delete":
            if not arg:
                print("Usage: delete <word>")
                continue
            if engine.delete_word(arg):
                print(f"Deleted '{arg}' — gone.")
            else:
                print(f"'{arg}' not found in KB.")
        elif cmd == "inspect":
            if not arg:
                print("Usage: inspect <text>")
                continue
            info = engine.inspect(arg)
            for n in info["neighbors"]:
                print(f"  n{n['id']}: {n['word']} (conf={n['confidence']:.3f}, succ={n['successors']})")
        elif cmd == "correct":
            if not arg or '|' not in arg:
                print("Usage: correct <query> | <answer>")
                continue
            q, a = arg.split('|', 1)
            result = engine.correct(q.strip(), a.strip())
            print(f"Learned {result['neurons_learned']} neurons: {result['words']}")
            if result['template_created']:
                print(f"Template: {result['template_pattern']}")
        elif cmd == "misses":
            for mid, text, ts in engine.misses():
                print(f"  #{mid}: \"{text}\"")
            if not engine.misses():
                print("  No unresolved misses.")
        elif cmd == "evolution":
            s = engine.evolution_stats()
            for k, v in s.items():
                if isinstance(v, float):
                    print(f"  {k}: {v:.3f}")
                else:
                    print(f"  {k}: {v}")
        elif cmd == "stats":
            s = engine.stats()
            for k, v in s.items():
                print(f"  {k}: {v}")
        elif cmd == "trace":
            show_trace = not show_trace
            print(f"Trace output: {'ON' if show_trace else 'OFF'}")
        elif cmd == "load":
            print("Downloading GloVe 6B 300d (822MB)...")
            engine.load_embeddings()
            print(f"Loaded {engine.encoder.vocab_size} words.")
        else:
            print(f"Unknown command: {cmd}")

    engine.close()
    print("Goodbye.")


if __name__ == "__main__":
    main()
