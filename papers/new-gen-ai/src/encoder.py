"""
Encoder: text → vector.

Loads pretrained word embeddings (GloVe 6B 300d) and converts text to vectors.
No training. The map is borrowed. Confidence is earned through use.

Word-level:  direct lookup in embedding table.
Sentence-level: weighted average of word vectors.
OOV: zero vector (honest — "I don't have this word").
"""

import io
import os
import re
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

GLOVE_URL = "https://nlp.stanford.edu/data/glove.6B.zip"
GLOVE_FILENAME = "glove.6B.300d.txt"
GLOVE_DIM = 300


class Encoder:
    """
    Encodes text to vectors using pretrained word embeddings.

    The encoder is a lookup table, not a model. It maps words to points
    in 300-dimensional space. Sentence encoding = weighted average of
    word vectors.
    """

    def __init__(self, data_dir: str, dim: int = GLOVE_DIM):
        self.dim = dim
        self.data_dir = Path(data_dir)
        self._vocab = {}       # word → np.ndarray (normalized)
        self._word_list = []   # index → word (for reverse lookup)
        self._faiss_index = None  # built after loading for fast nearest_words

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    def load(self, path: str = None):
        """
        Load embeddings from a GloVe-format text file.
        Each line: word float float float ...
        """
        if path is None:
            path = str(self.data_dir / GLOVE_FILENAME)

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Embeddings not found at {path}. "
                f"Run encoder.download() first or provide a valid path."
            )

        self._vocab = {}
        self._word_list = []

        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.rstrip().split(' ')
                word = parts[0]
                vec = np.array(parts[1:], dtype=np.float32)

                if len(vec) != self.dim:
                    continue

                # Normalize for cosine similarity
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm

                self._vocab[word] = vec
                self._word_list.append(word)

        self._build_index()

    def download(self):
        """Download GloVe embeddings if not present."""
        target = self.data_dir / GLOVE_FILENAME
        if target.exists():
            return str(target)

        self.data_dir.mkdir(parents=True, exist_ok=True)

        zip_path = self.data_dir / "glove.6B.zip"
        if not zip_path.exists():
            print(f"Downloading GloVe 6B to {zip_path} ...")
            urlretrieve(GLOVE_URL, str(zip_path))

        print(f"Extracting {GLOVE_FILENAME} ...")
        with zipfile.ZipFile(str(zip_path), 'r') as z:
            z.extract(GLOVE_FILENAME, str(self.data_dir))

        return str(target)

    def load_from_dict(self, word_vectors: dict):
        """
        Load from a dict of {word: np.ndarray}.
        Useful for testing without downloading GloVe.
        """
        self._vocab = {}
        self._word_list = []

        for word, vec in word_vectors.items():
            vec = np.array(vec, dtype=np.float32)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            self._vocab[word] = vec
            self._word_list.append(word)

        self._build_index()

    def encode_word(self, word: str) -> np.ndarray:
        """
        Encode a single word. Returns its embedding vector.
        OOV words return zero vector — honest about not knowing.
        """
        word = word.lower().strip()
        if word in self._vocab:
            return self._vocab[word].copy()
        return np.zeros(self.dim, dtype=np.float32)

    def has_word(self, word: str) -> bool:
        return word.lower().strip() in self._vocab

    def encode_sentence(self, text: str) -> np.ndarray:
        """
        Encode a sentence as weighted average of word vectors.

        Tokenization: split on whitespace + punctuation.
        Weighting: position-based (later words slightly less weight,
                   mimics recency in attention).
        OOV words skipped — they contribute nothing.

        Returns zero vector if no known words found (honest abstention).
        """
        tokens = self._tokenize(text)
        if not tokens:
            return np.zeros(self.dim, dtype=np.float32)

        vectors = []
        weights = []
        for i, token in enumerate(tokens):
            vec = self.encode_word(token)
            if np.any(vec != 0):  # skip OOV
                vectors.append(vec)
                # Position weight: slight decay, all words matter
                weights.append(1.0 / (1.0 + 0.1 * i))

        if not vectors:
            return np.zeros(self.dim, dtype=np.float32)

        vectors = np.array(vectors)
        weights = np.array(weights, dtype=np.float32)
        weights = weights / weights.sum()

        result = np.average(vectors, axis=0, weights=weights).astype(np.float32)

        # Normalize
        norm = np.linalg.norm(result)
        if norm > 0:
            result = result / norm

        return result

    def _build_index(self):
        """Build FAISS index over vocabulary for fast nearest_words."""
        if not HAS_FAISS or not self._word_list:
            self._faiss_index = None
            return

        vecs = np.array([self._vocab[w] for w in self._word_list], dtype=np.float32)
        self._faiss_index = faiss.IndexFlatIP(self.dim)
        self._faiss_index.add(vecs)

    def nearest_words(self, vector: np.ndarray, k: int = 5) -> list:
        """
        Find k nearest words to a vector. Returns [(word, similarity), ...].
        Useful for inspecting what a vector "means" — invariant #2.
        Uses FAISS index when available (O(log N) vs O(N)).
        """
        vec = np.array(vector, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm == 0:
            return []
        vec = vec / norm

        # Fast path: FAISS index
        if self._faiss_index is not None:
            k = min(k, self._faiss_index.ntotal)
            scores, indices = self._faiss_index.search(vec.reshape(1, -1), k)
            results = []
            for idx, score in zip(indices[0], scores[0]):
                if 0 <= idx < len(self._word_list):
                    results.append((self._word_list[idx], float(score)))
            return results

        # Slow path: linear scan (small vocab / no FAISS)
        scores = []
        for word, wvec in self._vocab.items():
            sim = float(np.dot(vec, wvec))
            scores.append((word, sim))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]

    def _tokenize(self, text: str) -> list:
        """Simple tokenizer: lowercase, split on non-alphanumeric."""
        text = text.lower()
        tokens = re.findall(r'[a-z0-9]+', text)
        return tokens
