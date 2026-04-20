"""
Encoder: text → vector.

No pretrained embeddings. No borrowed understanding. The system
starts with zero dimensions and grows as it learns.

Each dimension is a concept axis. When the system learns a new word,
it gets a dimension. Every word's vector describes how it relates
to every known concept. More teaching = more dimensions = more depth.

Positions are EARNED through co-occurrence, not assigned by hash
or borrowed from GloVe. "paris" and "france" start unrelated.
Teach "paris is the capital of france" → they move closer on
each other's dimensions.
"""

import os
import re
import struct
import hashlib
import zipfile
from collections import OrderedDict
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np

GLOVE_URL = "https://nlp.stanford.edu/data/glove.6B.zip"
GLOVE_FILENAME = "glove.6B.300d.txt"
GLOVE_DIM = 300

# How much co-occurring words pull toward each other
COOCCURRENCE_PULL = 0.3

# GloVe streaming cache size
_GLOVE_CACHE_SIZE = 4096


class Encoder:
    """
    Self-growing vector encoder.

    Starts with 0 dimensions, 0 words. Each new word taught adds a
    dimension. A word's vector is its relationship to every other
    known word — earned through co-occurrence, not precomputed.

    The vector space IS the understanding. It grows with the system.

    Optionally, GloVe can be loaded as a fallback for richer vectors.
    """

    def __init__(self, data_dir: str = None, dim: int = None):
        # dim=None means self-growing. dim=N means fixed (GloVe/testing).
        self._fixed_dim = dim
        self.data_dir = Path(data_dir) if data_dir else None

        # Self-growing vocabulary
        self._words = []          # index → word (dimension order)
        self._word_to_idx = {}    # word → index (which dimension is this word)
        self._matrix = None       # shape (n_words, n_words) — relationship matrix

        # GloVe streaming (optional fallback)
        self._glove_offsets = {}
        self._glove_path = None
        self._glove_cache = OrderedDict()
        self._glove_dim = None

        # Dict vocab (for testing)
        self._dict_vocab = {}

    @property
    def dim(self):
        """Current dimensionality = number of known words."""
        if self._fixed_dim is not None:
            return self._fixed_dim
        if self._dict_vocab:
            return next(iter(self._dict_vocab.values())).shape[0]
        return len(self._words)

    @property
    def vocab_size(self) -> int:
        if self._glove_offsets:
            return len(self._glove_offsets)
        if self._dict_vocab:
            return len(self._dict_vocab)
        return len(self._words)

    # --- Self-growing vocabulary ---

    def learn_word(self, word: str) -> int:
        """
        Add a word to the vocabulary. Returns its dimension index.

        Each new word:
        1. Gets its own dimension (column in the matrix)
        2. Starts with identity on its own dimension (it IS itself)
        3. All other words get 0 on this dimension (no relationship yet)

        The dimension is the word. The word is the dimension.
        """
        word = word.lower().strip()
        if word in self._word_to_idx:
            return self._word_to_idx[word]

        idx = len(self._words)
        self._words.append(word)
        self._word_to_idx[word] = idx

        # Grow the matrix: add a row and column
        if self._matrix is None:
            self._matrix = np.array([[1.0]], dtype=np.float32)
        else:
            n = self._matrix.shape[0]
            # Add column (new dimension for all existing words)
            new_col = np.zeros((n, 1), dtype=np.float32)
            self._matrix = np.hstack([self._matrix, new_col])
            # Add row (new word's relationships to all words + itself)
            new_row = np.zeros((1, n + 1), dtype=np.float32)
            new_row[0, idx] = 1.0  # identity: word IS itself
            self._matrix = np.vstack([self._matrix, new_row])

        return idx

    def learn_cooccurrence(self, words: list):
        """
        Words that appear together pull toward each other.

        "paris is the capital of france" → paris and capital move
        closer, paris and france move closer, capital and france
        move closer. Positions are earned.

        This is Hebbian learning: fire together → wire together.
        No gradient descent. Just proximity from co-occurrence.
        """
        # Ensure all words are in the vocabulary
        indices = []
        for w in words:
            w = w.lower().strip()
            if w not in self._word_to_idx:
                self.learn_word(w)
            indices.append(self._word_to_idx[w])

        # Pull co-occurring words toward each other
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                a, b = indices[i], indices[j]
                # Symmetric pull: both words gain relationship
                self._matrix[a, b] += COOCCURRENCE_PULL
                self._matrix[b, a] += COOCCURRENCE_PULL

    def _get_self_vector(self, word: str) -> np.ndarray:
        """Get a word's vector from the self-grown matrix."""
        word = word.lower().strip()
        idx = self._word_to_idx.get(word)
        if idx is None:
            return None

        vec = self._matrix[idx].copy()
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    # --- GloVe streaming (optional) ---

    def load(self, path: str = None):
        """Index a GloVe file for on-demand streaming."""
        if path is None:
            if self.data_dir is None:
                raise ValueError("No data_dir set and no path provided")
            path = str(self.data_dir / GLOVE_FILENAME)

        if not os.path.exists(path):
            raise FileNotFoundError(f"Embeddings not found at {path}.")

        self._glove_path = path
        self._glove_offsets = {}
        self._glove_cache = OrderedDict()

        # Detect dimension from first line
        with open(path, 'rb') as f:
            first_line = f.readline().decode('utf-8').rstrip()
            parts = first_line.split(' ')
            self._glove_dim = len(parts) - 1
            self._fixed_dim = self._glove_dim

        with open(path, 'rb') as f:
            while True:
                offset = f.tell()
                line = f.readline()
                if not line:
                    break
                space_idx = line.index(b' ')
                word = line[:space_idx].decode('utf-8')
                self._glove_offsets[word] = offset

    def download(self):
        """Download GloVe embeddings if not present."""
        if self.data_dir is None:
            raise ValueError("No data_dir set")
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

    def _stream_glove(self, word: str) -> np.ndarray:
        """Read a single vector from the GloVe file by seeking."""
        if word in self._glove_cache:
            self._glove_cache.move_to_end(word)
            return self._glove_cache[word].copy()

        offset = self._glove_offsets.get(word)
        if offset is None:
            return None

        with open(self._glove_path, 'rb') as f:
            f.seek(offset)
            line = f.readline().decode('utf-8').rstrip()

        parts = line.split(' ')
        vec = np.array(parts[1:], dtype=np.float32)

        if len(vec) != self._glove_dim:
            return None

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        self._glove_cache[word] = vec
        if len(self._glove_cache) > _GLOVE_CACHE_SIZE:
            self._glove_cache.popitem(last=False)

        return vec.copy()

    # --- Testing support ---

    def load_from_dict(self, word_vectors: dict):
        """Load from a dict. For testing."""
        self._dict_vocab = {}
        for word, vec in word_vectors.items():
            vec = np.array(vec, dtype=np.float32)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            self._dict_vocab[word] = vec

    # --- Core API ---

    def encode_word(self, word: str) -> np.ndarray:
        """
        Encode a single word. Priority:
          1. Dict vocab (testing)
          2. GloVe stream (if loaded)
          3. Self-grown matrix (if word was learned)
          4. Zero vector (honest: "I don't know this word")
        """
        word = word.lower().strip()
        if not word:
            return np.zeros(self.dim or 1, dtype=np.float32)

        # Dict vocab (testing)
        if word in self._dict_vocab:
            return self._dict_vocab[word].copy()

        # GloVe stream
        if self._glove_path:
            vec = self._stream_glove(word)
            if vec is not None:
                return vec
            return np.zeros(self._glove_dim, dtype=np.float32)

        # Self-grown
        vec = self._get_self_vector(word)
        if vec is not None:
            return vec

        # Unknown word: zero vector
        d = self.dim if self.dim > 0 else 1
        return np.zeros(d, dtype=np.float32)

    def has_word(self, word: str) -> bool:
        word = word.lower().strip()
        if word in self._dict_vocab:
            return True
        if word in self._glove_offsets:
            return True
        if word in self._word_to_idx:
            return True
        return False

    def encode_sentence(self, text: str) -> np.ndarray:
        """Encode a sentence as weighted average of word vectors."""
        tokens = self._tokenize(text)
        if not tokens:
            d = self.dim if self.dim > 0 else 1
            return np.zeros(d, dtype=np.float32)

        vectors = []
        weights = []
        for i, token in enumerate(tokens):
            vec = self.encode_word(token)
            if np.any(vec != 0):
                vectors.append(vec)
                weights.append(1.0 / (1.0 + 0.1 * i))

        if not vectors:
            d = self.dim if self.dim > 0 else 1
            return np.zeros(d, dtype=np.float32)

        vectors = np.array(vectors)
        weights = np.array(weights, dtype=np.float32)
        weights = weights / weights.sum()

        result = np.average(vectors, axis=0, weights=weights).astype(np.float32)

        norm = np.linalg.norm(result)
        if norm > 0:
            result = result / norm

        return result

    def nearest_words(self, vector: np.ndarray, k: int = 5) -> list:
        """
        Find k nearest words to a vector.
        Searches all words we know about: self-grown, dict, and GloVe cache.
        """
        vec = np.array(vector, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm == 0:
            return []
        vec = vec / norm

        # GloVe streaming mode: search the cache
        if self._glove_cache:
            all_words = dict(self._dict_vocab)
            all_words.update(self._glove_cache)
            if all_words:
                words = list(all_words.keys())
                vecs = np.array([all_words[w] for w in words], dtype=np.float32)
                sims = vecs @ vec
                k = min(k, len(words))
                if k <= 0:
                    return []
                top_k = np.argpartition(-sims, k - 1)[:k] if len(sims) > k else np.arange(len(sims))
                top_k = top_k[np.argsort(-sims[top_k])]
                return [(words[i], float(sims[i])) for i in top_k]

        # Self-growing mode
        if self._matrix is not None and len(self._words) > 0:
            # Normalize all word vectors
            norms = np.linalg.norm(self._matrix, axis=1, keepdims=True)
            norms = np.where(norms > 0, norms, 1.0)
            normed = self._matrix / norms

            # Handle dimension mismatch (vector might be from before growth)
            if vec.shape[0] < normed.shape[1]:
                padded = np.zeros(normed.shape[1], dtype=np.float32)
                padded[:vec.shape[0]] = vec
                vec = padded
            elif vec.shape[0] > normed.shape[1]:
                vec = vec[:normed.shape[1]]

            sims = normed @ vec
            k = min(k, len(self._words))
            if k <= 0:
                return []
            top_k = np.argpartition(-sims, k - 1)[:k] if len(sims) > k else np.arange(len(sims))
            top_k = top_k[np.argsort(-sims[top_k])]
            return [(self._words[i], float(sims[i])) for i in top_k]

        # Dict vocab
        if self._dict_vocab:
            words = list(self._dict_vocab.keys())
            vecs = np.array([self._dict_vocab[w] for w in words], dtype=np.float32)
            sims = vecs @ vec
            k = min(k, len(words))
            if k <= 0:
                return []
            top_k = np.argpartition(-sims, k - 1)[:k] if len(sims) > k else np.arange(len(sims))
            top_k = top_k[np.argsort(-sims[top_k])]
            return [(words[i], float(sims[i])) for i in top_k]

        return []

    def _tokenize(self, text: str) -> list:
        """Simple tokenizer: lowercase, split on non-alphanumeric."""
        text = text.lower()
        tokens = re.findall(r'[a-z0-9]+', text)
        return tokens
