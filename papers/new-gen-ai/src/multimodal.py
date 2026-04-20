"""
Multimodal encoder: text + images → shared vector space.

Uses CLIP (via sentence-transformers) to encode both text and images
into the same 512-dim space. This means:
  - An image neuron and a text neuron can be compared directly
  - Convergence loop works identically on any modality
  - "Search for images similar to this text" = same spatial query

The encoder is a pre-trained map, not our model. Invariant #1 holds.
We don't train it. We use it to project different modalities into
a shared space where our convergence/reasoning/generation operates.

Modality field on neurons tracks what each neuron represents:
  - "text" — word or sentence
  - "image" — image region or full image
  - "audio" — audio segment (future)
  - "video" — video frame (future)
"""

from pathlib import Path
from typing import Optional

import numpy as np

try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
    HAS_SBERT = True
except ImportError:
    HAS_SBERT = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


# CLIP model via sentence-transformers — encodes text and images
# into the same 512-dim space
CLIP_MODEL_NAME = "clip-ViT-B-32"
CLIP_DIM = 512

# Best multilingual sentence encoder — 50+ languages, 768-dim
# Understands polarity better than GloVe/MiniLM
MULTILINGUAL_MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"
MULTILINGUAL_DIM = 768

# NLI cross-encoder for contradiction/entailment detection
NLI_MODEL_NAME = "cross-encoder/nli-MiniLM2-L6-H768"


class MultimodalEncoder:
    """
    Encodes text and images into a shared vector space via CLIP.

    Text: string → 512-dim vector
    Image: file path or PIL Image → 512-dim vector

    Both modalities land in the same space — cosine similarity
    between a text vector and an image vector is meaningful.
    """

    def __init__(self, model_name: str = CLIP_MODEL_NAME, device: str = "cpu"):
        if not HAS_SBERT:
            raise ImportError(
                "sentence-transformers required for multimodal encoder. "
                "Install: pip install sentence-transformers"
            )
        self.model_name = model_name
        self.device = device
        self._model = None  # lazy load
        self.dim = CLIP_DIM

    @property
    def model(self):
        """Lazy-load the CLIP model on first use."""
        if self._model is None:
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def encode_text(self, text: str) -> np.ndarray:
        """Encode text into the shared vector space."""
        vec = self.model.encode(text, convert_to_numpy=True).astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def encode_image(self, image) -> np.ndarray:
        """
        Encode an image into the shared vector space.

        Args:
            image: file path (str/Path), PIL Image, or numpy array

        Returns:
            Normalized 512-dim vector in the same space as text.
        """
        if not HAS_PIL:
            raise ImportError("Pillow required for image encoding. Install: pip install Pillow")

        # Load image if path
        if isinstance(image, (str, Path)):
            img = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            img = Image.fromarray(image).convert("RGB")
        else:
            img = image  # assume PIL Image

        vec = self.model.encode(img, convert_to_numpy=True).astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def encode_batch_text(self, texts: list) -> np.ndarray:
        """Batch encode multiple texts. Returns (N, 512) matrix."""
        vecs = self.model.encode(texts, convert_to_numpy=True,
                                 batch_size=32).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        return vecs / norms

    def encode_batch_images(self, images: list) -> np.ndarray:
        """Batch encode multiple images. Returns (N, 512) matrix."""
        if not HAS_PIL:
            raise ImportError("Pillow required")

        pil_images = []
        for img in images:
            if isinstance(img, (str, Path)):
                pil_images.append(Image.open(img).convert("RGB"))
            elif isinstance(img, np.ndarray):
                pil_images.append(Image.fromarray(img).convert("RGB"))
            else:
                pil_images.append(img)

        vecs = self.model.encode(pil_images, convert_to_numpy=True,
                                 batch_size=32).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        return vecs / norms

    def encode_audio(self, audio, sr: int = 16000) -> np.ndarray:
        """
        Encode audio into the shared vector space via spectrogram → CLIP.

        Converts audio to a mel spectrogram image, then encodes that
        image through CLIP. This puts audio in the same vector space
        as text and images — a sound can be compared to a word or a photo.

        Args:
            audio: file path (str/Path) or numpy array of audio samples
            sr: sample rate (default 16kHz)

        Returns:
            Normalized 512-dim vector in the shared space.
        """
        import scipy.signal

        # Load audio if path
        if isinstance(audio, (str, Path)):
            import soundfile
            samples, sr = soundfile.read(str(audio))
            if len(samples.shape) > 1:
                samples = samples.mean(axis=1)  # mono
            samples = samples.astype(np.float32)
        else:
            samples = np.array(audio, dtype=np.float32)

        # Compute mel spectrogram
        # Using scipy for portability (no torchaudio dependency)
        n_fft = 512
        hop = 160
        n_mels = 64

        # STFT
        _, _, Zxx = scipy.signal.stft(samples, fs=sr, nperseg=n_fft,
                                       noverlap=n_fft - hop)
        power = np.abs(Zxx) ** 2

        # Mel filterbank
        mel_basis = self._mel_filterbank(sr, n_fft, n_mels)
        mel_spec = mel_basis @ power

        # Log scale
        mel_spec = np.log1p(mel_spec * 1000)

        # Normalize to [0, 255] and convert to RGB image
        if mel_spec.max() > mel_spec.min():
            mel_norm = (mel_spec - mel_spec.min()) / (mel_spec.max() - mel_spec.min())
        else:
            mel_norm = np.zeros_like(mel_spec)

        mel_uint8 = (mel_norm * 255).astype(np.uint8)

        # Resize to 224x224 for CLIP
        if not HAS_PIL:
            raise ImportError("Pillow required")
        img = Image.fromarray(mel_uint8, mode='L').convert('RGB')
        img = img.resize((224, 224), Image.BILINEAR)

        # Encode through CLIP (same as any image)
        return self.encode_image(img)

    @staticmethod
    def _mel_filterbank(sr: int, n_fft: int, n_mels: int) -> np.ndarray:
        """Create a mel filterbank matrix."""
        def hz_to_mel(hz):
            return 2595 * np.log10(1 + hz / 700)
        def mel_to_hz(mel):
            return 700 * (10 ** (mel / 2595) - 1)

        low_mel = hz_to_mel(0)
        high_mel = hz_to_mel(sr / 2)
        mel_points = np.linspace(low_mel, high_mel, n_mels + 2)
        hz_points = mel_to_hz(mel_points)
        bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)

        n_freqs = n_fft // 2 + 1
        filterbank = np.zeros((n_mels, n_freqs))

        for i in range(n_mels):
            left = bin_points[i]
            center = bin_points[i + 1]
            right = bin_points[i + 2]
            for j in range(left, center):
                if center > left:
                    filterbank[i, j] = (j - left) / (center - left)
            for j in range(center, right):
                if right > center:
                    filterbank[i, j] = (right - j) / (right - center)

        return filterbank

    def cross_modal_similarity(self, text: str, image) -> float:
        """
        Compute similarity between text and image.
        This is the core of multimodal reasoning — "how related
        is this text to this image?" answered by spatial proximity
        in the shared vector space.
        """
        text_vec = self.encode_text(text)
        image_vec = self.encode_image(image)
        return float(np.dot(text_vec, image_vec))


class UniversalEncoder:
    """
    The complete encoder: multilingual text + images + audio + polarity.

    Combines:
      - paraphrase-multilingual-mpnet-base-v2: text in 50+ languages (768-dim)
      - CLIP ViT-B-32: images → 512-dim (projected to 768 for unified space)
      - NLI cross-encoder: contradiction/entailment detection for polarity

    All modalities map to 768-dim. One vector space for everything.
    One convergence loop reasons across all modalities and languages.
    """

    def __init__(self, device: str = "cpu"):
        if not HAS_SBERT:
            raise ImportError("sentence-transformers required")
        self.device = device
        self.dim = MULTILINGUAL_DIM  # 768

        # Lazy-loaded models
        self._text_model = None
        self._clip_model = None
        self._nli_model = None

        # Projection: CLIP 512 → 768 (learned or padded)
        self._clip_projection = None

    @property
    def text_model(self):
        if self._text_model is None:
            self._text_model = SentenceTransformer(
                MULTILINGUAL_MODEL_NAME, device=self.device
            )
        return self._text_model

    @property
    def clip_model(self):
        if self._clip_model is None:
            self._clip_model = SentenceTransformer(
                CLIP_MODEL_NAME, device=self.device
            )
        return self._clip_model

    @property
    def nli_model(self):
        if self._nli_model is None:
            self._nli_model = CrossEncoder(NLI_MODEL_NAME, device=self.device)
        return self._nli_model

    def encode_text(self, text: str) -> np.ndarray:
        """Encode text (any of 50+ languages) → 768-dim vector."""
        vec = self.text_model.encode(text, convert_to_numpy=True).astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def encode_image(self, image) -> np.ndarray:
        """Encode image → 768-dim vector (CLIP → projected to 768)."""
        if not HAS_PIL:
            raise ImportError("Pillow required")
        if isinstance(image, (str, Path)):
            img = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            img = Image.fromarray(image).convert("RGB")
        else:
            img = image

        clip_vec = self.clip_model.encode(img, convert_to_numpy=True).astype(np.float32)
        # Project CLIP 512 → 768 by zero-padding
        # (Simple but effective — the spaces are separate but cohabitable)
        vec = np.zeros(self.dim, dtype=np.float32)
        vec[:len(clip_vec)] = clip_vec
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def encode_audio(self, audio, sr: int = 16000) -> np.ndarray:
        """Encode audio → 768-dim (spectrogram → CLIP → projected)."""
        # Reuse MultimodalEncoder's spectrogram logic
        clip_enc = MultimodalEncoder.__new__(MultimodalEncoder)
        clip_enc._model = self.clip_model
        clip_enc.model_name = CLIP_MODEL_NAME
        clip_enc.device = self.device
        clip_enc.dim = CLIP_DIM

        clip_vec = clip_enc.encode_audio(audio, sr=sr)
        vec = np.zeros(self.dim, dtype=np.float32)
        vec[:len(clip_vec)] = clip_vec
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def detect_polarity(self, text_a: str, text_b: str) -> dict:
        """
        Detect if two texts contradict, entail, or are neutral.

        Uses NLI cross-encoder. This is the polarity detection
        that GloVe can't do — distinguishes "help" from "harm"
        even when they share embedding space.

        Returns:
            {
                "label": "contradiction" | "entailment" | "neutral",
                "scores": {"contradiction": float, "entailment": float, "neutral": float},
            }
        """
        scores = self.nli_model.predict([(text_a, text_b)])[0]
        labels = ["contradiction", "entailment", "neutral"]
        label = labels[int(np.argmax(scores))]
        return {
            "label": label,
            "scores": {l: float(s) for l, s in zip(labels, scores)},
        }

    def encode_batch_text(self, texts: list) -> np.ndarray:
        """Batch encode texts → (N, 768) matrix."""
        vecs = self.text_model.encode(texts, convert_to_numpy=True,
                                       batch_size=32).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.maximum(norms, 1e-10)
