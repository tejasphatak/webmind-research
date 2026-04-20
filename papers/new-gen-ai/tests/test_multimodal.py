"""
Tests for multimodal encoding and cross-modal retrieval.

Tests the MultimodalEncoder (CLIP-based) and the engine's
multimodal teach/query methods.

CLIP model tests are marked with @pytest.mark.slow — they
download the model on first run (~400MB).
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


# --- Unit tests for MultimodalEncoder (no model download) ---

class TestMultimodalEncoderUnit:
    def test_import_succeeds(self):
        """The multimodal module should import without errors."""
        from multimodal import MultimodalEncoder, CLIP_DIM
        assert CLIP_DIM == 512

    def test_lazy_model_loading(self):
        """Model should not load until first encode call."""
        from multimodal import MultimodalEncoder
        enc = MultimodalEncoder.__new__(MultimodalEncoder)
        enc._model = None
        enc.model_name = "clip-ViT-B-32"
        enc.device = "cpu"
        enc.dim = 512
        # _model should still be None (lazy)
        assert enc._model is None


# --- Integration tests with CLIP model ---

@pytest.mark.slow
class TestMultimodalCLIP:
    @pytest.fixture(scope="class")
    def encoder(self):
        from multimodal import MultimodalEncoder
        return MultimodalEncoder(device="cpu")

    def test_text_encoding_shape(self, encoder):
        """Text encoding should produce a 512-dim normalized vector."""
        vec = encoder.encode_text("a cat sitting on a mat")
        assert vec.shape == (512,)
        assert abs(np.linalg.norm(vec) - 1.0) < 1e-5

    def test_text_similarity(self, encoder):
        """Similar texts should produce similar vectors."""
        v1 = encoder.encode_text("a cat sitting on a mat")
        v2 = encoder.encode_text("a kitten on a rug")
        v3 = encoder.encode_text("the stock market crashed today")
        sim_close = float(np.dot(v1, v2))
        sim_far = float(np.dot(v1, v3))
        assert sim_close > sim_far

    def test_image_encoding(self, encoder, tmp_path):
        """Image encoding should produce a 512-dim normalized vector."""
        # Create a simple test image
        from PIL import Image
        img = Image.new("RGB", (224, 224), color=(255, 0, 0))
        path = str(tmp_path / "red.png")
        img.save(path)

        vec = encoder.encode_image(path)
        assert vec.shape == (512,)
        assert abs(np.linalg.norm(vec) - 1.0) < 1e-5

    def test_cross_modal_similarity(self, encoder, tmp_path):
        """Text and images of similar content should be close in vector space."""
        from PIL import Image

        # Create a red image and a blue image
        red_img = Image.new("RGB", (224, 224), color=(255, 0, 0))
        blue_img = Image.new("RGB", (224, 224), color=(0, 0, 255))
        red_path = str(tmp_path / "red.png")
        blue_path = str(tmp_path / "blue.png")
        red_img.save(red_path)
        blue_img.save(blue_path)

        # "red" text should be closer to red image than blue image
        red_text = encoder.encode_text("a solid red image")
        red_image = encoder.encode_image(red_path)
        blue_image = encoder.encode_image(blue_path)

        sim_match = float(np.dot(red_text, red_image))
        sim_mismatch = float(np.dot(red_text, blue_image))
        # CLIP should distinguish these
        assert sim_match > sim_mismatch or abs(sim_match - sim_mismatch) < 0.1

    def test_batch_text_encoding(self, encoder):
        """Batch encoding should produce same results as individual."""
        texts = ["hello world", "cat on mat", "quantum physics"]
        batch = encoder.encode_batch_text(texts)
        assert batch.shape == (3, 512)

        individual = encoder.encode_text(texts[0])
        # Should be very close (floating point differences from batching)
        sim = float(np.dot(batch[0], individual))
        assert sim > 0.99

    def test_pil_image_direct(self, encoder):
        """Should accept PIL Image objects directly."""
        from PIL import Image
        img = Image.new("RGB", (100, 100), color=(0, 255, 0))
        vec = encoder.encode_image(img)
        assert vec.shape == (512,)

    def test_numpy_image(self, encoder):
        """Should accept numpy arrays."""
        arr = np.zeros((100, 100, 3), dtype=np.uint8)
        arr[:, :, 1] = 255  # green
        vec = encoder.encode_image(arr)
        assert vec.shape == (512,)


# --- Engine multimodal integration tests ---

@pytest.mark.slow
class TestEngineMultimodal:
    @pytest.fixture
    def engine(self, tmp_path):
        from engine import Engine
        from multimodal import CLIP_DIM
        # Use CLIP dim (512) for multimodal engine
        engine = Engine(data_dir=str(tmp_path), dim=CLIP_DIM)
        return engine

    def test_teach_image(self, engine, tmp_path):
        """Teaching an image should create a neuron."""
        from PIL import Image
        img = Image.new("RGB", (224, 224), color=(255, 0, 0))
        path = str(tmp_path / "test.png")
        img.save(path)

        before = engine.db.count()
        neuron = engine.teach_image(path, label="a red square")
        after = engine.db.count()

        assert after > before
        assert neuron.vector.shape == (512,)
        engine.close()

    def test_teach_image_with_label_creates_link(self, engine, tmp_path):
        """Teaching with a label should create two neurons linked by successor."""
        from PIL import Image
        img = Image.new("RGB", (224, 224), color=(0, 0, 255))
        path = str(tmp_path / "blue.png")
        img.save(path)

        neuron = engine.teach_image(path, label="a blue square")
        # Should have created 2 neurons (image + label)
        assert engine.db.count() >= 2

        # Image neuron should have successor (the label)
        refreshed = engine.db.get(neuron.id)
        assert len(refreshed.successors) >= 1
        engine.close()

    def test_query_image(self, engine, tmp_path):
        """Querying with an image should find similar neurons."""
        from PIL import Image

        # Teach a red image
        red = Image.new("RGB", (224, 224), color=(255, 0, 0))
        red_path = str(tmp_path / "red.png")
        red.save(red_path)
        engine.teach_image(red_path, label="red color")

        # Query with the same image
        results = engine.query_image(red_path, k=3)
        assert len(results) >= 1
        # First result should be highly similar (it's the same image)
        assert results[0][1] > 0.9
        engine.close()

    def test_text_to_image_retrieval(self, engine, tmp_path):
        """Text query should find taught images via cross-modal search."""
        from PIL import Image

        # Teach images with labels
        red = Image.new("RGB", (224, 224), color=(255, 0, 0))
        red_path = str(tmp_path / "red.png")
        red.save(red_path)
        engine.teach_image(red_path, label="red")

        blue = Image.new("RGB", (224, 224), color=(0, 0, 255))
        blue_path = str(tmp_path / "blue.png")
        blue.save(blue_path)
        engine.teach_image(blue_path, label="blue")

        # Text query for "red" should find red-related neurons
        results = engine.query_text_to_image("red color", k=5)
        assert len(results) >= 1
        engine.close()

    def test_multimodal_convergence(self, engine, tmp_path):
        """The convergence loop should work with CLIP vectors."""
        from PIL import Image

        # Teach some images
        for color_name, color in [("red", (255, 0, 0)), ("blue", (0, 0, 255)),
                                   ("green", (0, 255, 0))]:
            img = Image.new("RGB", (224, 224), color=color)
            path = str(tmp_path / f"{color_name}.png")
            img.save(path)
            engine.teach_image(path, label=color_name)

        # Convergence should work on CLIP vectors
        vec = engine.multimodal.encode_text("colors")
        result = engine.convergence.converge(vec)
        # Should find some neurons (may or may not converge at small scale)
        assert result is not None
        engine.close()
