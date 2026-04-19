"""
Shared test fixtures.

Session-scoped GloVe loading: 990MB loaded once, reused by all tests.
Cuts real-world test suite from ~7 min to ~1 min.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

DATA_DIR = str(Path(__file__).parent.parent / "data")
GLOVE_PATH = str(Path(DATA_DIR) / "glove.6B.300d.txt")


@pytest.fixture(scope="session")
def glove_encoder():
    """Load GloVe once per test session. Returns a ready Encoder."""
    from encoder import Encoder
    enc = Encoder(data_dir=DATA_DIR, dim=300)
    enc.load(GLOVE_PATH)
    return enc


@pytest.fixture
def glove_engine(glove_encoder):
    """
    Fresh Engine with shared GloVe encoder.

    Each test gets its own in-memory NeuronDB (clean state)
    but shares the heavy GloVe vocabulary (no reload).
    """
    from engine import Engine
    engine = Engine(dim=300)
    # Share the already-loaded vocab instead of reloading
    engine.encoder._vocab = glove_encoder._vocab
    engine.encoder._word_list = glove_encoder._word_list
    engine.encoder._faiss_index = glove_encoder._faiss_index
    return engine
