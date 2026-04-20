"""
Shared test fixtures.

Session-scoped GloVe loading: indexed once, reused by all tests.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

DATA_DIR = str(Path(__file__).parent.parent / "data")
GLOVE_PATH = str(Path(DATA_DIR) / "glove.6B.300d.txt")


@pytest.fixture(scope="session")
def glove_encoder():
    """Index GloVe once per test session. Returns a ready Encoder."""
    from encoder import Encoder
    enc = Encoder(data_dir=DATA_DIR, dim=300)
    enc.load(GLOVE_PATH)
    return enc


@pytest.fixture
def glove_engine(glove_encoder):
    """
    Fresh Engine with shared GloVe encoder.

    Each test gets its own in-memory NeuronDB (clean state)
    but shares the GloVe offset index (no re-scan).
    """
    from engine import Engine
    engine = Engine(dim=300)
    # Share the already-indexed GloVe offsets and cache
    engine.encoder._glove_offsets = glove_encoder._glove_offsets
    engine.encoder._glove_path = glove_encoder._glove_path
    engine.encoder._glove_cache = glove_encoder._glove_cache
    engine.encoder._glove_dim = glove_encoder._glove_dim
    engine.encoder._fixed_dim = glove_encoder._fixed_dim
    return engine
