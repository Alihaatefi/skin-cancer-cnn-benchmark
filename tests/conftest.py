"""Shared fixtures.

The suite is built to run without TensorFlow. The layers that carry the claims most
likely to be silently wrong -- label mapping, split stratification, threshold choice,
metric arithmetic -- are pure NumPy/pandas and are always exercised. Tests that need
a real Keras graph are marked ``tensorflow`` and skip cleanly when it is absent.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

HAS_TENSORFLOW = importlib.util.find_spec("tensorflow") is not None

requires_tensorflow = pytest.mark.skipif(
    not HAS_TENSORFLOW, reason="TensorFlow is not installed in this environment"
)


def pytest_collection_modifyitems(config, items):
    skip = pytest.mark.skip(reason="TensorFlow is not installed in this environment")
    for item in items:
        if "tensorflow" in item.keywords and not HAS_TENSORFLOW:
            item.add_marker(skip)


@pytest.fixture
def synthetic_dataset(tmp_path: Path) -> Path:
    """A tiny on-disk dataset with the same directory shape as the real one.

    Deliberately uses the ``Cancer``/``Non_cancer`` folder spelling rather than the
    canonical one, so the alias mapping is on the critical path of most data tests
    instead of being tested only in isolation.
    """
    from PIL import Image

    rng = np.random.default_rng(0)
    root = tmp_path / "dataset"
    layout = {
        ("Training", "Cancer"): 12,
        ("Training", "Non_cancer"): 36,
        ("Testing", "Cancer"): 6,
        ("Testing", "Non_cancer"): 18,
    }

    for (split, klass), count in layout.items():
        directory = root / split / klass
        directory.mkdir(parents=True)
        for index in range(count):
            pixels = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(directory / f"{klass.lower()}_{index:03d}.jpg")

    return root


@pytest.fixture
def separable_scores() -> tuple[np.ndarray, np.ndarray]:
    """Imbalanced labels plus scores that are informative but not perfect.

    42/162 matches the real test split's prevalence, so tests exercise the same
    degenerate-accuracy regime the metric module is designed around.
    """
    rng = np.random.default_rng(11)
    y_true = np.concatenate([np.ones(42, dtype=int), np.zeros(162, dtype=int)])
    scores = np.concatenate(
        [rng.normal(0.70, 0.18, 42), rng.normal(0.35, 0.18, 162)]
    )
    return y_true, np.clip(scores, 0.0, 1.0)
