"""Seed control and run provenance.

Reproducibility here means two separate things, and the distinction matters when
reporting numbers:

*Seeded* -- every stochastic choice (weight init, shuffling, augmentation, the
bootstrap resampler) is drawn from one declared seed, so a re-run on the same
machine reproduces the same figures.

*Deterministic* -- cuDNN is additionally forced onto deterministic kernels, so the
same seed reproduces the same figures on different machines with the same software
stack. This costs roughly 10-25% throughput on GPU, so it is opt-in via
``deterministic=True`` (the benchmark entry point turns it on).

Because the dataset is small, a single seed is not evidence. Every headline number
in this repository is reported as a mean over repeated seeded runs with a spread,
never as one run's best epoch.
"""

from __future__ import annotations

import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any

_DETERMINISM_ENV = {
    "TF_DETERMINISTIC_OPS": "1",
    "TF_CUDNN_DETERMINISTIC": "1",
    "PYTHONHASHSEED": None,  # filled in from the seed at call time
}


def seed_everything(seed: int, *, deterministic: bool = False) -> int:
    """Seed Python, NumPy and TensorFlow. Returns the seed, for logging.

    Must run before TensorFlow allocates any op, so call it at process start --
    the CLI does this before importing any training code.
    """
    env = dict(_DETERMINISM_ENV)
    env["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        os.environ.update({k: v for k, v in env.items() if v is not None})
    else:
        os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover - numpy is a hard dependency in practice
        pass

    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
        if deterministic and hasattr(tf.config.experimental, "enable_op_determinism"):
            tf.config.experimental.enable_op_determinism()
    except ImportError:
        # Seeding is useful on its own for the pure-NumPy metric paths, which are
        # exercised in CI without a TensorFlow install.
        pass

    return seed


def run_manifest(**extra: Any) -> dict[str, Any]:
    """Capture what a reader needs in order to trust or reproduce a result file."""
    manifest: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git_commit": _git_commit(),
    }

    try:
        import tensorflow as tf

        manifest["tensorflow"] = tf.__version__
        manifest["gpus"] = [d.name for d in tf.config.list_physical_devices("GPU")]
    except ImportError:
        manifest["tensorflow"] = None
        manifest["gpus"] = []

    manifest.update(extra)
    return manifest


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None
