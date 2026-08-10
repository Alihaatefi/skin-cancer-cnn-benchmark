"""A reproducible benchmark for CNN-based malignant/benign skin-lesion classification.

The package re-runs the four architectures of Surendren & Sumitha (ICDICI 2024) and
the three architectures added by this project under a *single* evaluation protocol,
so that per-model differences reflect the models rather than the harness.

Public surface:

``config``   experiment definitions, loaded from YAML and validated up front
``data``     dataset manifests, canonical labels, stratified splits, tf.data inputs
``models``   the architecture registry and transfer-learning head
``metrics``  threshold-free and operating-point scoring with bootstrap intervals
``train``    single-run and cross-validated training loops
``evaluate`` scoring of a trained model against a held-out split
``gradcam``  saliency overlays for qualitative inspection
"""

from __future__ import annotations

__version__ = "1.0.0"

__all__ = [
    "__version__",
    "config",
    "data",
    "evaluate",
    "gradcam",
    "metrics",
    "models",
    "train",
]
