"""Scoring for an imbalanced binary screening task.

Accuracy is the wrong headline for this dataset. The test split is 42 malignant to
162 benign, so a model that predicts "benign" for every image scores 79.4% while
missing every cancer -- a number that beats several genuinely-trained models and
means nothing. Accuracy also moves when the class ratio moves, which makes it
useless for comparing against a paper that used a different split.

So every model is reported on:

* **Threshold-free** -- ROC-AUC and average precision, which rank-order the model's
  scores without committing to a decision threshold.
* **At a chosen operating point** -- sensitivity, specificity, balanced accuracy and
  MCC. The threshold is chosen on validation data and then applied unchanged to
  test, because choosing it on test is the same leak as tuning on test.
* **With an uncertainty interval** -- 204 test images means a single percentage
  point is roughly two images. Bootstrap intervals keep that visible instead of
  implying three-digit precision.

This module is deliberately pure NumPy/scikit-learn: it carries the claims that
most need testing, and it runs in CI without a TensorFlow install.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

#: Metrics whose fold-level values are averaged into a benchmark summary.
SUMMARY_KEYS = (
    "roc_auc",
    "average_precision",
    "balanced_accuracy",
    "sensitivity",
    "specificity",
    "f1",
    "mcc",
    "accuracy",
)


def confusion_counts(y_true: Sequence[int], y_pred: Sequence[int]) -> dict[str, int]:
    """Return TN/FP/FN/TP, counted explicitly so an all-one-class split is still valid.

    ``sklearn.confusion_matrix`` collapses to a 1x1 matrix when only one label is
    present, which is a real case here: an under-trained model on a small fold can
    predict a single class, and the caller should get zeros rather than an unpack
    error.
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    y_pred = np.asarray(y_pred).astype(int).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")

    return {
        "tn": int(np.sum((y_true == 0) & (y_pred == 0))),
        "fp": int(np.sum((y_true == 0) & (y_pred == 1))),
        "fn": int(np.sum((y_true == 1) & (y_pred == 0))),
        "tp": int(np.sum((y_true == 1) & (y_pred == 1))),
    }


def operating_point_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> dict[str, float]:
    """Threshold-dependent scores, with zero-denominator cases returning 0.0."""
    counts = confusion_counts(y_true, y_pred)
    tn, fp, fn, tp = counts["tn"], counts["fp"], counts["fn"], counts["tp"]

    sensitivity = _ratio(tp, tp + fn)
    specificity = _ratio(tn, tn + fp)
    precision = _ratio(tp, tp + fp)
    total = tn + fp + fn + tp

    return {
        **{k: float(v) for k, v in counts.items()},
        "accuracy": _ratio(tp + tn, total),
        "balanced_accuracy": (sensitivity + specificity) / 2.0,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "npv": _ratio(tn, tn + fn),
        "f1": _ratio(2 * precision * sensitivity, precision + sensitivity),
        "mcc": _mcc(tn, fp, fn, tp),
    }


def _mcc(tn: int, fp: int, fn: int, tp: int) -> float:
    """Matthews correlation coefficient, computed from the counts directly.

    Defined as 0.0 whenever a row or column of the matrix is empty. That case is
    common here -- a model that predicts one class on a small fold hits it -- and 0
    is the right answer for it (no correlation), so it is returned rather than
    treated as an error.
    """
    denominator = float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denominator <= 0:
        return 0.0
    return float((tp * tn - fp * fn) / np.sqrt(denominator))


def threshold_free_metrics(y_true: Sequence[int], y_score: Sequence[float]) -> dict[str, float]:
    """ROC-AUC and average precision. Undefined for a single-class ``y_true``."""
    y_true = np.asarray(y_true).astype(int).ravel()
    y_score = np.asarray(y_score, dtype=float).ravel()
    if len(np.unique(y_true)) < 2:
        return {"roc_auc": float("nan"), "average_precision": float("nan")}
    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "average_precision": float(average_precision_score(y_true, y_score)),
    }


def youden_threshold(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    """Threshold maximising Youden's J (sensitivity + specificity - 1).

    Chosen on validation scores, never on test -- picking the threshold on the same
    data you report is a leak that inflates every downstream number.
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    y_score = np.asarray(y_score, dtype=float).ravel()
    if len(np.unique(y_true)) < 2:
        return 0.5

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    best = int(np.argmax(tpr - fpr))
    threshold = float(thresholds[best])
    # roc_curve prepends an unreachable +inf threshold; clamp it to a usable value.
    return 1.0 if not np.isfinite(threshold) else threshold


def threshold_at_sensitivity(
    y_true: Sequence[int], y_score: Sequence[float], min_sensitivity: float = 0.90
) -> float:
    """Loosest threshold still achieving ``min_sensitivity``.

    This is the screening-oriented operating point: fix the tolerable miss rate for
    malignant lesions first, then accept whatever specificity follows. Falls back to
    the most permissive threshold when the target is unreachable.
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    y_score = np.asarray(y_score, dtype=float).ravel()
    if len(np.unique(y_true)) < 2:
        return 0.5

    _, tpr, thresholds = roc_curve(y_true, y_score)
    feasible = np.flatnonzero(tpr >= min_sensitivity)
    if feasible.size == 0:
        return float(np.min(y_score))
    threshold = float(thresholds[feasible[0]])
    return 1.0 if not np.isfinite(threshold) else threshold


def choose_threshold(
    y_true: Sequence[int],
    y_score: Sequence[float],
    *,
    strategy: str = "youden",
    min_sensitivity: float = 0.90,
) -> float:
    if strategy == "argmax":
        return 0.5
    if strategy == "youden":
        return youden_threshold(y_true, y_score)
    if strategy == "min_sensitivity":
        return threshold_at_sensitivity(y_true, y_score, min_sensitivity)
    raise ValueError(f"unknown threshold strategy {strategy!r}")


def summarise(
    y_true: Sequence[int],
    y_score: Sequence[float],
    *,
    threshold: float = 0.5,
    bootstrap_samples: int = 0,
    seed: int = 11,
) -> dict[str, Any]:
    """Full scorecard for one split at one threshold."""
    y_true = np.asarray(y_true).astype(int).ravel()
    y_score = np.asarray(y_score, dtype=float).ravel()
    y_pred = (y_score >= threshold).astype(int)

    report: dict[str, Any] = {
        "threshold": float(threshold),
        "n": int(y_true.size),
        "n_positive": int(np.sum(y_true == 1)),
        **threshold_free_metrics(y_true, y_score),
        **operating_point_metrics(y_true, y_pred),
    }
    if bootstrap_samples > 0:
        report["ci95"] = bootstrap_ci(
            y_true, y_score, threshold=threshold, samples=bootstrap_samples, seed=seed
        )
    return report


def bootstrap_ci(
    y_true: Sequence[int],
    y_score: Sequence[float],
    *,
    threshold: float = 0.5,
    samples: int = 2000,
    seed: int = 11,
    keys: Iterable[str] = SUMMARY_KEYS,
) -> dict[str, list[float]]:
    """Percentile bootstrap 95% intervals over test cases.

    Resampling is *stratified* within each class so every replicate keeps the real
    42/162 prevalence. An unstratified bootstrap on 42 positives occasionally draws
    a replicate with almost none, which widens the interval for a reason that has
    nothing to do with the model.
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    y_score = np.asarray(y_score, dtype=float).ravel()
    rng = np.random.default_rng(seed)

    pos = np.flatnonzero(y_true == 1)
    neg = np.flatnonzero(y_true == 0)
    if pos.size == 0 or neg.size == 0:
        return {key: [float("nan"), float("nan")] for key in keys}

    draws: dict[str, list[float]] = {key: [] for key in keys}
    for _ in range(samples):
        idx = np.concatenate(
            [rng.choice(pos, pos.size, replace=True), rng.choice(neg, neg.size, replace=True)]
        )
        replicate = {
            **threshold_free_metrics(y_true[idx], y_score[idx]),
            **operating_point_metrics(y_true[idx], (y_score[idx] >= threshold).astype(int)),
        }
        for key in draws:
            draws[key].append(replicate[key])

    return {
        key: [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]
        for key, vals in draws.items()
    }


def aggregate(reports: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    """Mean and sample standard deviation across folds or repeated seeds.

    ``ddof=1`` because these are a sample of runs, not the population of them; with
    5 folds the difference from ``ddof=0`` is not cosmetic.
    """
    if not reports:
        raise ValueError("aggregate() needs at least one report")

    summary: dict[str, dict[str, float]] = {}
    for key in SUMMARY_KEYS:
        values = np.array([float(r[key]) for r in reports if key in r], dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        summary[key] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
            "min": float(values.min()),
            "max": float(values.max()),
            "n_runs": int(values.size),
        }
    return summary


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0
