"""Dataset manifests, canonical labels, stratified splits and tf.data inputs.

The manifest layer (everything above ``make_dataset``) is pure pandas, so splits and
label handling are unit-testable without TensorFlow or a GPU -- which matters,
because a silent label flip is the class of bug that produces a plausible-looking
number that happens to be measuring the wrong thing.

Two hazards this module is built around:

*Label naming is inconsistent in the source data.* The Kaggle export uses
``malignant``/``benign`` in some copies and ``Cancer``/``Non_cancer`` in others.
Every raw folder name is mapped through :data:`LABEL_ALIASES` to a canonical label,
and an unrecognised folder name is a hard error rather than a silently-added third
class.

*Class order is not alphabetical by accident.* ``malignant`` must be index 1, since
every metric in :mod:`.metrics` treats index 1 as the positive class. That ordering
is pinned here rather than inherited from a loader's alphabetical sort, which would
put ``Cancer`` before ``Non_cancer`` in one export and ``benign`` before
``malignant`` in another -- flipping sensitivity and specificity between runs of the
same code, on nothing but a folder name.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

#: Canonical class names, ordered so that index 1 is the positive (malignant) class.
CLASS_NAMES: tuple[str, str] = ("benign", "malignant")

#: Raw directory names seen across the published copies of the Kaggle export.
LABEL_ALIASES: dict[str, str] = {
    "benign": "benign",
    "non_cancer": "benign",
    "noncancer": "benign",
    "non-cancer": "benign",
    "nevus": "benign",
    "malignant": "malignant",
    "cancer": "malignant",
    "melanoma": "malignant",
}

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"})


class DatasetError(RuntimeError):
    """Raised when the on-disk dataset cannot produce a usable manifest."""


def scan_split(directory: str | Path) -> pd.DataFrame:
    """Build a manifest of one split directory of class subfolders.

    Returns columns ``filepath``, ``raw_label``, ``label`` and ``y`` (0/1), sorted by
    path so the manifest is byte-identical across filesystems with different
    directory ordering -- otherwise a "seeded" split is only seeded per machine.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise DatasetError(
            f"split directory not found: {directory}\n"
            "Fetch the dataset first: python scripts/download_dataset.py"
        )

    rows: list[dict[str, Any]] = []
    unknown: set[str] = set()
    for class_dir in sorted(p for p in directory.iterdir() if p.is_dir()):
        canonical = LABEL_ALIASES.get(class_dir.name.strip().lower())
        if canonical is None:
            unknown.add(class_dir.name)
            continue
        for image in sorted(class_dir.iterdir()):
            if image.suffix.lower() in IMAGE_SUFFIXES:
                rows.append(
                    {
                        "filepath": str(image),
                        "raw_label": class_dir.name,
                        "label": canonical,
                        "y": CLASS_NAMES.index(canonical),
                    }
                )

    if unknown:
        raise DatasetError(
            f"{directory}: unrecognised class folder(s) {sorted(unknown)}. "
            f"Add them to LABEL_ALIASES or remove them; silently ignoring a folder "
            f"would change the task without changing the code."
        )
    if not rows:
        raise DatasetError(f"{directory}: no images found under any class subfolder")

    frame = pd.DataFrame(rows).sort_values("filepath", ignore_index=True)
    present = set(frame["label"])
    if present != set(CLASS_NAMES):
        raise DatasetError(
            f"{directory}: expected both classes {CLASS_NAMES}, found {sorted(present)}"
        )
    return frame


def load_manifests(data_config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Manifests for the train and test directories named by the config."""
    return scan_split(data_config.train_dir), scan_split(data_config.test_dir)


def stratified_holdout(
    frame: pd.DataFrame, *, val_fraction: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split off a class-balanced validation set from the training manifest.

    Every early-stopping decision, learning-rate drop and threshold choice is made
    against this split, so the test directory stays untouched until final scoring.
    """
    from sklearn.model_selection import train_test_split

    train_idx, val_idx = train_test_split(
        np.arange(len(frame)),
        test_size=val_fraction,
        random_state=seed,
        stratify=frame["y"].to_numpy(),
    )
    return (
        frame.iloc[np.sort(train_idx)].reset_index(drop=True),
        frame.iloc[np.sort(val_idx)].reset_index(drop=True),
    )


def stratified_folds(
    frame: pd.DataFrame, *, n_splits: int, seed: int
) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
    """Yield ``(train, val)`` manifests for stratified k-fold cross-validation.

    With 84 training images a single holdout of 17 is too small for its score to
    mean much -- a two-image swing moves it by 12 points. Cross-validation spends
    k times the compute to put every image in a validation fold exactly once, which
    is what makes a spread reportable instead of a single fragile number.
    """
    from sklearn.model_selection import StratifiedKFold

    labels = frame["y"].to_numpy()
    minority = int(np.min(np.bincount(labels)))
    if n_splits > minority:
        raise DatasetError(
            f"cannot build {n_splits} stratified folds: the minority class has only "
            f"{minority} example(s) in this split"
        )

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_idx, val_idx in splitter.split(np.zeros(len(frame)), labels):
        yield (
            frame.iloc[train_idx].reset_index(drop=True),
            frame.iloc[val_idx].reset_index(drop=True),
        )


def class_weights(frame: pd.DataFrame) -> dict[int, float]:
    """Inverse-frequency weights, normalised to mean 1.0.

    Normalising keeps the loss on the same scale as an unweighted run, so learning
    rates and early-stopping patience transfer between weighted and unweighted
    configurations instead of needing a silent re-tune.
    """
    counts = frame["y"].value_counts().to_dict()
    total = sum(counts.values())
    n_classes = len(counts)
    raw = {int(k): total / (n_classes * v) for k, v in counts.items()}
    mean = sum(raw.values()) / len(raw)
    return {k: v / mean for k, v in raw.items()}


def describe(frame: pd.DataFrame) -> dict[str, Any]:
    """Counts and prevalence, for the run manifest and the dataset-card table."""
    counts = frame["label"].value_counts().to_dict()
    total = int(len(frame))
    return {
        "n": total,
        "counts": {name: int(counts.get(name, 0)) for name in CLASS_NAMES},
        "prevalence_malignant": float(counts.get("malignant", 0) / total) if total else 0.0,
    }


def make_dataset(
    frame: pd.DataFrame,
    *,
    preprocess,
    image_size: tuple[int, int],
    batch_size: int,
    shuffle: bool = False,
    augment: bool = False,
    seed: int = 11,
):
    """Build a ``tf.data.Dataset`` of ``(image, one_hot_label)`` from a manifest.

    Augmentation is applied *after* preprocessing-independent decoding but is only
    ever enabled for training folds. It is limited to flips, small rotations, zooms
    and translations: colour jitter is deliberately excluded, because hue and
    saturation carry diagnostic signal in dermoscopy and perturbing them trains the
    model to ignore a real feature.
    """
    import tensorflow as tf

    paths = frame["filepath"].to_numpy()
    labels = tf.keras.utils.to_categorical(frame["y"].to_numpy(), num_classes=len(CLASS_NAMES))

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        dataset = dataset.shuffle(len(frame), seed=seed, reshuffle_each_iteration=True)

    def load(path, label):
        image = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        image = tf.image.resize(image, image_size, method="bilinear")
        image.set_shape((*image_size, 3))
        return tf.cast(image, tf.float32), label

    dataset = dataset.map(load, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)

    if augment:
        augmenter = build_augmenter(seed=seed)
        dataset = dataset.map(
            lambda x, y: (augmenter(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    # Preprocessing runs last so augmentation always sees raw [0, 255] pixels,
    # whatever normalisation the backbone happens to want.
    dataset = dataset.map(
        lambda x, y: (preprocess(x), y), num_parallel_calls=tf.data.AUTOTUNE
    )
    return dataset.prefetch(tf.data.AUTOTUNE)


def build_augmenter(*, seed: int = 11):
    """Geometric-only augmentation pipeline, as a Keras layer."""
    import keras
    from keras import layers

    return keras.Sequential(
        [
            layers.RandomFlip("horizontal_and_vertical", seed=seed),
            layers.RandomRotation(0.15, fill_mode="reflect", seed=seed),
            layers.RandomZoom(0.15, fill_mode="reflect", seed=seed),
            layers.RandomTranslation(0.1, 0.1, fill_mode="reflect", seed=seed),
        ],
        name="augmentation",
    )


def labels_from(frame: pd.DataFrame) -> np.ndarray:
    """Ground-truth vector in the manifest's own row order.

    Predictions must be aligned to this, which is why every inference path in the
    package iterates an unshuffled dataset built from the same frame.
    """
    return frame["y"].to_numpy().astype(int)


def as_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return frame.to_dict(orient="records")


def summarise_splits(splits: Sequence[tuple[str, pd.DataFrame]]) -> str:
    """Human-readable split table for stdout and the run log."""
    lines = [f"{'split':<12}{'n':>6}{'benign':>9}{'malignant':>11}{'prev.':>8}"]
    for name, frame in splits:
        info = describe(frame)
        lines.append(
            f"{name:<12}{info['n']:>6}{info['counts']['benign']:>9}"
            f"{info['counts']['malignant']:>11}{info['prevalence_malignant']:>8.1%}"
        )
    return "\n".join(lines)
