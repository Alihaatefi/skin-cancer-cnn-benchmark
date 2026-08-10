"""Experiment configuration: typed, YAML-backed, validated before anything trains.

Every knob that changes a number lives here rather than in code, for two reasons.
It makes a run reconstructible from the config file alone, and it makes the
*protocol* explicit -- when seven architectures are compared, the claim that they
were trained identically has to be checkable, not asserted.

Validation is deliberately eager: a typo in an architecture name or a fold count of
1 fails in milliseconds rather than after a GPU has been warm for ten minutes.
"""

from __future__ import annotations

import copy
import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a configuration file cannot produce a runnable experiment."""


@dataclass(frozen=True)
class DataConfig:
    """Where the images live and how they are batched.

    ``val_fraction`` carves a validation split out of the *training* directory.
    Passing the test directory as Keras' ``validation_data`` is the common shortcut
    and it turns the test set into a model-selection signal, so the test directory is
    never named here as anything but the final scoring split; see docs/METHODOLOGY.md.
    """

    root: Path = Path("data/New_Skin_Data_1")
    train_subdir: str = "Training"
    test_subdir: str = "Testing"
    image_size: tuple[int, int] = (224, 224)
    batch_size: int = 16
    val_fraction: float = 0.2

    def __post_init__(self) -> None:
        if not 0.0 < self.val_fraction < 0.5:
            raise ConfigError(
                f"data.val_fraction must be in (0.0, 0.5), got {self.val_fraction}. "
                "The training split is small; a larger holdout leaves too little to fit."
            )
        if self.batch_size < 1:
            raise ConfigError(f"data.batch_size must be >= 1, got {self.batch_size}")
        if len(self.image_size) != 2 or any(s < 32 for s in self.image_size):
            raise ConfigError(f"data.image_size must be two ints >= 32, got {self.image_size}")

    @property
    def train_dir(self) -> Path:
        return self.root / self.train_subdir

    @property
    def test_dir(self) -> Path:
        return self.root / self.test_subdir


@dataclass(frozen=True)
class ModelConfig:
    """Which backbone, and how much of it is allowed to move.

    ``fine_tune_layers = 0`` freezes the backbone entirely, which trains the head as
    a linear probe on frozen ImageNet features. A positive value unfreezes that many
    layers from the top for a second, low-learning-rate pass.
    """

    architecture: str = "efficientnetb0"
    dense_units: int = 256
    dropout: float = 0.3
    fine_tune_layers: int = 0

    def __post_init__(self) -> None:
        # Imported here to keep config importable without TensorFlow installed.
        from .models import ARCHITECTURES

        if self.architecture not in ARCHITECTURES:
            known = ", ".join(sorted(ARCHITECTURES))
            raise ConfigError(
                f"model.architecture {self.architecture!r} is not registered. Known: {known}"
            )
        if not 0.0 <= self.dropout < 1.0:
            raise ConfigError(f"model.dropout must be in [0.0, 1.0), got {self.dropout}")
        if self.dense_units < 1:
            raise ConfigError(f"model.dense_units must be >= 1, got {self.dense_units}")
        if self.fine_tune_layers < 0:
            raise ConfigError(
                f"model.fine_tune_layers must be >= 0, got {self.fine_tune_layers}"
            )


@dataclass(frozen=True)
class TrainConfig:
    """Optimisation and regularisation.

    ``class_weight`` matters more than it looks: the test split is 42 malignant to
    162 benign, so an unweighted model can score 79% accuracy by never predicting
    malignant at all -- the one error that matters clinically.
    """

    epochs: int = 60
    learning_rate: float = 1e-3
    fine_tune_learning_rate: float = 1e-5
    early_stopping_patience: int = 10
    reduce_lr_patience: int = 5
    label_smoothing: float = 0.0
    class_weight: bool = True
    augment: bool = True
    monitor: str = "val_auc"

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ConfigError(f"train.epochs must be >= 1, got {self.epochs}")
        if self.learning_rate <= 0 or self.fine_tune_learning_rate <= 0:
            raise ConfigError("train learning rates must be positive")
        if not 0.0 <= self.label_smoothing < 0.5:
            raise ConfigError(
                f"train.label_smoothing must be in [0.0, 0.5), got {self.label_smoothing}"
            )


@dataclass(frozen=True)
class EvalConfig:
    """How predictions become reported numbers.

    ``operating_point`` decides where the probability threshold is placed. Accuracy
    at a hard-coded 0.5 is the wrong summary for an imbalanced screening task, so
    the default picks the threshold on the *validation* split and applies it to test.
    """

    operating_point: str = "youden"
    min_sensitivity: float = 0.90
    bootstrap_samples: int = 2000

    _OPERATING_POINTS = ("argmax", "youden", "min_sensitivity")

    def __post_init__(self) -> None:
        if self.operating_point not in self._OPERATING_POINTS:
            raise ConfigError(
                f"eval.operating_point must be one of {self._OPERATING_POINTS}, "
                f"got {self.operating_point!r}"
            )
        if not 0.0 < self.min_sensitivity <= 1.0:
            raise ConfigError(
                f"eval.min_sensitivity must be in (0.0, 1.0], got {self.min_sensitivity}"
            )
        if self.bootstrap_samples < 0:
            raise ConfigError("eval.bootstrap_samples must be >= 0")


@dataclass(frozen=True)
class ExperimentConfig:
    """One complete, runnable experiment."""

    name: str = "unnamed"
    seed: int = 11
    folds: int = 5
    repeats: int = 1
    deterministic: bool = True
    output_dir: Path = Path("results")
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    def __post_init__(self) -> None:
        if self.folds < 2:
            raise ConfigError(
                f"folds must be >= 2 for cross-validation, got {self.folds}. "
                "Use the `train` command for a single holdout run."
            )
        if self.repeats < 1:
            raise ConfigError(f"repeats must be >= 1, got {self.repeats}")

    @property
    def run_dir(self) -> Path:
        return self.output_dir / self.name

    def to_dict(self) -> dict[str, Any]:
        """Serialise back to plain types, so a result file can carry its own config."""
        return _as_plain(dataclasses.asdict(self))


_SECTIONS = {
    "data": DataConfig,
    "model": ModelConfig,
    "train": TrainConfig,
    "eval": EvalConfig,
}


def load_config(path: str | Path, overrides: Mapping[str, Any] | None = None) -> ExperimentConfig:
    """Build an :class:`ExperimentConfig` from YAML, applying dotted-key overrides.

    Overrides use CLI-style dotted paths so a sweep can vary one knob without
    writing a config file per point::

        load_config("configs/efficientnetb0.yaml", {"train.epochs": 5, "seed": 3})
    """
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")

    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: top level must be a mapping, got {type(raw).__name__}")

    raw = copy.deepcopy(raw)
    for dotted, value in (overrides or {}).items():
        _apply_override(raw, dotted, value)

    raw.setdefault("name", path.stem)
    return from_dict(raw)


def from_dict(raw: Mapping[str, Any]) -> ExperimentConfig:
    """Construct a config from an already-parsed mapping, rejecting unknown keys."""
    payload = dict(raw)
    sections: dict[str, Any] = {}
    for key, cls in _SECTIONS.items():
        section = payload.pop(key, None) or {}
        if not isinstance(section, Mapping):
            raise ConfigError(f"{key!r} must be a mapping, got {type(section).__name__}")
        sections[key] = _build(cls, section, prefix=key)

    top = _coerce(ExperimentConfig, payload, prefix="")
    return ExperimentConfig(**top, **sections)


def _build(cls: type, section: Mapping[str, Any], *, prefix: str) -> Any:
    return cls(**_coerce(cls, section, prefix=prefix))


def _coerce(cls: type, section: Mapping[str, Any], *, prefix: str) -> dict[str, Any]:
    """Filter to the dataclass' own fields, coercing Path/tuple types on the way in."""
    fields = {f.name: f for f in dataclasses.fields(cls) if f.name not in _SECTIONS}
    unknown = set(section) - set(fields)
    if unknown:
        label = f"{prefix}." if prefix else ""
        raise ConfigError(
            f"unknown config key(s): {', '.join(sorted(label + u for u in unknown))}. "
            f"Valid keys under {prefix or '<top level>'}: {', '.join(sorted(fields))}"
        )

    out: dict[str, Any] = {}
    for key, value in section.items():
        annotation = fields[key].type
        if "Path" in str(annotation) and value is not None:
            value = Path(value)
        elif "tuple" in str(annotation) and isinstance(value, list):
            value = tuple(value)
        out[key] = value
    return out


def _apply_override(raw: dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    cursor = raw
    for part in parts[:-1]:
        nested = cursor.setdefault(part, {})
        if not isinstance(nested, dict):
            raise ConfigError(f"override {dotted!r} traverses non-mapping key {part!r}")
        cursor = nested
    cursor[parts[-1]] = value


def _as_plain(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: _as_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_plain(v) for v in value]
    return value
