"""Render the README figures from the recorded result files.

Figures are generated rather than pasted so they stay attached to the numbers in
``results/baseline_run/`` and ``results/benchmark.csv``: change a matrix or re-run the
benchmark, re-run this, and the chart follows.
Each figure is written twice, for light and dark viewers, and the README selects
between them with a ``<picture>`` element.

Design constraints applied throughout: one measure per axis, no dual scales, a
validated colourblind-safe categorical palette, direct labels rather than a legend
wherever two series fit, and recessive grid lines.

    python scripts/render_figures.py
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from skin_cancer_benchmark.metrics import operating_point_metrics  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "baseline_run"
BENCHMARK = ROOT / "results" / "benchmark.csv"
ASSETS = ROOT / "assets" / "figures"

#: The test split every model is scored on: 42 malignant, 162 benign.
TEST_POSITIVES, TEST_NEGATIVES = 42, 162

MODEL_ORDER = ("alexnet", "inceptionv3", "efficientnetb0")


@dataclass(frozen=True)
class Theme:
    """Colour tokens for one viewing mode.

    Both palettes are the same eight-hue categorical system re-stepped for their own
    surface, and the three slots used here pass the all-pairs CVD and normal-vision
    separation checks in both modes.
    """

    name: str
    surface: str
    ink: str
    ink_secondary: str
    muted: str
    grid: str
    axis: str
    series: tuple[str, str, str]
    ramp: str

    @property
    def suffix(self) -> str:
        return "" if self.name == "light" else "-dark"


LIGHT = Theme(
    name="light",
    surface="#fcfcfb",
    ink="#0b0b0b",
    ink_secondary="#52514e",
    muted="#898781",
    grid="#e1e0d9",
    axis="#c3c2b7",
    series=("#2a78d6", "#eb6834", "#1baf7a"),
    ramp="Blues",
)

DARK = Theme(
    name="dark",
    surface="#1a1a19",
    ink="#ffffff",
    ink_secondary="#c3c2b7",
    muted="#898781",
    grid="#2c2c2a",
    axis="#383835",
    series=("#3987e5", "#d95926", "#199e70"),
    ramp="Blues",
)


def apply_theme(theme: Theme) -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": theme.surface,
            "axes.facecolor": theme.surface,
            "savefig.facecolor": theme.surface,
            "text.color": theme.ink,
            "axes.labelcolor": theme.ink_secondary,
            "axes.edgecolor": theme.axis,
            "xtick.color": theme.muted,
            "ytick.color": theme.muted,
            "grid.color": theme.grid,
            "grid.linewidth": 0.8,
            "axes.grid": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "lines.linewidth": 2.0,
            "figure.dpi": 160,
        }
    )


def load() -> tuple[dict, dict]:
    matrices = json.loads((RESULTS / "confusion_matrices.json").read_text(encoding="utf-8"))
    history = json.loads(
        (ROOT / "assets" / "baseline_run" / "training_history.json").read_text(encoding="utf-8")
    )
    return matrices, history


def scorecard(matrix: list[list[int]]) -> dict[str, float]:
    """Expand a confusion matrix back into the metric set, via the package's own code."""
    (tn, fp), (fn, tp) = matrix
    y_true = np.array([0] * (tn + fp) + [1] * (fn + tp))
    y_pred = np.array([0] * tn + [1] * fp + [0] * fn + [1] * tp)
    return operating_point_metrics(y_true, y_pred)


def figure_training_curves(history: dict, matrices: dict, theme: Theme) -> Path:
    """Small multiples: loss and accuracy per model, training against validation.

    Loss and accuracy get separate rows rather than a shared twin axis -- two
    measures on two scales in one frame invite reading a crossing point that has no
    meaning.
    """
    apply_theme(theme)
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 5.4), sharex="col")

    for column, key in enumerate(MODEL_ORDER):
        record = history[key]
        title = matrices["models"][key]["display_name"]
        epochs = np.arange(1, len(record["loss"]) + 1)

        for row, (metric, label) in enumerate([("loss", "Loss"), ("accuracy", "Accuracy")]):
            ax = axes[row, column]
            ax.plot(epochs, record[metric], color=theme.series[0], label="train")

            validation = record.get(f"val_{metric}", [])
            if validation:
                ax.plot(epochs, validation, color=theme.series[1], label="validation")

            ax.set_xlim(1, len(epochs))
            ax.grid(True, axis="y")
            ax.grid(False, axis="x")
            if row == 0:
                ax.set_title(title, color=theme.ink, pad=8)
                ax.set_yscale("log")
            else:
                ax.set_ylim(0, 1.02)
                ax.set_xlabel("Epoch")
            if column == 0:
                ax.set_ylabel(label)

            # Direct labels at the series end beat a legend box: identity is read
            # where the eye already is, and it survives a colourblind viewer.
            _label_end(ax, epochs, record[metric], "train", theme.series[0], theme)
            if validation:
                _label_end(ax, epochs, validation, "validation", theme.series[1], theme)

    axes[0, 0].set_ylabel("Loss (log scale)")
    fig.suptitle(
        "Baseline run: training dynamics on a single 84/204 split",
        color=theme.ink,
        fontsize=12,
        fontweight="bold",
        x=0.01,
        ha="left",
        y=0.99,
    )
    fig.text(
        0.01,
        0.005,
        "AlexNet trains from random initialisation for 100 epochs and has no "
        "validation curve in this run; the pretrained backbones run 30.",
        color=theme.muted,
        fontsize=8,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.955))
    return _save(fig, f"training-curves{theme.suffix}.png")


def _label_end(ax, xs, ys, text: str, colour: str, theme: Theme) -> None:
    ax.annotate(
        text,
        xy=(xs[-1], ys[-1]),
        xytext=(4, 0),
        textcoords="offset points",
        color=colour,
        fontsize=7.5,
        va="center",
        clip_on=False,
    )


def figure_confusion(matrices: dict, theme: Theme) -> Path:
    """Three confusion matrices, on a shared count scale so cells are comparable."""
    apply_theme(theme)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.7))
    labels = ["benign", "malignant"]
    ceiling = max(
        max(max(row) for row in matrices["models"][k]["matrix"]) for k in MODEL_ORDER
    )

    for ax, key in zip(axes, MODEL_ORDER, strict=True):
        record = matrices["models"][key]
        matrix = np.array(record["matrix"])
        scores = scorecard(record["matrix"])

        ax.imshow(matrix, cmap=theme.ramp, vmin=0, vmax=ceiling)
        ax.set_title(record["display_name"], color=theme.ink, pad=8)
        ax.set_xticks(range(2), labels)
        ax.set_yticks(range(2), labels)
        ax.set_xlabel("predicted")
        if key == MODEL_ORDER[0]:
            ax.set_ylabel("true")
        ax.grid(False)

        for i in range(2):
            for j in range(2):
                # Ink colour flips against cell darkness so every count stays legible
                # regardless of where the cell sits on the ramp.
                intensity = matrix[i, j] / ceiling
                ax.text(
                    j,
                    i,
                    f"{matrix[i, j]}",
                    ha="center",
                    va="center",
                    fontsize=13,
                    fontweight="bold",
                    color="#fcfcfb" if intensity > 0.55 else "#0b0b0b",
                )

        ax.text(
            0.5,
            -0.34,
            f"sensitivity {scores['sensitivity']:.2f}   "
            f"specificity {scores['specificity']:.2f}",
            transform=ax.transAxes,
            ha="center",
            color=theme.ink_secondary,
            fontsize=8.5,
        )

    fig.suptitle(
        "Where the errors land: 42 malignant and 162 benign test images",
        color=theme.ink,
        fontsize=12,
        fontweight="bold",
        x=0.01,
        ha="left",
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.93))
    return _save(fig, f"confusion-matrices{theme.suffix}.png")


def figure_operating_points(matrices: dict, theme: Theme) -> Path:
    """Sensitivity against specificity per model, as a paired dot plot.

    A single accuracy bar per model would hide the point of this chart: the three
    models fail in different directions, and one of them buys its recall by calling
    almost everything malignant.
    """
    apply_theme(theme)
    fig, ax = plt.subplots(figsize=(8.4, 3.6))

    rows = list(reversed(MODEL_ORDER))
    positions = np.arange(len(rows))

    for y, key in zip(positions, rows, strict=True):
        scores = scorecard(matrices["models"][key]["matrix"])
        sensitivity, specificity = scores["sensitivity"], scores["specificity"]

        ax.plot(
            [sensitivity, specificity],
            [y, y],
            color=theme.axis,
            linewidth=2,
            solid_capstyle="round",
            zorder=1,
        )
        ax.scatter(
            [sensitivity], [y], s=110, color=theme.series[1], zorder=3,
            edgecolor=theme.surface, linewidth=2,
        )
        ax.scatter(
            [specificity], [y], s=110, color=theme.series[0], zorder=3,
            edgecolor=theme.surface, linewidth=2,
        )

        for value, colour, offset in (
            (sensitivity, theme.series[1], 12),
            (specificity, theme.series[0], -12),
        ):
            ax.annotate(
                f"{value:.2f}",
                xy=(value, y),
                xytext=(0, offset),
                textcoords="offset points",
                ha="center",
                color=colour,
                fontsize=8.5,
                fontweight="bold",
            )

    ax.axvline(0.5, color=theme.grid, linewidth=1, zorder=0)
    ax.set_yticks(positions, [matrices["models"][k]["display_name"] for k in rows])
    ax.set_xlim(0, 1.0)
    # Headroom for the value label sitting above the topmost row, which would
    # otherwise run into the title.
    ax.set_ylim(-0.55, len(rows) - 0.25)
    ax.set_xlabel("Rate")
    ax.grid(True, axis="x")
    ax.grid(False, axis="y")
    ax.tick_params(axis="y", length=0)

    ax.scatter([], [], s=110, color=theme.series[1], label="sensitivity (malignant caught)")
    ax.scatter([], [], s=110, color=theme.series[0], label="specificity (benign cleared)")
    legend = ax.legend(
        loc="lower center", bbox_to_anchor=(0.5, -0.42), ncol=2, frameon=False, fontsize=8.5
    )
    for text in legend.get_texts():
        text.set_color(theme.ink_secondary)

    ax.set_title(
        "A wide gap means the model trades one class off against the other",
        color=theme.ink,
        loc="left",
        pad=14,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return _save(fig, f"operating-points{theme.suffix}.png")


def always_benign() -> dict[str, float]:
    """The trivial reference: every test image called benign, via the package's own code.

    Its scores are a constant, which ranks nothing, so its ROC-AUC is 0.5 by definition.
    """
    y_true = np.array([1] * TEST_POSITIVES + [0] * TEST_NEGATIVES)
    scores = operating_point_metrics(y_true, np.zeros_like(y_true))
    scores["roc_auc"] = 0.5
    return scores


def figure_full_benchmark(rows: list[dict[str, str]], theme: Theme) -> Path:
    """Fold mean +/- std per model on three headline metrics, against always-benign.

    Small multiples share the model rows, one metric per panel and one scale per axis.
    The dashed line is what a model that never flags anything scores, so a whisker
    that reaches it is a model this experiment cannot tell apart from doing nothing
    on that metric.
    """
    apply_theme(theme)
    panels = (("roc_auc", "ROC-AUC"), ("balanced_accuracy", "Balanced accuracy"), ("mcc", "MCC"))
    reference = always_benign()
    ordered = list(reversed(rows))  # table order (by year) reads top to bottom
    positions = np.arange(len(ordered))

    fig, axes = plt.subplots(1, len(panels), figsize=(10.5, 4.2), sharey=True)
    for ax, (key, label) in zip(axes, panels, strict=True):
        means = np.array([float(r[key]) for r in ordered])
        stds = np.array([float(r[f"{key}_std"]) for r in ordered])

        ax.axvline(reference[key], color=theme.muted, linewidth=1.2, linestyle=(0, (4, 3)),
                   zorder=1)
        ax.hlines(positions, means - stds, means + stds, color=theme.series[0], linewidth=2,
                  zorder=2)
        ax.scatter(means, positions, s=64, color=theme.series[0], edgecolor=theme.surface,
                   linewidth=2, zorder=3)
        for y, mean, std in zip(positions, means, stds, strict=True):
            ax.annotate(f"{mean:.2f}", xy=(mean + std, y), xytext=(5, 0),
                        textcoords="offset points", va="center", color=theme.ink_secondary,
                        fontsize=8)

        low = min(reference[key], float(np.min(means - stds)))
        high = float(np.max(means + stds))
        pad = 0.08 * (high - low)
        ax.set_xlim(low - pad, high + 2.2 * pad)
        ax.set_title(label, color=theme.ink, loc="left", pad=8)
        ax.grid(True, axis="x")
        ax.grid(False, axis="y")
        ax.tick_params(axis="y", length=0)
        # The reference label gets its own band above the top row, clear of any whisker.
        ax.annotate(f"always benign ({reference[key]:.1f})",
                    xy=(reference[key], len(ordered) - 0.5), xytext=(4, 0),
                    textcoords="offset points", ha="left", va="bottom", color=theme.muted,
                    fontsize=7.5)

    axes[0].set_yticks(positions, [r["display_name"] for r in ordered])
    axes[0].set_ylim(-0.6, len(ordered) + 0.1)
    fig.suptitle(
        "Full benchmark: 15 runs per model (3 seeds x 5 folds), scored on the same 204 test images",
        color=theme.ink, fontsize=12, fontweight="bold", x=0.01, ha="left", y=0.99,
    )
    fig.text(
        0.01, 0.005,
        "Dot: mean over the 15 runs. Whisker: +/- 1 standard deviation across runs. "
        "Dashed line: a model that calls every image benign.",
        color=theme.muted, fontsize=8, ha="left",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    return _save(fig, f"full-benchmark{theme.suffix}.png")


def load_benchmark() -> list[dict[str, str]]:
    with BENCHMARK.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _save(fig, filename: str) -> Path:
    ASSETS.mkdir(parents=True, exist_ok=True)
    path = ASSETS / filename
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path.relative_to(ROOT)}")
    return path


def main() -> int:
    matrices, history = load()
    benchmark = load_benchmark()
    for theme in (LIGHT, DARK):
        figure_training_curves(history, matrices, theme)
        figure_confusion(matrices, theme)
        figure_operating_points(matrices, theme)
        figure_full_benchmark(benchmark, theme)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
