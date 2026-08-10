"""Aggregation of per-architecture results into the benchmark table.

Nothing here trains. It reads the ``results.json`` files written by
:mod:`.train` and produces the comparison table and the figures the README shows,
which keeps "how a model scored" separate from "how the models compare".
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import metrics as metrics_mod

#: Column order for the published benchmark table.
TABLE_COLUMNS = (
    "display_name",
    "year",
    "origin",
    "roc_auc",
    "average_precision",
    "balanced_accuracy",
    "sensitivity",
    "specificity",
    "accuracy",
    "mcc",
)


def load_results(run_dir: str | Path) -> list[dict[str, Any]]:
    """Load every ``results.json`` under ``run_dir``, newest-first by architecture."""
    run_dir = Path(run_dir)
    found = sorted(run_dir.glob("*/results.json"))
    if not found:
        raise FileNotFoundError(
            f"no results.json under {run_dir}. Run `skin-benchmark benchmark` first."
        )
    return [json.loads(path.read_text(encoding="utf-8")) for path in found]


def benchmark_table(results: Iterable[dict[str, Any]]) -> pd.DataFrame:
    """One row per architecture: fold mean +/- std, plus the ensemble score.

    The mean-of-folds column is the one to read. The ensemble column is reported
    because it is what a deployed model would be, but it is a single number from a
    single test set and carries no spread of its own.
    """
    rows = []
    for result in results:
        summary = result["fold_summary"]
        row: dict[str, Any] = {
            "architecture": result["architecture"],
            "display_name": result["display_name"],
            "year": result["year"],
            "origin": result["origin"],
            "n_folds": len(result["folds"]),
        }
        for key in metrics_mod.SUMMARY_KEYS:
            stats = summary.get(key)
            row[key] = stats["mean"] if stats else float("nan")
            row[f"{key}_std"] = stats["std"] if stats else float("nan")
            row[f"{key}_ensemble"] = float(result["ensemble"].get(key, float("nan")))
        rows.append(row)

    frame = pd.DataFrame(rows)
    return frame.sort_values(["year", "display_name"], ignore_index=True)


def to_markdown(table: pd.DataFrame, *, with_std: bool = True) -> str:
    """Render the benchmark table as GitHub-flavoured Markdown.

    Values are shown to one decimal place with their spread. Three-decimal accuracy
    on 204 test images would imply a resolution the data does not have -- one image
    is 0.5 percentage points.
    """
    header = (
        "| Model | Year | Source | ROC-AUC | AP | Balanced acc. | Sensitivity | "
        "Specificity | Accuracy | MCC |"
    )
    divider = "|" + "|".join(["---"] * 10) + "|"
    lines = [header, divider]

    origin_label = {"reference-paper": "paper", "this-work": "added"}
    for _, row in table.iterrows():
        cells = [
            f"**{row['display_name']}**",
            str(int(row["year"])),
            origin_label.get(row["origin"], row["origin"]),
        ]
        for key in ("roc_auc", "average_precision", "balanced_accuracy", "sensitivity",
                    "specificity", "accuracy", "mcc"):
            cells.append(_cell(row[key], row.get(f"{key}_std"), with_std=with_std))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _cell(mean: float, std: float | None, *, with_std: bool) -> str:
    if not np.isfinite(mean):
        return "-"
    if with_std and std is not None and np.isfinite(std):
        return f"{mean:.3f} ± {std:.3f}"
    return f"{mean:.3f}"


def write_table(results: Iterable[dict[str, Any]], destination: str | Path) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    table = benchmark_table(results)
    table.to_csv(destination.with_suffix(".csv"), index=False)
    destination.with_suffix(".md").write_text(to_markdown(table), encoding="utf-8")
    return destination.with_suffix(".md")


def compare_to_reference(table: pd.DataFrame, reference: dict[str, float]) -> pd.DataFrame:
    """Attach the reference paper's published accuracies alongside our own.

    The delta column is intentionally *not* computed. The published numbers come
    from a different split of a larger dataset (2,367 train / 660 test against our
    84 / 204) under an unstated protocol, so subtracting them would manufacture a
    comparison the two sets of numbers cannot support. They are shown side by side
    as context, and the like-for-like comparison is the one *within* this table.
    """
    frame = table.copy()
    frame["reported_accuracy_paper"] = frame["architecture"].map(reference)
    return frame
