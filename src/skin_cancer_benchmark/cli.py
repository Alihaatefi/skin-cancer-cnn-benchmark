"""Command-line entry point.

``seed_everything`` has to run before TensorFlow builds any op, so the heavy imports
are deferred into each subcommand rather than done at module import.

    skin-benchmark inspect  --config configs/efficientnetb0.yaml
    skin-benchmark train    --config configs/efficientnetb0.yaml
    skin-benchmark benchmark --all --repeats 3
    skin-benchmark report   --results results
    skin-benchmark gradcam  --config configs/efficientnetb0.yaml \\
                            --weights results/efficientnetb0/r0f0/best.weights.h5
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

CONFIG_DIR = Path("configs")


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "handler", None):
        parser.print_help()
        return 1
    try:
        return args.handler(args)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="skin-benchmark",
        description="Reproducible CNN benchmark for skin-lesion classification.",
    )
    sub = parser.add_subparsers(dest="command")

    inspect = sub.add_parser("inspect", help="validate a config and print the split table")
    _add_config_args(inspect)
    inspect.set_defaults(handler=_cmd_inspect)

    train = sub.add_parser("train", help="cross-validate one architecture")
    _add_config_args(train)
    train.set_defaults(handler=_cmd_train)

    bench = sub.add_parser("benchmark", help="run every architecture under one protocol")
    bench.add_argument("--all", action="store_true", help="run every config in configs/")
    bench.add_argument("--configs", nargs="*", type=Path, default=None)
    bench.add_argument("--repeats", type=int, default=None)
    bench.add_argument("--epochs", type=int, default=None)
    bench.add_argument("--folds", type=int, default=None)
    bench.add_argument("--output-dir", type=Path, default=None)
    bench.set_defaults(handler=_cmd_benchmark)

    report = sub.add_parser("report", help="build the comparison table from results/")
    report.add_argument("--results", type=Path, default=Path("results"))
    report.add_argument("--out", type=Path, default=Path("results/benchmark"))
    report.set_defaults(handler=_cmd_report)

    cam = sub.add_parser("gradcam", help="write Grad-CAM overlays for sample images")
    _add_config_args(cam)
    cam.add_argument("--weights", type=Path, required=True)
    cam.add_argument("--images", nargs="*", type=str, default=None)
    cam.add_argument("--limit", type=int, default=8)
    cam.add_argument("--out", type=Path, default=Path("assets/figures/gradcam"))
    cam.set_defaults(handler=_cmd_gradcam)

    return parser


def _add_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="dotted-key override, e.g. --set train.epochs=5 --set seed=3",
    )


def _load(args: argparse.Namespace):
    from .config import load_config

    return load_config(args.config, _parse_overrides(args.overrides))


def _parse_overrides(pairs: Sequence[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"--set expects KEY=VALUE, got {pair!r}")
        key, _, raw = pair.partition("=")
        out[key.strip()] = _coerce_scalar(raw.strip())
    return out


def _coerce_scalar(raw: str) -> Any:
    lowered = raw.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    for cast in (int, float):
        try:
            return cast(raw)
        except ValueError:
            continue
    return raw


def _cmd_inspect(args: argparse.Namespace) -> int:
    from . import data as data_mod
    from . import models as models_mod

    config = _load(args)
    arch = models_mod.get_architecture(config.model.architecture)

    print(f"experiment : {config.name}")
    print(f"backbone   : {arch.display_name} ({arch.year}, {arch.origin})")
    print(f"weights    : {'ImageNet' if arch.pretrained else 'random init'}")
    print(f"protocol   : {config.repeats}x{config.folds}-fold, seed {config.seed}")
    print(f"data root  : {config.data.root}")
    if arch.notes:
        print(f"note       : {arch.notes}")

    train_frame, test_frame = data_mod.load_manifests(config.data)
    print()
    print(data_mod.summarise_splits([("train", train_frame), ("test", test_frame)]))
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    config = _load(args)
    from .seeding import seed_everything

    seed_everything(config.seed, deterministic=config.deterministic)

    from .train import cross_validate

    print(f"[{config.name}] {config.repeats}x{config.folds}-fold cross-validation")
    result = cross_validate(config)
    _print_summary(result)
    return 0


def _cmd_benchmark(args: argparse.Namespace) -> int:
    from .config import load_config
    from .evaluate import benchmark_table, to_markdown
    from .seeding import seed_everything

    paths = list(args.configs or [])
    if args.all or not paths:
        paths = sorted(CONFIG_DIR.glob("*.yaml"))
    if not paths:
        raise FileNotFoundError(f"no configs found in {CONFIG_DIR}")

    overrides: dict[str, Any] = {}
    if args.repeats is not None:
        overrides["repeats"] = args.repeats
    if args.folds is not None:
        overrides["folds"] = args.folds
    if args.epochs is not None:
        overrides["train.epochs"] = args.epochs
    if args.output_dir is not None:
        overrides["output_dir"] = str(args.output_dir)

    results = []
    for path in paths:
        config = load_config(path, overrides)
        seed_everything(config.seed, deterministic=config.deterministic)
        print(f"\n=== {config.name} ===", flush=True)

        from .train import cross_validate

        result = cross_validate(config)
        _print_summary(result)
        results.append(result)

    table = benchmark_table(results)
    print("\n" + to_markdown(table))
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    from .evaluate import benchmark_table, load_results, to_markdown, write_table

    results = load_results(args.results)
    table = benchmark_table(results)
    written = write_table(results, args.out)
    print(to_markdown(table))
    print(f"\nwrote {written} and {written.with_suffix('.csv')}")
    return 0


def _cmd_gradcam(args: argparse.Namespace) -> int:
    config = _load(args)

    from . import data as data_mod
    from . import gradcam as gradcam_mod
    from . import models as models_mod

    arch = models_mod.get_architecture(config.model.architecture)
    model = models_mod.build_model(
        config.model.architecture,
        input_shape=(*config.data.image_size, 3),
        dense_units=config.model.dense_units,
        dropout=config.model.dropout,
    )
    model.load_weights(str(args.weights))

    paths = args.images
    if not paths:
        _, test_frame = data_mod.load_manifests(config.data)
        # Sample across both classes so the overlays are not all of one label.
        per_class = max(1, args.limit // 2)
        paths = (
            test_frame.groupby("label", group_keys=False)
            .head(per_class)["filepath"]
            .tolist()
        )

    written = gradcam_mod.explain_paths(
        model,
        paths,
        preprocess=arch.preprocess(),
        image_size=config.data.image_size,
        output_dir=args.out,
    )
    print(f"wrote {len(written)} overlay(s) to {args.out}")
    return 0


def _print_summary(result: dict[str, Any]) -> None:
    summary = result["fold_summary"]
    ensemble = result["ensemble"]
    print(f"\n  {result['display_name']} - mean over {len(result['folds'])} folds:")
    for key in ("roc_auc", "balanced_accuracy", "sensitivity", "specificity", "accuracy"):
        stats = summary.get(key)
        if stats:
            print(f"    {key:<20} {stats['mean']:.3f} ± {stats['std']:.3f}")
    print(f"    {'ensemble ROC-AUC':<20} {ensemble['roc_auc']:.3f}")
    if "ci95" in ensemble:
        low, high = ensemble["ci95"]["roc_auc"]
        print(f"    {'  95% CI':<20} [{low:.3f}, {high:.3f}]")


if __name__ == "__main__":
    sys.exit(main())
