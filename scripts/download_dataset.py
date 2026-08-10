"""Fetch the dermoscopy dataset and lay it out the way the package expects.

The images are not vendored into this repository -- they belong to their original
publisher and are redistributed under that publisher's terms, not this project's
licence. This script pulls them from Kaggle and normalises whatever directory
spelling that copy happens to use into the canonical layout::

    data/New_Skin_Data_1/
        Training/{benign,malignant}/*.jpg
        Testing/{benign,malignant}/*.jpg

Requires Kaggle API credentials (``~/.kaggle/kaggle.json``, or the ``KAGGLE_USERNAME``
and ``KAGGLE_KEY`` environment variables). See docs/REPRODUCIBILITY.md.

    python scripts/download_dataset.py
    python scripts/download_dataset.py --dest data/New_Skin_Data_1 --force
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from skin_cancer_benchmark.data import (  # noqa: E402
    IMAGE_SUFFIXES,
    LABEL_ALIASES,
    DatasetError,
    describe,
    scan_split,
)

DATASET = "veerendratelaprolu/skin-cancer-image-classification"
DEFAULT_DEST = Path("data/New_Skin_Data_1")

#: Directory names that mean "training" / "test" across published copies.
SPLIT_ALIASES = {
    "train": "Training",
    "training": "Training",
    "test": "Testing",
    "testing": "Testing",
    "val": "Testing",
    "validation": "Testing",
}


def download(dest: Path, *, force: bool) -> Path:
    if dest.exists() and not force:
        print(f"{dest} already exists; pass --force to re-download.")
        return dest

    try:
        import kagglehub
    except ImportError:
        raise SystemExit(
            "kagglehub is not installed. Either:\n"
            "  pip install kagglehub\n"
            "or download the dataset manually from\n"
            f"  https://www.kaggle.com/datasets/{DATASET}\n"
            f"and unpack it into {dest} as Training/ and Testing/ class folders."
        ) from None

    print(f"downloading {DATASET} ...")
    cached = Path(kagglehub.dataset_download(DATASET))
    print(f"  cached at {cached}")
    return organise(cached, dest, force=force)


def organise(source: Path, dest: Path, *, force: bool) -> Path:
    """Copy the cached download into the canonical split/class layout."""
    if dest.exists() and force:
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)

    copied = 0
    for split_dir in _find_split_dirs(source):
        canonical_split = SPLIT_ALIASES[split_dir.name.strip().lower()]
        for class_dir in (p for p in split_dir.iterdir() if p.is_dir()):
            canonical_class = LABEL_ALIASES.get(class_dir.name.strip().lower())
            if canonical_class is None:
                print(f"  skipping unrecognised class folder {class_dir}")
                continue
            target = dest / canonical_split / canonical_class
            target.mkdir(parents=True, exist_ok=True)
            for image in class_dir.iterdir():
                if image.suffix.lower() in IMAGE_SUFFIXES:
                    shutil.copy2(image, target / image.name)
                    copied += 1

    if copied == 0:
        raise SystemExit(
            f"no images copied from {source}.\n"
            "The download's directory layout is not one this script recognises; "
            "arrange it manually as Training/ and Testing/ class folders."
        )
    print(f"  organised {copied} image(s) into {dest}")
    return dest


def _find_split_dirs(root: Path) -> list[Path]:
    """Locate the train/test directories, wherever the archive nested them."""
    found = [
        path
        for path in root.rglob("*")
        if path.is_dir() and path.name.strip().lower() in SPLIT_ALIASES
    ]
    if not found:
        raise SystemExit(
            f"no train/test directories under {root}. "
            f"Looked for any of: {sorted(set(SPLIT_ALIASES))}"
        )
    return found


def verify(dest: Path) -> int:
    """Confirm the result is loadable by the same code the training run uses."""
    try:
        train = scan_split(dest / "Training")
        test = scan_split(dest / "Testing")
    except DatasetError as exc:
        print(f"verification failed: {exc}", file=sys.stderr)
        return 1

    for name, frame in (("Training", train), ("Testing", test)):
        info = describe(frame)
        print(
            f"  {name:<9} {info['n']:>5} images  "
            f"({info['counts']['benign']} benign / {info['counts']['malignant']} malignant, "
            f"{info['prevalence_malignant']:.1%} malignant)"
        )

    overlap = set(train["filepath"]) & set(test["filepath"])
    if overlap:
        print(f"verification failed: {len(overlap)} file(s) in both splits", file=sys.stderr)
        return 1

    print("dataset ready.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument("--from-dir", type=Path, help="organise an already-downloaded copy")
    parser.add_argument("--force", action="store_true", help="re-download and overwrite --dest")
    args = parser.parse_args(argv)

    dest = organise(args.from_dir, args.dest, force=args.force) if args.from_dir else download(
        args.dest, force=args.force
    )
    return verify(dest)


if __name__ == "__main__":
    sys.exit(main())
