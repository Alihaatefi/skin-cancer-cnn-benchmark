# Reproducibility

Everything needed to re-run the benchmark end to end.

---

## 1. Environment

```bash
git clone https://github.com/Alihaatefi/skin-cancer-cnn-benchmark.git
cd skin-cancer-cnn-benchmark

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -e ".[dev,train]"
```

Python 3.10–3.12. TensorFlow 2.16–2.20 (Keras 3).

**Without a GPU**, everything still installs and the full test suite runs; a full
benchmark on CPU is slow but tractable, since the backbones are frozen for most of
each run. **With a GPU**, install the appropriate TensorFlow build for your CUDA
version first, then `pip install -e ".[dev]"` to avoid overwriting it.

Verify:

```bash
pytest -q                      # 85 tests; graph tests skip without TensorFlow
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

## 2. Data

```bash
pip install kagglehub
python scripts/download_dataset.py
```

Needs Kaggle API credentials — `~/.kaggle/kaggle.json`, or `KAGGLE_USERNAME` and
`KAGGLE_KEY` in the environment. The script fetches
[veerendratelaprolu/skin-cancer-image-classification](https://www.kaggle.com/datasets/veerendratelaprolu/skin-cancer-image-classification),
normalises whatever directory spelling that copy uses, verifies both splits load, and
checks that no file appears in both.

Already have a copy? Point the script at it:

```bash
python scripts/download_dataset.py --from-dir /path/to/existing/copy
```

Expected layout:

```
data/New_Skin_Data_1/
├── Training/{benign,malignant}/*.jpg
└── Testing/{benign,malignant}/*.jpg
```

Confirm before spending GPU time:

```bash
skin-benchmark inspect --config configs/efficientnetb0.yaml
```

## 3. Run

```bash
# One architecture, 3 × 5-fold
skin-benchmark train --config configs/efficientnetb0.yaml

# All seven, one protocol
skin-benchmark benchmark --all

# Collapse to the comparison table
skin-benchmark report
```

Cheap smoke test before committing to the full run:

```bash
skin-benchmark train --config configs/efficientnetb0.yaml \
  --set train.epochs=2 --set folds=2 --set repeats=1 \
  --set eval.bootstrap_samples=0
```

Any config value is overridable with `--set key.path=value`, so a sweep needs no new
files:

```bash
for seed in 1 2 3; do
  skin-benchmark train --config configs/densenet121.yaml \
    --set seed=$seed --set name=densenet121_seed$seed
done
```

## 4. Cost

Measured on one laptop RTX 3050 Ti (4 GB), frozen backbones, batch 16.

| | Wall clock |
|---|---|
| One fold | 1.5–3 min |
| One architecture (3 × 5 folds) | 25–45 min |
| Full benchmark (7 architectures) | 3–5 h |

VGG-16 and VGG-19 dominate; both are far heavier than the rest for the accuracy they
deliver. Drop `--set model.fine_tune_layers=0` to skip the fine-tuning pass and roughly
halve the total.

If a fold OOMs, lower `data.batch_size` — the harness clears the Keras session between
folds, so memory pressure comes from a single model, not from accumulation.

## 5. Determinism

`deterministic: true` (the default in every shipped config) sets `TF_DETERMINISTIC_OPS`,
`TF_CUDNN_DETERMINISTIC` and `PYTHONHASHSEED`, and calls
`tf.config.experimental.enable_op_determinism()` before any op is built — which is why
`seed_everything` runs at CLI entry, before the training modules are imported.

With it on, the same seed and the same software stack reproduce the same figures on
different machines, at roughly 10–25% lower GPU throughput. With it off, a run is
reproducible on the machine that produced it but not necessarily elsewhere.

Every `results.json` records the git commit, Python and TensorFlow versions, platform
string and visible GPUs, so a number can always be traced back to the stack that
produced it.

**A caveat worth stating.** Determinism makes a number repeatable; it does not make it
stable. With 84 training images, changing the seed moves the result by more than most
of the architectural differences being measured — which is the entire reason results
are reported as a mean over 15 runs with a spread rather than as a best run.

## 6. Figures

```bash
python scripts/render_figures.py
```

Regenerates every README figure, in light and dark variants, from the committed JSON
in `results/baseline_run/` and `assets/baseline_run/`. Charts are generated rather
than pasted so they cannot drift from the numbers behind them; CI rebuilds them on
every push.
