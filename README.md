# Skin Cancer CNN Benchmark

**A single-protocol comparison of seven convolutional architectures for
malignant/benign dermoscopic lesion classification — built so that a difference in the
results table is attributable to the model rather than to the harness.**

[![CI](https://github.com/Alihaatefi/skin-cancer-cnn-benchmark/actions/workflows/ci.yml/badge.svg)](https://github.com/Alihaatefi/skin-cancer-cnn-benchmark/actions/workflows/ci.yml)
[![Python 3.10–3.12](https://img.shields.io/badge/python-3.10--3.12-blue.svg)](https://www.python.org/)
[![TensorFlow 2.16+](https://img.shields.io/badge/tensorflow-2.16%2B-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## What this is

Published comparisons of CNN architectures on small medical-imaging datasets are hard
to read, because the numbers usually come from different splits, different
preprocessing and different unstated training details. The differences that get
attributed to architecture are often differences in protocol.

This repository is the other thing: **one harness, one protocol, seven backbones.**
AlexNet (2012) through EfficientNet-B0 (2019) — including the four architectures
compared by [Surendren & Sumitha, ICDICI 2024](https://doi.org/10.1109/ICDICI62993.2024.10810998) —
are trained under identical folds, seeds, augmentation, class weighting and stopping
rules, then scored with threshold-free metrics and bootstrap confidence intervals.
That the protocol is genuinely shared is enforced by a test, not by a promise.

It is a research and teaching benchmark. **It is not a medical device** — see
[LIMITATIONS.md](docs/LIMITATIONS.md).

## Results

A single-split baseline across three architectures has been measured. Full protocol
and figures in [RESULTS.md](docs/RESULTS.md).

**84 training / 204 test images (162 benign, 42 malignant).** Pretrained backbones
frozen, Adam, batch 16, 30 epochs; AlexNet from random init for 100.

| Model | Year | Accuracy | Balanced acc. | Sensitivity | Specificity | MCC |
|---|---:|---:|---:|---:|---:|---:|
| AlexNet | 2012 | 58.3% | 0.561 | 0.524 | 0.599 | 0.100 |
| GoogLeNet (Inception-v3) | 2014 | 65.7% | 0.775 | **0.976** | 0.574 | 0.446 |
| **EfficientNet-B0** | 2019 | **82.4%** | **0.792** | 0.738 | **0.846** | **0.529** |

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/figures/operating-points-dark.png">
  <img alt="Sensitivity and specificity per model, shown as paired dots. GoogLeNet reaches 0.98 sensitivity but only 0.57 specificity; EfficientNet-B0 is balanced at 0.74 and 0.85; AlexNet sits near 0.52 and 0.60." src="assets/figures/operating-points.png">
</picture>

**Three findings, and the third is the one that shaped the design of this repository:**

**1. Pretraining, not architecture, is what the dataset can support.** AlexNet reaches
MCC 0.10 against a chance baseline of 0.0 — barely better than guessing. Its row marks
how much of every other row comes from ImageNet weights rather than from architectural
merit.

**2. EfficientNet-B0 is the best-balanced model, and the cheapest.** Highest balanced
accuracy and MCC, and the fastest to train of the three despite being the newest.

**3. Accuracy ranks these models wrong, so the benchmark does not use it as the
headline.** A model that answers "benign" for all 204 test images scores **79.4%** —
beating two of the three rows above while catching zero cancers. And the two leading
models fail in opposite directions: GoogLeNet catches 41 of 42 malignant lesions but
flags 69 healthy ones, while EfficientNet clears benign cases far better and misses
11 malignancies. Which error profile is preferable is a deployment question, not an
accuracy question — so every model here is reported on sensitivity, specificity,
balanced accuracy, MCC and ROC-AUC, at an operating point chosen on validation data
and never on test.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/figures/confusion-matrices-dark.png">
  <img alt="Confusion matrices for the three models on 204 test images. AlexNet 97/65/20/22, GoogLeNet 93/69/1/41, EfficientNet-B0 137/25/11/31." src="assets/figures/confusion-matrices.png">
</picture>

The full seven-architecture benchmark is implemented and tested; running it needs the
dataset locally and 3–5 GPU-hours. See [RESULTS.md](docs/RESULTS.md#full-benchmark).

## Method

Full specification in [METHODOLOGY.md](docs/METHODOLOGY.md). The decisions that matter:

**Preprocessing travels with the architecture.** Every Keras application was trained
under its own input normalisation — Inception wants [-1, 1], VGG and ResNet want
Caffe-style mean-subtracted BGR, and Keras 3's EfficientNet normalises *inside* the
graph and so wants raw [0, 255]. One shared input pipeline is the natural way to write
a multi-model benchmark and the reliable way to get it wrong, because whichever
convention it picks is correct for at most one family. Getting it wrong does not
raise — it quietly degrades the features. So each architecture declares its own
`preprocess_input` in the registry and the pipeline applies whatever the chosen model
declares.

**The test split is never a validation signal.** Validation folds come out of the
training directory. Early stopping, learning-rate schedule, checkpoint selection and
the decision threshold are all fitted there; test is scored once, after the model is
frozen.

**Uncertainty is reported.** Stratified 5-fold × 3 seeds = 15 runs per architecture,
reported as mean ± std, plus a stratified bootstrap 95% interval over test cases. With
84 training images, changing the seed moves the result by more than most of the
architectural differences being measured — reporting a single run would be reporting
noise.

**Augmentation is geometric only.** No colour jitter: pigmentation asymmetry is
diagnostic signal in dermoscopy, and perturbing it trains the model to discount a real
feature.

## Quickstart

```bash
pip install -e ".[dev,train]"
python scripts/download_dataset.py          # needs Kaggle API credentials

skin-benchmark inspect --config configs/efficientnetb0.yaml   # verify the splits
skin-benchmark train   --config configs/efficientnetb0.yaml   # 3 × 5-fold
skin-benchmark benchmark --all                                # all seven
skin-benchmark report                                         # comparison table
```

Any config value is overridable, so a smoke test or a sweep needs no new files:

```bash
skin-benchmark train --config configs/densenet121.yaml \
  --set train.epochs=2 --set folds=2 --set repeats=1
```

Grad-CAM overlays, to check *where* a model is looking:

```bash
skin-benchmark gradcam --config configs/efficientnetb0.yaml \
  --weights results/efficientnetb0/r0f0/best.weights.h5
```

Full setup, GPU notes and runtime costs: [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## Layout

```
src/skin_cancer_benchmark/
├── config.py      typed YAML experiments, validated before anything trains
├── data.py        manifests, canonical labels, stratified splits, tf.data inputs
├── models.py      architecture registry + shared head; preprocessing contract
├── metrics.py     threshold-free + operating-point scoring, bootstrap intervals
├── train.py       cross-validated training; test stays untouched
├── evaluate.py    per-architecture results → the comparison table
├── gradcam.py     saliency overlays
└── seeding.py     seed control, determinism, run provenance

configs/           one YAML per architecture; only `model` differs between them
tests/             85 tests, 78 of which need no TensorFlow
scripts/           dataset download, figure rendering
docs/              methodology, results, reproducibility, limitations
```

**Tests.** 85 in total. The layers carrying the claims most likely to be silently
wrong — label mapping, split disjointness, threshold selection, metric arithmetic —
are pure NumPy/pandas and run everywhere in about six seconds; the seven Keras-graph
tests build every registered backbone for real. Among them:

- `malignant` is index 1 regardless of how the source folders are spelled — otherwise
  sensitivity and specificity silently swap between two copies of the same dataset
- no image appears on both sides of any split, across all folds
- every architecture declares its own preprocessing, and no two families share one
- the backbone is inlined rather than nested by the functional API, which is the
  assumption every fine-tuning path depends on
- fine-tuning moves only the top of the backbone, never the bottom, and never
  BatchNorm
- all seven configs are identical outside their `model` section — the benchmark's
  central claim, asserted rather than promised

## Documentation

| | |
|---|---|
| [METHODOLOGY.md](docs/METHODOLOGY.md) | Task, splits, training protocol, metric choices |
| [RESULTS.md](docs/RESULTS.md) | Measured results, figures, full-benchmark spec |
| [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) | Environment, data, runtime cost, determinism |
| [LIMITATIONS.md](docs/LIMITATIONS.md) | **What this does not establish** |

## Limitations

The short version, in full in [LIMITATIONS.md](docs/LIMITATIONS.md):

- **Not a medical device.** No clinical validation, no regulatory clearance.
- **No skin-tone stratification is possible.** Fitzpatrick phototype is not annotated
  in this dataset, and dermoscopy collections skew heavily toward lighter skin. Nothing
  here says anything about performance on darker skin — the largest gap between these
  numbers and a clinical claim, and one this dataset cannot close.
- **84 training images.** Cross-validation makes the uncertainty visible; it does not
  make it small. Treat overlapping intervals as unseparated.
- **One dataset, no external validation.** Generalisation across cameras and clinics is
  untested.
- **The reference paper's accuracies are not comparable to this table** and no delta
  between them should be computed — different split, different dataset size, unstated
  protocol.

## Citation

```bibtex
@software{hatefi_skin_cancer_benchmark,
  author  = {Hatefi, Ali},
  title   = {Skin Cancer CNN Benchmark: a reproducible comparison of seven
             convolutional architectures for dermoscopic lesion classification},
  year    = {2025},
  version = {1.0.0},
  url     = {https://github.com/Alihaatefi/skin-cancer-cnn-benchmark}
}
```

Reference paper: Surendren D. and Sumitha J., "Skin Cancer Detection using
Convolutional Neural Network Models: VGG-16, VGG-19, ResNet and DenseNet,"
*2024 5th International Conference on Data Intelligence and Cognitive Informatics
(ICDICI)*, pp. 1254–1258. [doi:10.1109/ICDICI62993.2024.10810998](https://doi.org/10.1109/ICDICI62993.2024.10810998)

Dataset: [Skin Cancer Image Classification](https://www.kaggle.com/datasets/veerendratelaprolu/skin-cancer-image-classification)
(Kaggle). Images are not redistributed here.

## Author

**Ali Hatefi** — M.Sc. student, Amirkabir University of Technology
· [alihatefi@aut.ac.ir](mailto:alihatefi@aut.ac.ir)
· [github.com/Alihaatefi](https://github.com/Alihaatefi)

Licensed under the [MIT License](LICENSE).
