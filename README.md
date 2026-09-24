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

**Full benchmark, measured 2026-09-24.** Seven architectures, 15 runs each (3 seeds ×
stratified 5-fold cross-validation on the 84 training images), every run scored once
on the same 204 held-out test images (162 benign, 42 malignant). `mean ± std` over the
15 runs; the complete table, bootstrap intervals and run record are in
[RESULTS.md](docs/RESULTS.md#full-benchmark).

| Model | Year | ROC-AUC | Balanced acc. | Sensitivity | Specificity | MCC |
|---|---:|---:|---:|---:|---:|---:|
| AlexNet (random init) | 2012 | 0.545 ± 0.092 | 0.525 ± 0.064 | 0.446 ± 0.229 | 0.603 ± 0.278 | 0.057 ± 0.124 |
| GoogLeNet (Inception-v3) | 2014 | 0.827 ± 0.023 | 0.756 ± 0.015 | 0.687 ± 0.071 | 0.826 ± 0.074 | 0.472 ± 0.052 |
| VGG-16 | 2014 | 0.868 ± 0.028 | 0.774 ± 0.042 | 0.752 ± 0.155 | 0.795 ± 0.092 | 0.487 ± 0.047 |
| VGG-19 | 2014 | 0.877 ± 0.030 | 0.767 ± 0.051 | 0.722 ± 0.174 | 0.812 ± 0.089 | 0.487 ± 0.050 |
| ResNet-50 | 2015 | 0.937 ± 0.008 | 0.845 ± 0.029 | 0.800 ± 0.103 | 0.890 ± 0.080 | 0.661 ± 0.068 |
| DenseNet-121 | 2017 | 0.909 ± 0.017 | 0.813 ± 0.038 | 0.802 ± 0.140 | 0.824 ± 0.083 | 0.565 ± 0.040 |
| EfficientNet-B0 | 2019 | 0.882 ± 0.023 | 0.774 ± 0.034 | 0.690 ± 0.146 | 0.857 ± 0.109 | 0.536 ± 0.067 |

For reference, a model that answers "benign" for every test image scores ROC-AUC
0.500, balanced accuracy 0.500, sensitivity 0, specificity 1, MCC 0 — and 79.4%
accuracy.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/figures/full-benchmark-dark.png">
  <img alt="Mean plus or minus one standard deviation over 15 runs for each of seven models on ROC-AUC, balanced accuracy and MCC, with a dashed line at the always-benign value. The six pretrained backbones sit well to the right of the line (ROC-AUC 0.83 to 0.94); AlexNet's whisker crosses it on all three metrics." src="assets/figures/full-benchmark.png">
</picture>

**What the numbers support, and what they do not.** A difference is claimed only when
neither the run-to-run spread nor the bootstrap 95% interval over test images overlaps
([how](docs/RESULTS.md#against-a-model-that-does-nothing)).

**1. Every ImageNet-pretrained backbone outperforms answering "benign" every time;
AlexNet does not.** All six clear the always-benign reference on ROC-AUC, balanced
accuracy and MCC. AlexNet, trained from random initialisation, overlaps it on all
three — on 84 images it is indistinguishable from a model that ignores the image. It is
also the only model without pretraining, so its row measures what ImageNet weights
contribute, not an architectural verdict.

**2. The six pretrained backbones cannot be ranked by this experiment.** ResNet-50 has
the highest mean on ROC-AUC, balanced accuracy, specificity and MCC, but its bootstrap
interval overlaps every other pretrained backbone's on every metric. With 42 malignant
test images, the test set cannot confirm the ordering.

**3. Accuracy would have told a different story, which is why it is not the headline.**
Only ResNet-50's accuracy (0.872 ± 0.047) is separated from the 79.4% of always-benign.
GoogLeNet (Inception-v3) averages 79.7% accuracy — level with doing nothing — while
ranking lesions at ROC-AUC 0.827. Every model is therefore reported on ROC-AUC,
sensitivity, specificity, balanced accuracy and MCC, at an operating point chosen on
validation data and never on test.

**4. Where a rerun was checked, it reproduced bit for bit.** Four architectures
completed in two separate full runs and produced identical results both times. The
full benchmark took 1 h 57 min on one RTX 4060 Ti.

An earlier single-split baseline (three architectures, frozen backbones, argmax
threshold) is kept in [RESULTS.md](docs/RESULTS.md#baseline-run) with its own figures.
It used a different protocol, so its numbers are not comparable to the table above.

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
| [RESULTS.md](docs/RESULTS.md) | Full-benchmark results, intervals and run record; the earlier baseline |
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
