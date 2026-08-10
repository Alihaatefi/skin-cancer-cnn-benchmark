# Methodology

This document specifies the evaluation protocol. The goal is narrow and worth stating
plainly: **make the benchmark table's differences attributable to the backbone.** Seven
architectures spanning 2012–2019 are compared, and unless everything except the
backbone is held fixed, a gap in the table says nothing about the architectures.

---

## 1. The task

Binary classification of a dermoscopic lesion image as **benign** or **malignant**.

`malignant` is index 1 — the positive class — throughout the codebase. This is pinned
in [`data.CLASS_NAMES`](../src/skin_cancer_benchmark/data.py) rather than inherited
from a loader's alphabetical directory sort, because the published copies of this
dataset do not agree on folder naming (`malignant`/`benign` in some,
`Cancer`/`Non_cancer` in others). An alphabetical sort silently swaps the positive
class between those two copies, which swaps sensitivity and specificity in the
reported table while every line of code stays the same.

## 2. Data

| | Images | Benign | Malignant | Prevalence |
|---|---:|---:|---:|---:|
| Training | 84 | — | — | — |
| Test | 204 | 162 | 42 | 20.6% |

The per-class training counts are printed by `skin-benchmark inspect --config <cfg>`
against your own copy of the data; they are left blank here rather than quoted from
memory, since the class balance of the training directory is the one number that
changes what `class_weight` does.

Source: [Skin Cancer Image Classification](https://www.kaggle.com/datasets/veerendratelaprolu/skin-cancer-image-classification)
(Kaggle, public). Images are not redistributed here; `scripts/download_dataset.py`
fetches them and normalises the directory layout.

Three properties of this dataset drive most of the design below:

**It is very small.** 84 training images is small enough that a single train/validation
split is not a measurement. Removing two images from a 17-image validation split moves
its accuracy by 12 points, so any single-split number is dominated by which images
happened to land where.

**The test split is larger than the training split.** That is unusual and it is a
property of the published data, not a choice made here. It is left as published so
that results remain comparable to other work using the same source, but it means the
*training* side is the binding constraint on what can be learned, while the test side
is comparatively well-powered.

**It is imbalanced, 4:1 toward benign.** This is what makes accuracy an actively
misleading metric here — see §5.

## 3. Splitting

```
Training/ (84) ─── stratified 5-fold ──▶ 5 × (fit 67, validate 17)
                                                    │
                                    early stopping, LR schedule,
                                    checkpoint selection, threshold
                                                    │
Testing/ (204) ─────────────────────────────────────▼ scored once, per fold
```

- **Validation always comes out of the training directory.** Every decision that could
  overfit — when to stop, when to drop the learning rate, which checkpoint to keep,
  where to put the decision threshold — is made against a validation fold.
- **Stratified 5-fold, repeated 3× with different seeds.** 15 models per architecture.
  Every training image validates exactly once per repeat, which converts "one fragile
  number" into a mean with a spread.
- **The test split is never a validation signal.** Passing the test directory as
  Keras' `validation_data` is the standard tutorial shortcut and it is tempting here,
  because 17 images is a painful thing to give up. But it prints a test score every
  epoch, and anything chosen against that score is chosen on test — which turns the
  headline into an optimistic bound rather than a held-out estimate.

Splits are asserted disjoint in [`tests/test_data.py`](../tests/test_data.py), not
assumed.

## 4. Training protocol

Identical for all seven architectures except where an architecture forces a
difference. Defined in [`configs/`](../configs), one file per model.

| | |
|---|---|
| Input | 224 × 224 × 3 |
| Head | `GlobalAvgPool → Dropout(0.3) → Dense(256, ReLU) → Dense(2, softmax)` |
| Backbone | ImageNet weights, frozen; then 20 top layers fine-tuned at 1e-5 |
| Loss | Categorical cross-entropy, label smoothing 0.05 |
| Optimiser | Adam, 1e-3 (head) / 1e-5 (fine-tune) |
| Batch size | 16 |
| Epochs | 60 max, early stopping on `val_auc`, patience 10 |
| Class weights | Inverse frequency, normalised to mean 1.0 |
| Augmentation | Flips, ±15° rotation, ±15% zoom, ±10% translation |

**Preprocessing travels with the architecture.** This is the single most important
implementation detail. Each Keras application was trained under its own input
normalisation — Inception wants [-1, 1], VGG and ResNet want Caffe-style
mean-subtracted BGR, and Keras 3's EfficientNet normalises *inside* the graph and
therefore wants raw [0, 255]. Building one shared input pipeline is the natural way to
write a multi-model benchmark and the reliable way to get it wrong, because whichever
convention that pipeline picks is correct for at most one family. Feeding the wrong
one does not raise; it quietly degrades the features and shows up only as an oddly
large starting loss. So each architecture declares its own `preprocess_input` in the
registry, and the input pipeline applies whatever the selected architecture declares.

**Augmentation is geometric only.** No colour jitter, no hue or saturation shifts.
Colour is diagnostic signal in dermoscopy — asymmetry of pigmentation is part of what
distinguishes a melanoma — so perturbing it trains the model to discount a real
feature.

**BatchNorm stays frozen during fine-tuning.** With a batch size of 16, letting
BatchNorm update its running statistics makes fine-tuning look worse than the frozen
baseline for reasons that have nothing to do with the backbone.

**AlexNet is the one deliberate exception.** Keras ships no ImageNet weights for it, so
it trains from random initialisation with a longer schedule at a lower learning rate.
It is included as the pre-transfer-learning reference point, and its numbers should be
read as "what this dataset supports without pretraining," not as a like-for-like
comparison with the six pretrained backbones.

## 5. Metrics

**Accuracy is reported but is not the headline.** On a 204-image test split of 42
malignant and 162 benign, a model that answers "benign" for every image scores
**79.4%** — higher than a genuinely-trained model can reach — while catching no cancer
at all. Accuracy also moves when prevalence moves, so it cannot compare across
differently-split datasets.

The reported set:

| Metric | Why |
|---|---|
| **ROC-AUC** | Threshold-free ranking quality; the primary headline |
| **Average precision** | Threshold-free, and unlike AUC it is sensitive to the minority class |
| **Sensitivity** | Fraction of malignant lesions caught — the clinically costly error |
| **Specificity** | Fraction of benign lesions correctly cleared |
| **Balanced accuracy** | Mean of the two above; immune to the 4:1 prevalence |
| **MCC** | Single-number summary that degrades honestly on a degenerate confusion matrix |
| **Accuracy** | Reported for comparability with published work only |

**Threshold selection is a fitted parameter, so it is fitted on validation.** The
default operating point maximises Youden's J on the validation fold; that threshold is
then applied unchanged to test. `min_sensitivity` is also available, which fixes the
tolerable miss rate for malignant lesions first and accepts whatever specificity
follows — the screening-oriented framing. Choosing the threshold on test would inflate
every downstream number by the same mechanism as tuning on test.

**Uncertainty is reported, not implied away.** Two numbers accompany every headline:
the fold-to-fold standard deviation across the 15 runs, and a stratified bootstrap 95%
interval over test cases. On 204 test images one percentage point is roughly two
images, and three-decimal precision without an interval overstates what the data can
support. The bootstrap resamples within each class so every replicate keeps the real
42/162 prevalence — an unstratified bootstrap on 42 positives occasionally draws a
replicate with almost none, widening the interval for reasons unrelated to the model.

## 6. Reproducibility

- One seed drives weight init, shuffling, augmentation and the bootstrap.
- `deterministic: true` additionally forces deterministic cuDNN kernels, so a re-run
  reproduces the same figures on a different machine with the same stack. It costs
  roughly 10–25% throughput on GPU.
- Every result file carries its own config, git commit, Python and TensorFlow
  versions, and the GPUs it saw.
- Dependencies are upper-bounded. Keras 3 moved EfficientNet's normalisation inside
  the graph, so the *correct* preprocessing for that family differs between Keras 2 and
  3 — a floating pin would silently change what the code does.

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) to re-run.

## 7. What this protocol does not establish

Held separately in [LIMITATIONS.md](LIMITATIONS.md), and worth reading before quoting
any number from this repository.
