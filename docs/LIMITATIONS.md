# Limitations

What this repository does not establish. Read this before quoting any number from it.

---

## Clinical

**This is not a medical device.** It is a research and teaching benchmark. It has not
been clinically validated, has no regulatory clearance, and must not be used to make
or support a diagnostic decision about any person.

**No skin-tone stratification is possible with this data.** Dermoscopy datasets are
overwhelmingly drawn from lighter-skinned populations, and Fitzpatrick phototype is
not annotated in this source. A model that performs well on the test split here has
demonstrated nothing about how it performs on darker skin — where melanoma is
diagnosed later and outcomes are worse. This is the single most important gap between
these numbers and any clinical claim, and it cannot be closed with this dataset.

**"Benign vs malignant" collapses a much harder problem.** Real dermatological triage
distinguishes many lesion classes, several of which are ambiguous even to specialists
from imaging alone.

## Statistical

**84 training images.** This is the binding constraint on everything. Cross-validation
and repeated seeds make the uncertainty *visible*; they do not make it small. Expect
fold-to-fold standard deviations of several percentage points, and treat any two
models whose intervals overlap as unseparated by this experiment. In the measured run
the run-to-run standard deviation of sensitivity was 7–17 points for the pretrained
backbones, and no two pretrained backbones were separated on any metric
([RESULTS.md](RESULTS.md#reading-the-table)).

**204 test images means coarse resolution.** One image is roughly half a percentage
point of accuracy, and roughly 2.4 points of sensitivity (there are only 42 malignant
cases). Differences smaller than the reported intervals are noise.

**One dataset, one source.** No external validation set. Generalisation to images from
different cameras, clinics or acquisition protocols is untested, and dermoscopy models
are known to be sensitive to exactly that.

**Fine-tuning is not separately ablated by default.** The shipped configs fine-tune 20
top layers after the frozen-head pass. Whether that helps for a given backbone at this
dataset size is an open question the harness can answer (`--set model.fine_tune_layers=0`)
but the default table does not decompose.

## Methodological

**Architecture choice is confounded with pretraining.** AlexNet trains from random
initialisation because Keras distributes no ImageNet weights for it; the other six
start from ImageNet. AlexNet's position in the table therefore reflects "no
pretraining on 84 images," not an architectural verdict.

**Inception-v3 stands in for GoogLeNet.** GoogLeNet (Inception-v1) is not distributed
with Keras. Inception-v3 is a later, stronger member of the same family, so its row
should be read as representing the Inception line rather than the 2014 model
specifically.

**Grad-CAM is a diagnostic, not evidence.** It shows where gradient support is
concentrated. Heat on the lesion is consistent with the model using lesion morphology;
it does not prove it. Heat on a ruler mark, ink, hair or a vignette corner is,
however, strong evidence of the opposite — which is what it is included for.

**No formal significance testing between architectures.** The reported intervals
support "these two overlap" but the harness does not run a paired test across folds.

## Comparison to published work

The reference paper — Surendren & Sumitha, ICDICI 2024
([10.1109/ICDICI62993.2024.10810998](https://doi.org/10.1109/ICDICI62993.2024.10810998))
— reports VGG-16, VGG-19, ResNet and DenseNet at 72.6%, 73.2%, 82.4% and 84.4%
accuracy respectively.

**Those numbers are not directly comparable to this table, and no delta between them
should be computed.** They come from a different split (2,367 train / 660 test) of a
larger collection, at a different class balance, under a training protocol the paper
does not fully specify, with accuracy at an unstated threshold. Any subtraction across
the two sets of numbers would be measuring the difference in protocol, not in model.

They are listed alongside for context. The comparison this repository can actually
support is the one *within* its own table, where the protocol is fixed by construction
and checked in [`tests/test_config.py`](../tests/test_config.py).
