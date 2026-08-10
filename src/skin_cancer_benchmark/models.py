"""Architecture registry and the shared transfer-learning head.

Seven architectures are registered: the four compared by Surendren & Sumitha
(ICDICI 2024) and the three added by this project. They share one classification
head and one training loop, so a difference in score is attributable to the
backbone rather than to incidental differences in how each was wired up.

**The preprocessing contract is the important part of this module.** Each Keras
application was trained under a specific input normalisation -- Inception expects
[-1, 1], the VGG/ResNet family expects Caffe-style mean-subtracted BGR, and the
Keras 3 EfficientNet carries its own rescaling *inside* the graph and therefore
wants raw [0, 255]. Feeding one family's convention to another silently degrades the
features rather than raising: it is a wrong-answer bug, not a crash, and it is
invisible in the training curve beyond an oddly large starting loss.

Sharing a single input pipeline across backbones is the natural way to write this
and the reliable way to get it wrong, because whichever normalisation the pipeline
picks is correct for at most one family. So the preprocessing function travels with
the architecture in the registry, and :func:`~.data.make_dataset` applies whatever
the chosen architecture declares. The pairing cannot drift.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from typing import Any

#: Positive class index for the two-class problem. Malignant is positive because
#: sensitivity to malignancy is the metric that carries clinical cost.
POSITIVE_INDEX = 1

#: The shared classification head. Named so that fine-tuning can tell head from
#: backbone in the assembled model, where the two are one flat list of layers.
HEAD_LAYER_NAMES = frozenset({"head_pool", "head_dropout", "head_dense", "predictions"})


@dataclass(frozen=True)
class Architecture:
    """Everything the pipeline needs to know about one backbone.

    ``preprocess_module`` is the ``keras.applications`` submodule holding the
    matching ``preprocess_input``; resolving it lazily keeps this module importable
    (and the registry testable) without TensorFlow present.
    """

    key: str
    display_name: str
    year: int
    origin: str
    application: str | None
    preprocess_module: str | None
    input_size: tuple[int, int] = (224, 224)
    notes: str = ""

    @property
    def pretrained(self) -> bool:
        return self.application is not None

    def preprocess(self) -> Callable[[Any], Any]:
        """Return the input-normalisation function this backbone was trained under."""
        if self.preprocess_module is None:
            # AlexNet is trained from scratch here (no ImageNet weights ship with
            # Keras), so plain [0, 1] scaling is the honest choice.
            def scale_to_unit(x: Any) -> Any:
                return x / 255.0

            return scale_to_unit

        module = import_module(f"keras.applications.{self.preprocess_module}")
        return module.preprocess_input


ARCHITECTURES: dict[str, Architecture] = {
    "alexnet": Architecture(
        key="alexnet",
        display_name="AlexNet",
        year=2012,
        origin="this-work",
        application=None,
        preprocess_module=None,
        notes="Trained from scratch; no ImageNet weights are distributed with Keras.",
    ),
    "vgg16": Architecture(
        key="vgg16",
        display_name="VGG-16",
        year=2014,
        origin="reference-paper",
        application="VGG16",
        preprocess_module="vgg16",
    ),
    "vgg19": Architecture(
        key="vgg19",
        display_name="VGG-19",
        year=2014,
        origin="reference-paper",
        application="VGG19",
        preprocess_module="vgg19",
    ),
    "inceptionv3": Architecture(
        key="inceptionv3",
        display_name="GoogLeNet (Inception-v3)",
        year=2014,
        origin="this-work",
        application="InceptionV3",
        preprocess_module="inception_v3",
        notes="Stands in for GoogLeNet/Inception-v1, which Keras does not ship.",
    ),
    "resnet50": Architecture(
        key="resnet50",
        display_name="ResNet-50",
        year=2015,
        origin="reference-paper",
        application="ResNet50",
        preprocess_module="resnet",
    ),
    "densenet121": Architecture(
        key="densenet121",
        display_name="DenseNet-121",
        year=2017,
        origin="reference-paper",
        application="DenseNet121",
        preprocess_module="densenet",
    ),
    "efficientnetb0": Architecture(
        key="efficientnetb0",
        display_name="EfficientNet-B0",
        year=2019,
        origin="this-work",
        application="EfficientNetB0",
        preprocess_module="efficientnet",
        notes="Keras 3 normalises inside the graph; preprocess_input is a pass-through.",
    ),
}


def get_architecture(key: str) -> Architecture:
    try:
        return ARCHITECTURES[key]
    except KeyError:
        raise KeyError(
            f"unknown architecture {key!r}; known: {', '.join(sorted(ARCHITECTURES))}"
        ) from None


def build_model(
    architecture: str,
    *,
    input_shape: tuple[int, int, int] = (224, 224, 3),
    dense_units: int = 256,
    dropout: float = 0.3,
    fine_tune_layers: int = 0,
):
    """Build a two-class classifier on the requested backbone.

    The head is ``GlobalAveragePooling -> Dropout -> Dense(relu) -> Dense(2, softmax)``
    for every architecture. Two details in it are load-bearing:

    * The output is **softmax with categorical cross-entropy**, not a 2-unit sigmoid
      with binary cross-entropy. The sigmoid pairing is a common way to spell a
      two-class head and it lets the two scores vary independently: they need not sum
      to 1, so ``predict`` output is not a distribution and thresholding it means
      something different per model.
    * Dropout sits before the dense layer. With 84 training images against a 256-unit
      head, an unregularised head memorises the training split within a few dozen
      epochs while validation stops improving.
    """
    import keras
    from keras import layers

    arch = get_architecture(architecture)

    if not arch.pretrained:
        backbone = _build_alexnet(input_shape)
    else:
        application = getattr(keras.applications, arch.application)
        backbone = application(
            weights="imagenet", include_top=False, input_shape=input_shape
        )
    backbone.trainable = False

    x = layers.GlobalAveragePooling2D(name="head_pool")(backbone.output)
    if dropout > 0:
        x = layers.Dropout(dropout, name="head_dropout")(x)
    x = layers.Dense(dense_units, activation="relu", name="head_dense")(x)
    outputs = layers.Dense(2, activation="softmax", name="predictions")(x)

    model = keras.Model(
        inputs=backbone.input, outputs=outputs, name=f"{arch.key}_classifier"
    )
    if fine_tune_layers > 0:
        unfreeze_top(model, fine_tune_layers)
    return model


def unfreeze_top(model, count: int) -> list[str]:
    """Unfreeze the top ``count`` backbone layers of an assembled classifier.

    Takes the *assembled* model, not a separate backbone object. Building the
    classifier with the functional API on ``backbone.output`` inlines the backbone's
    layers into the new graph rather than nesting it, so after :func:`build_model`
    there is no backbone sub-model left to hand around -- ``model.layers`` is the
    flat list of the backbone's own layers followed by the head's.

    Head layers are excluded by name and stay trainable throughout; BatchNorm is held
    frozen everywhere, because updating its running statistics on a batch of 16 lets
    them drift toward noise and makes fine-tuning look worse than the frozen baseline
    for reasons that have nothing to do with the backbone.

    Returns the names of the layers it unfroze, so a caller can log what moved.
    """
    from keras import layers

    backbone_layers = [layer for layer in model.layers if layer.name not in HEAD_LAYER_NAMES]
    unfrozen: list[str] = []
    for layer in backbone_layers[-count:]:
        if isinstance(layer, layers.BatchNormalization):
            continue
        layer.trainable = True
        unfrozen.append(layer.name)

    for layer in model.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
    return unfrozen


def _build_alexnet(input_shape: tuple[int, int, int]):
    """AlexNet (Krizhevsky et al., 2012), as a feature extractor.

    The 5-conv trunk and 11x11 stride-4 stem follow the paper; local response
    normalisation is replaced by BatchNorm, which is the standard modern
    substitution. AlexNet's own dense classifier is dropped in favour of this
    module's shared head, so it is compared on the same footing as the pretrained
    backbones -- the variable under test is the convolutional trunk, not the size of
    each architecture's original fully-connected stack.
    """
    import keras
    from keras import layers

    inputs = keras.Input(shape=input_shape, name="alexnet_input")
    x = layers.Conv2D(96, 11, strides=4, activation="relu", padding="valid")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(3, strides=2)(x)

    x = layers.Conv2D(256, 5, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(3, strides=2)(x)

    x = layers.Conv2D(384, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(384, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(256, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(3, strides=2)(x)

    return keras.Model(inputs=inputs, outputs=x, name="alexnet_backbone")


def last_conv_layer_name(model) -> str:
    """Name of the deepest 4-D activation, used as the Grad-CAM target.

    Head layers are skipped explicitly: pooling collapses the spatial dimensions, so
    the last 4-D tensor is by definition the top of the backbone, but being explicit
    keeps this correct if the head ever grows a spatial layer of its own.
    """
    for layer in reversed(model.layers):
        if layer.name in HEAD_LAYER_NAMES:
            continue
        shape = getattr(getattr(layer, "output", None), "shape", None)
        if shape is not None and len(shape) == 4:
            return layer.name
    raise ValueError(f"{model.name} has no 4-D activation to attach Grad-CAM to")
