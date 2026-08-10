"""Grad-CAM overlays, for checking *why* a model scored well.

On a dataset this small, a high AUC is not by itself evidence that the model learned
lesion morphology. Dermoscopy images frequently contain rulers, ink marks, hair and
vignetting, and a classifier can reach a respectable score by keying on an artefact
that correlates with how the malignant images happened to be acquired.

Grad-CAM (Selvaraju et al., 2017) does not prove the model is right, but it makes
that failure mode visible: heat that sits on the lesion is consistent with the
intended behaviour, heat that sits on a corner marker is not.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from . import models as models_mod


def compute_heatmap(
    model,
    image_batch,
    *,
    layer_name: str | None = None,
    class_index: int | None = None,
):
    """Grad-CAM heatmap for one preprocessed image, normalised to [0, 1].

    ``image_batch`` must already carry the backbone's own preprocessing, since the
    gradient is taken through the network as it is actually run.
    """
    import keras
    import tensorflow as tf

    layer_name = layer_name or models_mod.last_conv_layer_name(model)
    grad_model = keras.Model(
        model.inputs, [model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        activations, predictions = grad_model(image_batch, training=False)
        if class_index is None:
            class_index = int(tf.argmax(predictions[0]))
        score = predictions[:, class_index]

    gradients = tape.gradient(score, activations)
    # Channel importance = spatially averaged gradient; the CAM is that weighted sum
    # of feature maps, rectified so only evidence *for* the class is shown.
    weights = tf.reduce_mean(gradients, axis=(0, 1, 2))
    cam = tf.reduce_sum(activations[0] * weights, axis=-1)
    cam = tf.nn.relu(cam)

    confidence = float(predictions[0, class_index])
    peak = tf.reduce_max(cam)
    if float(peak) <= 0:
        # A flat map means no positive evidence at this layer; return zeros rather
        # than dividing by zero and rendering noise as if it were signal.
        return np.zeros(cam.shape, dtype=np.float32), int(class_index), confidence
    return (cam / peak).numpy(), int(class_index), confidence


def overlay(image_rgb: np.ndarray, heatmap: np.ndarray, *, alpha: float = 0.45) -> np.ndarray:
    """Blend a heatmap over the original RGB image, returned as uint8."""
    import matplotlib.cm as cm
    from PIL import Image

    resized = np.asarray(
        Image.fromarray((heatmap * 255).astype(np.uint8)).resize(
            (image_rgb.shape[1], image_rgb.shape[0]), Image.BILINEAR
        )
    ) / 255.0

    coloured = cm.get_cmap("inferno")(resized)[..., :3]
    blended = (1 - alpha) * (image_rgb / 255.0) + alpha * coloured
    return (np.clip(blended, 0, 1) * 255).astype(np.uint8)


def explain_paths(
    model,
    paths: Sequence[str],
    *,
    preprocess,
    image_size: tuple[int, int],
    output_dir: str | Path,
) -> list[Path]:
    """Write one Grad-CAM overlay per input path. Returns the files written."""
    import tensorflow as tf
    from PIL import Image

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for path in paths:
        raw = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        resized = tf.image.resize(raw, image_size, method="bilinear")
        rgb = np.asarray(tf.cast(resized, tf.uint8))

        batch = preprocess(tf.cast(resized, tf.float32)[None, ...])
        heatmap, class_index, confidence = compute_heatmap(model, batch)

        label = ("benign", "malignant")[class_index]
        destination = output_dir / f"{Path(path).stem}_{label}_{confidence:.2f}.png"
        Image.fromarray(overlay(rgb, heatmap)).save(destination)
        written.append(destination)

    return written
