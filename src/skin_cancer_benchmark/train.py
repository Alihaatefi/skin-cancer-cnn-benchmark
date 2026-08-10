"""Training loops: one fold, k folds, and the seven-architecture benchmark.

The single most consequential design decision in this module is that the test split
is *never* touched during training. Validation folds come out of the training
directory; early stopping, learning-rate reduction, checkpoint selection and the
decision threshold are all fitted on those folds. Test predictions are taken once,
after the model is frozen.

The tempting shortcut on a dataset this small is to pass the test directory as
Keras' ``validation_data`` -- it is what most tutorials show, and with 84 training
images it is genuinely painful to give up 17 of them. But every epoch then prints a
test score, and any choice made against that score is a choice made on test, which
turns the reported number into an optimistic bound rather than a held-out estimate.
Cross-validation buys back the lost images instead.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import data as data_mod
from . import metrics as metrics_mod
from . import models as models_mod
from .config import ExperimentConfig
from .seeding import run_manifest, seed_everything


def build_callbacks(config: ExperimentConfig, checkpoint: Path) -> list[Any]:
    """Early stopping, LR reduction and best-weight checkpointing on the monitor.

    The monitor defaults to ``val_auc`` rather than ``val_accuracy``: on a 4:1 split
    accuracy is dominated by the majority class, so it can sit flat while the model's
    actual ranking of malignant lesions improves or degrades underneath it.
    """
    import keras

    mode = "min" if config.train.monitor.endswith("loss") else "max"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    return [
        keras.callbacks.EarlyStopping(
            monitor=config.train.monitor,
            mode=mode,
            patience=config.train.early_stopping_patience,
            restore_best_weights=True,
            verbose=0,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor=config.train.monitor,
            mode=mode,
            factor=0.5,
            patience=config.train.reduce_lr_patience,
            min_lr=1e-7,
            verbose=0,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint),
            monitor=config.train.monitor,
            mode=mode,
            save_best_only=True,
            save_weights_only=True,
            verbose=0,
        ),
    ]


def compile_model(model, *, learning_rate: float, label_smoothing: float) -> None:
    import keras

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing),
        metrics=[
            keras.metrics.CategoricalAccuracy(name="accuracy"),
            keras.metrics.AUC(name="auc", curve="ROC"),
        ],
    )


def train_fold(
    config: ExperimentConfig,
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    *,
    seed: int,
    workdir: Path,
) -> dict[str, Any]:
    """Fit one model on one fold and score it on that fold's validation split.

    Returns the fitted model, its history, the validation scorecard and the decision
    threshold chosen on validation -- the threshold travels with the model so that
    test-time scoring never has to look at test labels to pick one.
    """
    seed_everything(seed, deterministic=config.deterministic)
    arch = models_mod.get_architecture(config.model.architecture)
    preprocess = arch.preprocess()

    train_ds = data_mod.make_dataset(
        train_frame,
        preprocess=preprocess,
        image_size=config.data.image_size,
        batch_size=config.data.batch_size,
        shuffle=True,
        augment=config.train.augment,
        seed=seed,
    )
    val_ds = data_mod.make_dataset(
        val_frame,
        preprocess=preprocess,
        image_size=config.data.image_size,
        batch_size=config.data.batch_size,
    )

    model = models_mod.build_model(
        config.model.architecture,
        input_shape=(*config.data.image_size, 3),
        dense_units=config.model.dense_units,
        dropout=config.model.dropout,
    )
    compile_model(
        model,
        learning_rate=config.train.learning_rate,
        label_smoothing=config.train.label_smoothing,
    )

    weights = data_mod.class_weights(train_frame) if config.train.class_weight else None
    checkpoint = workdir / "best.weights.h5"
    started = time.perf_counter()

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.train.epochs,
        class_weight=weights,
        callbacks=build_callbacks(config, checkpoint),
        verbose=0,
    )

    # Fine-tuning is a second pass at a much lower learning rate, run only after the
    # head has converged -- unfreezing at the head's initial LR destroys the
    # pretrained features in the first few steps.
    if config.model.fine_tune_layers > 0 and arch.pretrained:
        models_mod._unfreeze_top(model.layers[1], config.model.fine_tune_layers)
        compile_model(
            model,
            learning_rate=config.train.fine_tune_learning_rate,
            label_smoothing=config.train.label_smoothing,
        )
        fine_history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=config.train.epochs,
            class_weight=weights,
            callbacks=build_callbacks(config, checkpoint),
            verbose=0,
        )
        for key, values in fine_history.history.items():
            history.history.setdefault(key, []).extend(values)

    elapsed = time.perf_counter() - started

    val_scores = predict_scores(model, val_frame, config, preprocess)
    val_true = data_mod.labels_from(val_frame)
    threshold = metrics_mod.choose_threshold(
        val_true,
        val_scores,
        strategy=config.eval.operating_point,
        min_sensitivity=config.eval.min_sensitivity,
    )

    return {
        "model": model,
        "history": {k: [float(v) for v in vals] for k, vals in history.history.items()},
        "threshold": float(threshold),
        "epochs_run": len(history.history.get("loss", [])),
        "seconds": float(elapsed),
        "validation": metrics_mod.summarise(val_true, val_scores, threshold=threshold),
    }


def predict_scores(model, frame: pd.DataFrame, config: ExperimentConfig, preprocess) -> np.ndarray:
    """Probability of the malignant class, aligned to ``frame``'s row order.

    The dataset is built unshuffled from the same frame, which is the whole reason
    ``labels_from(frame)`` is a valid ground-truth vector for these scores.
    """
    dataset = data_mod.make_dataset(
        frame,
        preprocess=preprocess,
        image_size=config.data.image_size,
        batch_size=config.data.batch_size,
        shuffle=False,
        augment=False,
    )
    probabilities = model.predict(dataset, verbose=0)
    return np.asarray(probabilities)[:, models_mod.POSITIVE_INDEX]


def cross_validate(config: ExperimentConfig) -> dict[str, Any]:
    """Repeated stratified k-fold over the training directory, then score on test.

    Each fold produces a model and a validation-chosen threshold. Test predictions
    are averaged over the folds (a straightforward ensemble) *and* reported per fold,
    so the summary shows both the ensemble number and the fold-to-fold spread. On a
    204-image test set the spread is the more honest headline.
    """
    train_frame, test_frame = data_mod.load_manifests(config.data)
    test_true = data_mod.labels_from(test_frame)
    arch = models_mod.get_architecture(config.model.architecture)
    preprocess = arch.preprocess()

    run_dir = config.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    fold_reports: list[dict[str, Any]] = []
    fold_histories: list[dict[str, list[float]]] = []
    test_score_stack: list[np.ndarray] = []
    thresholds: list[float] = []

    for repeat in range(config.repeats):
        seed = config.seed + repeat * 1000
        folds = data_mod.stratified_folds(train_frame, n_splits=config.folds, seed=seed)
        for fold_index, (fold_train, fold_val) in enumerate(folds):
            tag = f"r{repeat}f{fold_index}"
            print(
                f"  {tag}: fit {len(fold_train)} / validate {len(fold_val)}",
                flush=True,
            )
            outcome = train_fold(
                config, fold_train, fold_val, seed=seed + fold_index, workdir=run_dir / tag
            )

            scores = predict_scores(outcome["model"], test_frame, config, preprocess)
            test_score_stack.append(scores)
            thresholds.append(outcome["threshold"])

            report = metrics_mod.summarise(
                test_true, scores, threshold=outcome["threshold"]
            )
            report.update(
                {
                    "fold": tag,
                    "seed": seed + fold_index,
                    "epochs_run": outcome["epochs_run"],
                    "seconds": outcome["seconds"],
                    "validation": outcome["validation"],
                }
            )
            fold_reports.append(report)
            fold_histories.append(outcome["history"])

            print(
                f"  {tag}: test AUC {report['roc_auc']:.3f} | "
                f"sens {report['sensitivity']:.3f} | spec {report['specificity']:.3f}",
                flush=True,
            )

            _release(outcome["model"])

    ensemble_scores = np.mean(np.vstack(test_score_stack), axis=0)
    ensemble_threshold = float(np.mean(thresholds))
    ensemble = metrics_mod.summarise(
        test_true,
        ensemble_scores,
        threshold=ensemble_threshold,
        bootstrap_samples=config.eval.bootstrap_samples,
        seed=config.seed,
    )

    result = {
        "name": config.name,
        "architecture": config.model.architecture,
        "display_name": arch.display_name,
        "year": arch.year,
        "origin": arch.origin,
        "config": config.to_dict(),
        "manifest": run_manifest(seed=config.seed),
        "splits": {
            "train": data_mod.describe(train_frame),
            "test": data_mod.describe(test_frame),
        },
        "folds": fold_reports,
        "fold_summary": metrics_mod.aggregate(fold_reports),
        "ensemble": ensemble,
        "histories": fold_histories,
    }

    output = run_dir / "results.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    np.save(run_dir / "test_scores.npy", ensemble_scores)
    print(f"  wrote {output}", flush=True)
    return result


def _release(model) -> None:
    """Drop a fold's model and clear the Keras graph.

    Without this, k folds of a 22M-parameter backbone accumulate in one session and
    the later folds OOM on a laptop GPU -- which looks like a fold-dependent failure
    and is really just a leak.
    """
    import gc

    import keras

    del model
    keras.backend.clear_session()
    gc.collect()
