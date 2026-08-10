"""Tests for the architecture registry and the shared head.

The registry tests run everywhere; the graph-building tests need Keras and are marked
accordingly. The preprocessing-pairing test is the one that matters most: a mismatched
normalisation is silent at runtime, so it has to be caught structurally.
"""

from __future__ import annotations

import pytest

from skin_cancer_benchmark import models
from skin_cancer_benchmark.evaluate import benchmark_table, to_markdown


class TestRegistry:
    def test_covers_the_reference_paper_and_the_added_models(self):
        by_origin: dict[str, set[str]] = {}
        for arch in models.ARCHITECTURES.values():
            by_origin.setdefault(arch.origin, set()).add(arch.key)

        assert by_origin["reference-paper"] == {"vgg16", "vgg19", "resnet50", "densenet121"}
        assert by_origin["this-work"] == {"alexnet", "inceptionv3", "efficientnetb0"}

    def test_keys_are_self_consistent(self):
        for key, arch in models.ARCHITECTURES.items():
            assert arch.key == key

    def test_every_pretrained_backbone_declares_a_preprocess_module(self):
        """The pairing that goes silently wrong if it is ever left implicit."""
        for arch in models.ARCHITECTURES.values():
            if arch.pretrained:
                assert arch.preprocess_module, f"{arch.key} has no preprocessing declared"
            else:
                assert arch.preprocess_module is None

    def test_preprocess_modules_are_family_specific(self):
        """Two families must not share one normalisation by accident."""
        assert models.ARCHITECTURES["inceptionv3"].preprocess_module == "inception_v3"
        assert models.ARCHITECTURES["efficientnetb0"].preprocess_module == "efficientnet"
        assert models.ARCHITECTURES["vgg16"].preprocess_module == "vgg16"
        assert models.ARCHITECTURES["resnet50"].preprocess_module == "resnet"

    def test_alexnet_is_the_only_from_scratch_model(self):
        not_pretrained = [a.key for a in models.ARCHITECTURES.values() if not a.pretrained]
        assert not_pretrained == ["alexnet"]

    def test_scratch_preprocess_scales_to_unit_range(self):
        """Resolvable without TensorFlow, since it is plain arithmetic."""
        import numpy as np

        preprocess = models.ARCHITECTURES["alexnet"].preprocess()
        assert preprocess(np.array([0.0, 255.0])).tolist() == [0.0, 1.0]

    def test_positive_index_matches_the_data_layer(self):
        from skin_cancer_benchmark.data import CLASS_NAMES

        assert CLASS_NAMES[models.POSITIVE_INDEX] == "malignant"

    def test_unknown_key_lists_alternatives(self):
        with pytest.raises(KeyError, match="known:"):
            models.get_architecture("mobilenet")


@pytest.mark.tensorflow
class TestBuildModel:
    @pytest.mark.parametrize("key", ["alexnet", "efficientnetb0"])
    def test_output_is_a_two_class_distribution(self, key):
        import numpy as np

        model = models.build_model(key, input_shape=(96, 96, 3))
        probabilities = model.predict(np.zeros((2, 96, 96, 3), dtype="float32"), verbose=0)

        assert probabilities.shape == (2, 2)
        # Softmax, not a 2-unit sigmoid: the rows must be distributions.
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, rtol=1e-5)

    def test_backbone_is_frozen_by_default(self):
        model = models.build_model("efficientnetb0", input_shape=(96, 96, 3))
        trainable = {layer.name for layer in model.layers if layer.trainable}
        assert trainable <= {"head_pool", "head_dropout", "head_dense", "predictions"}

    def test_fine_tuning_keeps_batchnorm_frozen(self):
        from keras import layers

        model = models.build_model(
            "efficientnetb0", input_shape=(96, 96, 3), fine_tune_layers=20
        )
        norms = [ly for ly in model.layers if isinstance(ly, layers.BatchNormalization)]
        assert norms and not any(ly.trainable for ly in norms)


class TestBenchmarkTable:
    def _result(self, key: str, auc: float) -> dict:
        arch = models.ARCHITECTURES[key]
        summary = {
            k: {"mean": auc, "std": 0.02, "min": auc, "max": auc, "n_runs": 5}
            for k in ("roc_auc", "sensitivity", "specificity", "balanced_accuracy",
                      "accuracy", "average_precision", "f1", "mcc")
        }
        return {
            "architecture": key,
            "display_name": arch.display_name,
            "year": arch.year,
            "origin": arch.origin,
            "folds": [{}] * 5,
            "fold_summary": summary,
            "ensemble": {k: auc for k in summary},
        }

    def test_orders_by_year(self):
        table = benchmark_table(
            [self._result("efficientnetb0", 0.9), self._result("alexnet", 0.6)]
        )
        assert table["year"].tolist() == [2012, 2019]

    def test_markdown_reports_mean_and_spread(self):
        rendered = to_markdown(benchmark_table([self._result("densenet121", 0.85)]))
        assert "DenseNet-121" in rendered
        assert "±" in rendered
        assert rendered.count("\n") == 2  # header, divider, one row
