"""Tests for the scoring layer.

The behaviour worth pinning here is not "does sklearn compute AUC" -- it is that this
project's specific choices hold: that accuracy is exposed as the misleading number it
is on a 4:1 split, that thresholds are chosen without seeing the labels they will be
scored against, and that degenerate folds return a defined value instead of raising.
"""

from __future__ import annotations

import numpy as np
import pytest

from skin_cancer_benchmark import metrics


class TestConfusionCounts:
    def test_counts_each_cell(self):
        y_true = [1, 1, 0, 0, 1, 0]
        y_pred = [1, 0, 0, 1, 1, 0]
        assert metrics.confusion_counts(y_true, y_pred) == {"tn": 2, "fp": 1, "fn": 1, "tp": 2}

    def test_single_class_prediction_does_not_raise(self):
        """A model that predicts one class is a real outcome on a small fold."""
        counts = metrics.confusion_counts([1, 0, 0, 0], [0, 0, 0, 0])
        assert counts == {"tn": 3, "fp": 0, "fn": 1, "tp": 0}

    def test_shape_mismatch_is_rejected(self):
        with pytest.raises(ValueError, match="shape mismatch"):
            metrics.confusion_counts([1, 0, 1], [1, 0])


class TestOperatingPointMetrics:
    def test_majority_class_predictor_scores_high_accuracy_and_zero_sensitivity(self):
        """The headline failure mode this module exists to make visible.

        Predicting 'benign' for all 204 test images yields 79.4% accuracy while
        catching no cancer at all, so accuracy alone cannot rank these models.
        """
        y_true = np.concatenate([np.ones(42, dtype=int), np.zeros(162, dtype=int)])
        y_pred = np.zeros(204, dtype=int)

        report = metrics.operating_point_metrics(y_true, y_pred)

        assert report["accuracy"] == pytest.approx(162 / 204, abs=1e-9)
        assert report["accuracy"] > 0.79
        assert report["sensitivity"] == 0.0
        assert report["balanced_accuracy"] == pytest.approx(0.5)
        assert report["mcc"] == 0.0

    def test_perfect_prediction(self):
        y_true = [1, 1, 0, 0]
        report = metrics.operating_point_metrics(y_true, y_true)
        assert report["sensitivity"] == 1.0
        assert report["specificity"] == 1.0
        assert report["mcc"] == pytest.approx(1.0)

    def test_zero_denominators_return_zero_not_nan(self):
        report = metrics.operating_point_metrics([0, 0], [0, 0])
        assert report["sensitivity"] == 0.0
        assert report["precision"] == 0.0
        assert report["f1"] == 0.0
        assert np.isfinite(report["mcc"])


class TestThresholdFreeMetrics:
    def test_auc_of_informative_scores(self, separable_scores):
        y_true, y_score = separable_scores
        report = metrics.threshold_free_metrics(y_true, y_score)
        assert 0.8 < report["roc_auc"] < 1.0
        assert report["average_precision"] > 0.5

    def test_single_class_truth_is_undefined_not_an_exception(self):
        report = metrics.threshold_free_metrics([1, 1, 1], [0.2, 0.7, 0.9])
        assert np.isnan(report["roc_auc"])
        assert np.isnan(report["average_precision"])


class TestThresholdSelection:
    def test_youden_beats_the_default_half_on_imbalanced_scores(self, separable_scores):
        y_true, y_score = separable_scores

        chosen = metrics.youden_threshold(y_true, y_score)
        at_youden = metrics.operating_point_metrics(y_true, (y_score >= chosen).astype(int))
        at_half = metrics.operating_point_metrics(y_true, (y_score >= 0.5).astype(int))

        assert at_youden["balanced_accuracy"] >= at_half["balanced_accuracy"]
        assert 0.0 < chosen < 1.0

    def test_min_sensitivity_threshold_meets_its_target(self, separable_scores):
        y_true, y_score = separable_scores

        chosen = metrics.threshold_at_sensitivity(y_true, y_score, 0.95)
        report = metrics.operating_point_metrics(y_true, (y_score >= chosen).astype(int))

        assert report["sensitivity"] >= 0.95

    def test_min_sensitivity_trades_specificity_for_recall(self, separable_scores):
        """Screening thresholds should be strictly more permissive than Youden."""
        y_true, y_score = separable_scores

        screening = metrics.threshold_at_sensitivity(y_true, y_score, 0.95)
        balanced = metrics.youden_threshold(y_true, y_score)

        assert screening <= balanced

    def test_thresholds_are_finite(self):
        """roc_curve prepends an infinite threshold; it must never reach a caller."""
        y_true = [1, 1, 0, 0]
        y_score = [0.9, 0.8, 0.2, 0.1]
        assert np.isfinite(metrics.youden_threshold(y_true, y_score))
        assert np.isfinite(metrics.threshold_at_sensitivity(y_true, y_score, 0.99))

    def test_unknown_strategy_is_rejected(self):
        with pytest.raises(ValueError, match="unknown threshold strategy"):
            metrics.choose_threshold([1, 0], [0.9, 0.1], strategy="vibes")


class TestSummarise:
    def test_reports_prevalence_and_size(self, separable_scores):
        y_true, y_score = separable_scores
        report = metrics.summarise(y_true, y_score, threshold=0.5)
        assert report["n"] == 204
        assert report["n_positive"] == 42
        assert report["threshold"] == 0.5

    def test_bootstrap_interval_brackets_the_point_estimate(self, separable_scores):
        y_true, y_score = separable_scores
        report = metrics.summarise(y_true, y_score, threshold=0.5, bootstrap_samples=200, seed=3)

        low, high = report["ci95"]["roc_auc"]
        assert low <= report["roc_auc"] <= high
        assert high - low > 0.0

    def test_bootstrap_is_reproducible_under_a_fixed_seed(self, separable_scores):
        y_true, y_score = separable_scores
        first = metrics.bootstrap_ci(y_true, y_score, samples=100, seed=7)
        second = metrics.bootstrap_ci(y_true, y_score, samples=100, seed=7)
        assert first == second

    def test_bootstrap_preserves_class_prevalence(self, separable_scores):
        """Stratified resampling: sensitivity must stay defined in every replicate.

        An unstratified bootstrap on 42 positives can draw a replicate with none,
        which yields a NaN interval bound for reasons unrelated to the model.
        """
        y_true, y_score = separable_scores
        interval = metrics.bootstrap_ci(y_true, y_score, samples=300, seed=1)
        assert all(np.isfinite(bound) for bound in interval["sensitivity"])


class TestAggregate:
    def test_mean_and_spread_across_folds(self):
        reports = [
            {"roc_auc": 0.80, "sensitivity": 0.70},
            {"roc_auc": 0.90, "sensitivity": 0.80},
        ]
        summary = metrics.aggregate(reports)
        assert summary["roc_auc"]["mean"] == pytest.approx(0.85)
        # ddof=1: a sample of runs, not the population of them.
        assert summary["roc_auc"]["std"] == pytest.approx(np.std([0.8, 0.9], ddof=1))
        assert summary["roc_auc"]["n_runs"] == 2

    def test_single_fold_has_zero_spread_rather_than_nan(self):
        summary = metrics.aggregate([{"roc_auc": 0.8}])
        assert summary["roc_auc"]["std"] == 0.0

    def test_nan_folds_are_excluded_from_the_mean(self):
        summary = metrics.aggregate(
            [{"roc_auc": 0.8}, {"roc_auc": float("nan")}, {"roc_auc": 0.9}]
        )
        assert summary["roc_auc"]["n_runs"] == 2
        assert summary["roc_auc"]["mean"] == pytest.approx(0.85)

    def test_empty_input_is_rejected(self):
        with pytest.raises(ValueError, match="at least one report"):
            metrics.aggregate([])
