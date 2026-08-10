"""Tests for manifest building and splitting.

A label bug does not crash; it produces a number that looks fine and measures the
wrong thing. These tests pin the two invariants that would hide such a bug: that
``malignant`` is always the positive class regardless of how the folders are spelled,
and that no image ever appears in both sides of a split.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from skin_cancer_benchmark import data


class TestScanSplit:
    def test_maps_folder_aliases_to_canonical_labels(self, synthetic_dataset):
        frame = data.scan_split(synthetic_dataset / "Training")
        assert set(frame["label"]) == {"benign", "malignant"}
        assert set(frame["raw_label"]) == {"Cancer", "Non_cancer"}

    def test_malignant_is_the_positive_class(self, synthetic_dataset):
        """Sensitivity is meaningless if this flips, and it flips on a folder rename."""
        frame = data.scan_split(synthetic_dataset / "Training")
        malignant = frame[frame["label"] == "malignant"]["y"].unique()
        benign = frame[frame["label"] == "benign"]["y"].unique()
        assert malignant.tolist() == [1]
        assert benign.tolist() == [0]
        assert data.CLASS_NAMES[1] == "malignant"

    def test_counts_match_the_directory(self, synthetic_dataset):
        frame = data.scan_split(synthetic_dataset / "Training")
        assert len(frame) == 48
        assert (frame["label"] == "malignant").sum() == 12

    def test_row_order_is_deterministic(self, synthetic_dataset):
        """Split reproducibility depends on manifest order, not on readdir order."""
        first = data.scan_split(synthetic_dataset / "Training")
        second = data.scan_split(synthetic_dataset / "Training")
        pd.testing.assert_frame_equal(first, second)
        assert first["filepath"].is_monotonic_increasing

    def test_non_image_files_are_ignored(self, synthetic_dataset):
        (synthetic_dataset / "Training" / "Cancer" / "notes.txt").write_text("x")
        frame = data.scan_split(synthetic_dataset / "Training")
        assert len(frame) == 48

    def test_unknown_class_folder_is_an_error(self, synthetic_dataset):
        (synthetic_dataset / "Training" / "unknown_lesion").mkdir()
        with pytest.raises(data.DatasetError, match="unrecognised class folder"):
            data.scan_split(synthetic_dataset / "Training")

    def test_missing_directory_points_at_the_download_script(self, tmp_path):
        with pytest.raises(data.DatasetError, match="download_dataset.py"):
            data.scan_split(tmp_path / "absent")

    def test_single_class_split_is_an_error(self, tmp_path):
        from PIL import Image

        directory = tmp_path / "Training" / "benign"
        directory.mkdir(parents=True)
        Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(directory / "a.png")
        with pytest.raises(data.DatasetError, match="expected both classes"):
            data.scan_split(tmp_path / "Training")


class TestStratifiedHoldout:
    def test_preserves_class_prevalence(self, synthetic_dataset):
        frame = data.scan_split(synthetic_dataset / "Training")
        train, val = data.stratified_holdout(frame, val_fraction=0.25, seed=11)

        assert len(train) + len(val) == len(frame)
        assert val["y"].mean() == pytest.approx(frame["y"].mean(), abs=0.05)

    def test_no_image_appears_on_both_sides(self, synthetic_dataset):
        frame = data.scan_split(synthetic_dataset / "Training")
        train, val = data.stratified_holdout(frame, val_fraction=0.25, seed=11)
        assert set(train["filepath"]).isdisjoint(val["filepath"])

    def test_is_reproducible_for_a_fixed_seed(self, synthetic_dataset):
        frame = data.scan_split(synthetic_dataset / "Training")
        a, _ = data.stratified_holdout(frame, val_fraction=0.25, seed=11)
        b, _ = data.stratified_holdout(frame, val_fraction=0.25, seed=11)
        assert a["filepath"].tolist() == b["filepath"].tolist()

    def test_different_seeds_give_different_splits(self, synthetic_dataset):
        frame = data.scan_split(synthetic_dataset / "Training")
        _, first = data.stratified_holdout(frame, val_fraction=0.25, seed=1)
        _, second = data.stratified_holdout(frame, val_fraction=0.25, seed=2)
        assert first["filepath"].tolist() != second["filepath"].tolist()


class TestStratifiedFolds:
    def test_every_image_validates_exactly_once(self, synthetic_dataset):
        frame = data.scan_split(synthetic_dataset / "Training")
        folds = list(data.stratified_folds(frame, n_splits=4, seed=11))

        validated = [p for _, val in folds for p in val["filepath"]]
        assert len(folds) == 4
        assert sorted(validated) == sorted(frame["filepath"])

    def test_folds_are_disjoint_within_themselves(self, synthetic_dataset):
        frame = data.scan_split(synthetic_dataset / "Training")
        for train, val in data.stratified_folds(frame, n_splits=4, seed=11):
            assert set(train["filepath"]).isdisjoint(val["filepath"])

    def test_each_fold_keeps_both_classes(self, synthetic_dataset):
        frame = data.scan_split(synthetic_dataset / "Training")
        for _, val in data.stratified_folds(frame, n_splits=4, seed=11):
            assert set(val["y"]) == {0, 1}

    def test_more_folds_than_minority_examples_is_rejected(self, synthetic_dataset):
        """Fails loudly rather than yielding a fold with no positives to score."""
        frame = data.scan_split(synthetic_dataset / "Training")
        with pytest.raises(data.DatasetError, match="minority class has only"):
            list(data.stratified_folds(frame, n_splits=13, seed=11))


class TestClassWeights:
    def test_minority_class_is_weighted_up(self, synthetic_dataset):
        frame = data.scan_split(synthetic_dataset / "Training")
        weights = data.class_weights(frame)
        assert weights[1] > weights[0]

    def test_weights_are_normalised_to_mean_one(self, synthetic_dataset):
        """Keeps the weighted loss on the same scale as an unweighted run."""
        frame = data.scan_split(synthetic_dataset / "Training")
        weights = data.class_weights(frame)
        assert np.mean(list(weights.values())) == pytest.approx(1.0)

    def test_balanced_data_gives_equal_weights(self):
        frame = pd.DataFrame({"y": [0, 1, 0, 1], "label": ["benign", "malignant"] * 2})
        weights = data.class_weights(frame)
        assert weights[0] == pytest.approx(weights[1])


class TestDescribe:
    def test_reports_counts_and_prevalence(self, synthetic_dataset):
        frame = data.scan_split(synthetic_dataset / "Testing")
        info = data.describe(frame)
        assert info["n"] == 24
        assert info["counts"] == {"benign": 18, "malignant": 6}
        assert info["prevalence_malignant"] == pytest.approx(0.25)

    def test_split_table_renders_both_splits(self, synthetic_dataset):
        train = data.scan_split(synthetic_dataset / "Training")
        test = data.scan_split(synthetic_dataset / "Testing")
        table = data.summarise_splits([("train", train), ("test", test)])
        assert "train" in table and "test" in table
        assert "malignant" in table


class TestLabelsFrom:
    def test_returns_manifest_row_order(self, synthetic_dataset):
        """Predictions are aligned to this vector, so order is the contract."""
        frame = data.scan_split(synthetic_dataset / "Testing")
        labels = data.labels_from(frame)
        assert labels.tolist() == frame["y"].tolist()
        assert labels.dtype == np.dtype(int)
