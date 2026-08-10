"""Tests for configuration loading and validation.

The point of eager validation is that a bad config costs milliseconds instead of
failing after a GPU has been warm for ten minutes, so these tests mostly assert that
invalid values are rejected *at construction* and with a message naming the key.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from skin_cancer_benchmark.config import (
    ConfigError,
    DataConfig,
    EvalConfig,
    ExperimentConfig,
    ModelConfig,
    TrainConfig,
    from_dict,
    load_config,
)

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


@pytest.fixture
def config_file(tmp_path: Path) -> Path:
    payload = {
        "name": "demo",
        "seed": 3,
        "folds": 4,
        "data": {"root": "data/demo", "batch_size": 8},
        "model": {"architecture": "resnet50", "dropout": 0.2},
        "train": {"epochs": 5},
    }
    path = tmp_path / "demo.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


class TestShippedConfigs:
    def test_every_config_in_the_repo_loads(self):
        paths = sorted(CONFIG_DIR.glob("*.yaml"))
        assert len(paths) == 7, "one config per benchmarked architecture"
        for path in paths:
            assert load_config(path).model.architecture == path.stem

    def test_all_configs_share_one_protocol(self):
        """The benchmark's core claim: only the backbone differs between runs."""
        configs = [load_config(path) for path in sorted(CONFIG_DIR.glob("*.yaml"))]

        assert len({(c.seed, c.folds, c.repeats) for c in configs}) == 1
        assert len({(c.data.image_size, c.data.batch_size, c.data.val_fraction) for c in configs}) == 1
        assert len({(c.eval.operating_point, c.eval.min_sensitivity) for c in configs}) == 1
        assert len({(c.train.class_weight, c.train.augment, c.train.monitor) for c in configs}) == 1


class TestLoading:
    def test_reads_sections_and_defaults(self, config_file):
        config = load_config(config_file)
        assert config.name == "demo"
        assert config.data.batch_size == 8
        assert config.model.architecture == "resnet50"
        assert config.train.epochs == 5
        assert config.eval.operating_point == "youden"  # default fills in

    def test_name_falls_back_to_the_filename(self, tmp_path):
        path = tmp_path / "densenet121.yaml"
        path.write_text(yaml.safe_dump({"model": {"architecture": "densenet121"}}), encoding="utf-8")
        assert load_config(path).name == "densenet121"

    def test_paths_are_coerced(self, config_file):
        config = load_config(config_file)
        assert isinstance(config.data.root, Path)
        assert config.data.train_dir == Path("data/demo/Training")

    def test_image_size_list_becomes_a_tuple(self, tmp_path):
        path = tmp_path / "c.yaml"
        path.write_text(yaml.safe_dump({"data": {"image_size": [160, 160]}}), encoding="utf-8")
        assert load_config(path).data.image_size == (160, 160)

    def test_missing_file_names_the_path(self, tmp_path):
        with pytest.raises(ConfigError, match="config file not found"):
            load_config(tmp_path / "nope.yaml")

    def test_unknown_key_is_rejected_with_the_valid_set(self, tmp_path):
        path = tmp_path / "c.yaml"
        path.write_text(yaml.safe_dump({"train": {"epochz": 5}}), encoding="utf-8")
        with pytest.raises(ConfigError, match="unknown config key"):
            load_config(path)


class TestOverrides:
    def test_dotted_keys_reach_nested_sections(self, config_file):
        config = load_config(config_file, {"train.epochs": 99, "seed": 42})
        assert config.train.epochs == 99
        assert config.seed == 42

    def test_overrides_do_not_mutate_the_file(self, config_file):
        load_config(config_file, {"train.epochs": 99})
        assert load_config(config_file).train.epochs == 5


class TestValidation:
    @pytest.mark.parametrize("value", [0.0, 0.5, 0.9, -0.1])
    def test_val_fraction_bounds(self, value):
        with pytest.raises(ConfigError, match="val_fraction"):
            DataConfig(val_fraction=value)

    def test_batch_size_must_be_positive(self):
        with pytest.raises(ConfigError, match="batch_size"):
            DataConfig(batch_size=0)

    def test_unknown_architecture_lists_the_known_ones(self):
        with pytest.raises(ConfigError, match="not registered"):
            ModelConfig(architecture="resnet9000")

    def test_dropout_bounds(self):
        with pytest.raises(ConfigError, match="dropout"):
            ModelConfig(dropout=1.0)

    def test_epochs_must_be_positive(self):
        with pytest.raises(ConfigError, match="epochs"):
            TrainConfig(epochs=0)

    def test_learning_rate_must_be_positive(self):
        with pytest.raises(ConfigError, match="learning rates"):
            TrainConfig(learning_rate=0.0)

    def test_operating_point_must_be_known(self):
        with pytest.raises(ConfigError, match="operating_point"):
            EvalConfig(operating_point="whatever")

    def test_single_fold_is_rejected_and_suggests_the_alternative(self):
        with pytest.raises(ConfigError, match="folds must be >= 2"):
            ExperimentConfig(folds=1)

    def test_repeats_must_be_positive(self):
        with pytest.raises(ConfigError, match="repeats"):
            ExperimentConfig(repeats=0)


class TestSerialisation:
    def test_round_trips_through_plain_types(self, config_file):
        config = load_config(config_file)
        payload = config.to_dict()

        assert isinstance(payload["data"]["root"], str)
        assert from_dict(payload).model.architecture == config.model.architecture

    def test_dict_output_is_json_serialisable(self, config_file):
        import json

        json.dumps(load_config(config_file).to_dict())

    def test_run_dir_combines_output_dir_and_name(self, config_file):
        config = load_config(config_file)
        assert config.run_dir == Path("results") / "demo"
