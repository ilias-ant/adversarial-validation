import numpy as np
import pandas as pd
import pytest

from advertion import validate


class TestPublic:
    def test_validate_and_datasets_follow_same_distribution(self):
        seed = 313233

        dataset = pd.DataFrame(
            {
                "feature-1": np.random.default_rng(seed).normal(1.0, 0.1, 5000),
                "feature-2": np.random.default_rng(seed).normal(3, 0.15, 5000),
                "feature-3": np.random.default_rng(seed).normal(7.0, 0.2, 5000),
                "feature-4": np.random.default_rng(seed).normal(9.0, 0.2, 5000),
                "feature-5": np.random.default_rng(seed).normal(6.0, 0.15, 5000),
                "label": np.random.default_rng(seed).choice([0, 1], 5000),
            }
        )

        train = dataset.sample(frac=0.6)

        test = dataset.drop(train.index)
        test = test.drop("label", axis=1)

        verdict = validate(
            trainset=train,
            testset=test,
            target="label",
            smart=True,
            verbose=True,
            random_state=np.random.RandomState(seed),
        )

        assert verdict["datasets_follow_same_distribution"] is True
        assert verdict["mean_roc_auc"] == pytest.approx(0.5, abs=0.1)
        assert verdict["adversarial_features"] == []

    def test_validate_and_datasets_follow_same_distribution_and_a_feature_is_deemed_adversarial(
        self,
    ):
        seed = 313233

        dataset = pd.DataFrame(
            {
                "feature-1": np.random.default_rng(seed).normal(1.0, 0.1, 5000),
                "feature-2": np.random.default_rng(seed).normal(3, 0.15, 5000),
                "feature-3": np.random.default_rng(seed).normal(7.0, 0.2, 5000),
                "feature-4": np.random.default_rng(seed).normal(9.0, 0.2, 5000),
                "feature-5": np.random.default_rng(seed).normal(6.0, 0.15, 5000),
                "label": np.random.default_rng(seed).choice([0, 1], 5000),
            }
        )

        train = dataset.sample(frac=0.6)

        test = dataset.drop(train.index)
        test = test.drop("label", axis=1)
        test["feature-4"] = np.random.default_rng(seed).normal(5.0, 0.2, 2000)

        verdict = validate(
            trainset=train,
            testset=test,
            target="label",
            smart=True,
            verbose=True,
            random_state=np.random.RandomState(seed),
        )

        assert verdict["datasets_follow_same_distribution"] is True
        assert verdict["mean_roc_auc"] == pytest.approx(0.5, abs=0.1)
        assert verdict["adversarial_features"] == ["feature-4"]

    def test_validate_and_datasets_do_not_follow_same_distribution_and_mean_roc_auc_is_low(
        self,
    ):
        seed = 313233

        train = pd.DataFrame(
            {
                "feature-1": np.random.default_rng(seed).normal(1.0, 0.1, 5000),
                "feature-2": np.random.default_rng(seed).normal(3, 0.15, 5000),
                "feature-3": np.random.default_rng(seed).normal(7.0, 0.2, 5000),
                "feature-4": np.random.default_rng(seed).normal(9.0, 0.2, 5000),
                "feature-5": np.random.default_rng(seed).normal(6.0, 0.15, 5000),
                "label": np.random.default_rng(seed).choice([0, 1], 5000),
            }
        )

        test = pd.DataFrame(
            {
                "feature-1": np.random.default_rng(seed).normal(1.0, 0.1, 5000),
                "feature-2": np.random.default_rng(seed).normal(3, 0.15, 5000),
                "feature-3": np.random.default_rng(seed).normal(7.0, 0.2, 5000),
                "feature-4": np.random.default_rng(seed).normal(9.0, 0.2, 5000),
                "feature-5": np.random.default_rng(seed).normal(6.0, 0.15, 5000),
            }
        )

        verdict = validate(
            trainset=train,
            testset=test,
            target="label",
            smart=True,
            verbose=True,
            random_state=np.random.RandomState(seed),
        )

        assert verdict["datasets_follow_same_distribution"] is False
        assert verdict["mean_roc_auc"] == pytest.approx(0.15, abs=0.1)
        assert verdict["adversarial_features"] == []

    def test_version_exists(self):
        try:
            from advertion import __version__

            assert isinstance(__version__, str)

        except ImportError:
            pytest.fail("advertion.__version__ does not exist.")
