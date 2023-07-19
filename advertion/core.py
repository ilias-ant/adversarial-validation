from math import isclose
from statistics import mean
from typing import Iterable, Union

import numpy as np
import pandas as pd
import xgboost as xgb
from pydantic import validate_call
from scipy import stats
from sklearn import metrics, model_selection

from .version import __version__


class AdversarialValidation(object):
    """`AdversarialValidation` is an internal interface performing adversarial validation
    on your training and test datasets.

    **Note**: Any attributes or methods prefixed with _underscores are forming a so-called "private" API, and is
    for internal use only. They may be changed or removed at anytime.
    """

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        smart: bool = True,
        n_splits: int = 5,
        verbose: bool = True,
        random_state: Union[int, np.random.RandomState, None] = None,
    ):
        self._smart = smart
        self._n_splits = n_splits
        self._verbose = verbose
        self._random_state = random_state
        self._av_target = "_av_target_"
        self._statistic_threshold = 0.1
        self._p_value_threshold = 0.05
        self._decision_tolerance = 0.05

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def perform(
        self, trainset: pd.DataFrame, testset: pd.DataFrame, target: str
    ) -> dict:
        """Orchestrates the adversarial validation process.

        - Creates a new feature in both the training and test datasets.
        - Sets the value of the new feature to 0.0 for the training dataset, and 1.0 for the test dataset.
        - Drops original target variable from training dataset.
        - Combines the training and test datasets into a single dataset - let's call it `meta-dataset`.
        - Optionally, identifies and drops adversarial features in the design matrix.
        - Performs cross-validation on the meta-dataset, using the new feature as the target variable.

        Args:
            trainset (pd.DataFrame): The training dataset.
            testset (pd.DataFrame): The test dataset.
            target (str): The target column name.

        Returns:
            dict: An informative key-valued response.
        """
        response = dict()

        print(
            "INFO: Working only with available numerical features, categorical features are not yet supported."
        )
        trainset = self.__preprocessing(trainset)
        testset = self.__preprocessing(testset)

        trainset[self._av_target] = 0.0
        testset[self._av_target] = 1.0

        trainset = trainset.drop(target, axis=1)

        combined = pd.concat([trainset, testset], axis=0, ignore_index=True)

        combined = combined.sample(frac=1)

        X = combined.drop(self._av_target, axis=1)

        y = combined[self._av_target]

        if self._smart:
            adv_features = [*self._identify_adversarial_features(trainset, testset)]
            X = X.drop(adv_features, axis=1)
            response["adversarial_features"] = adv_features

        mean_roc_auc = mean([*self._cross_validate(X, y)])

        no_better_than_random = isclose(
            0.5, mean_roc_auc, abs_tol=self._decision_tolerance
        )

        if self._verbose:
            info = "INFO: training and test datasets "
            info += "follow" if no_better_than_random else "do not follow"
            info += " the same underlying distribution"
            info += f" [mean ROC AUC: {round(mean_roc_auc, 3)}]."

            print(info)
            if mean_roc_auc < 0.4:
                print(
                    f"INFO: The reported ROC AUC value is very low, which may indicate a class confusion problem."
                )

        response["datasets_follow_same_distribution"] = no_better_than_random
        response["mean_roc_auc"] = mean_roc_auc

        return response

    def _cross_validate(self, X: pd.DataFrame, y: pd.Series) -> Iterable[float]:
        """Performs cross-validation, calculating the
        Area Under the Receiver Operating Characteristic Curve (ROC AUC)
        from prediction scores.

        Args:
            X (pd.DataFrame): The design matrix.
            y (pd.Series): The target variable.

        Returns:
            Iterable[float]: An iterable of ROC AUC scores.
        """
        cv = model_selection.StratifiedKFold(
            n_splits=self._n_splits, shuffle=True, random_state=self._random_state
        )

        for train, test in cv.split(X, y):
            X_train, X_test = X.iloc[train], X.iloc[test]
            y_train, y_test = y.iloc[train], y.iloc[test]

            clf = self._classifier()

            clf.fit(X_train, y_train)

            y_pred = clf.predict_proba(X_test)[:, 1]

            yield metrics.roc_auc_score(y_test, y_pred)

    def _identify_adversarial_features(
        self, trainset: pd.DataFrame, testset: pd.DataFrame
    ) -> Iterable[str]:
        """Prunes features from the design matrix, based on a feature importance scheme.

        Args:
            trainset (pd.DataFrame): The training dataset.
            testset (pd.DataFrame): The test dataset.

        Returns:
            Iterable[str]: An iterable of feature names.
        """
        print(
            f"INFO: Will try to identify adv. features "
            f"(see: https://advertion.readthedocs.io/en/{__version__}/adversarial-features)"
        )

        for feature in testset.columns:
            if isinstance(feature, str) and feature == self._av_target:
                continue

            statistic, p_value = stats.kstest(
                trainset[feature].values, testset[feature].values
            )

            if (
                statistic > self._statistic_threshold
                and p_value < self._p_value_threshold
            ):
                if self._verbose:
                    print(
                        f"INFO: Identified adversarial feature: "
                        f"[name: {feature}, statistic: {statistic}, p-value: {p_value}]."
                    )

                yield feature

    @staticmethod
    def __preprocessing(df: pd.DataFrame) -> pd.DataFrame:
        """Performs sensible preprocessing on `df` (in a copy of it).

        Args:
            df (pd.DataFrame): A DataFrame.

        Returns:
            pd.DataFrame: A DataFrame with only numeric-type columns.
        """
        return df.select_dtypes(include=np.number).copy()

    def _classifier(self, **params):
        """Returns a classifier instance.

        Args:
            **params: Arbitrary keyword arguments.

        Returns:
            xgb.XGBClassifier: An XGBoost classifier instance.
        """
        return xgb.XGBClassifier(**params, random_state=self._random_state)
