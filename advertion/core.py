from math import isclose
from statistics import mean
from typing import Iterable, Union

import numpy as np
import pandas as pd
import xgboost as xgb
from pydantic import validate_call
from sklearn import metrics, model_selection


class AdversarialValidation(object):
    """`AdversarialValidation` is an internal interface performing adversarial validation
    on your training and test datasets.

    **Note**: Any attributes or methods prefixed with _underscores are forming a so-called "private" API, and is
    for internal use only. They may be changed or removed at anytime.
    """

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        target: str,
        smart: bool = True,
        n_splits: int = 5,
        verbose: bool = True,
        random_state: Union[int, np.random.RandomState] = None,
    ):
        self._train = train
        self._test = test
        self._target = target
        self._smart = smart
        self._n_splits = n_splits
        self._verbose = verbose
        self._random_state = random_state
        self._av_target = "_av_target_"
        self._feature_importance_threshold = 0.95
        self._tol = 0.05

    def perform(self):
        """Orchestrates the adversarial validation process.

        - Creates a new feature in both the training and test datasets.
        - Sets the value of the new feature to 0.0 for the training dataset, and 1.0 for the test dataset.
        - Drops original target variable from training dataset.
        - Combines the training and test datasets into a single dataset - let's call it `meta-dataset`.
        - Performs cross-validation on the meta-dataset, using the new feature as the target variable.

        Returns:
            bool: Whether the train & test datasets follow the same underlying distribution.
        """
        train = self.__keep_numeric_types(self._train)
        test = self.__keep_numeric_types(self._test)

        train[self._av_target] = 0.0
        test[self._av_target] = 1.0

        train = train.drop(self._target, axis=1)

        combined = pd.concat([train, test], axis=0, ignore_index=True)

        X = combined.drop(self._av_target, axis=1)

        y = combined[self._av_target]

        if self._smart:
            X = self._prune_features(X, y)

        mean_roc_auc = mean([*self._cross_validate(X, y)])

        no_better_than_random = isclose(0.5, mean_roc_auc, abs_tol=self._tol)

        return (
            self.datasets_are_similar(mean_roc_auc)
            if no_better_than_random
            else self.datasets_are_different(mean_roc_auc)
        )

    def datasets_are_similar(self, metric: float) -> bool:
        """Handles the response when the training and test datasets follow the same underlying distribution.

        Args:
            metric (float): The mean ROC AUC score.

        Returns:
            bool: Always returns True.
        """
        if self._verbose:
            print(
                f"INFO: The training and test datasets are similar [mean ROC AUC: {round(metric, 3)}]."
            )
        return True

    def datasets_are_different(self, metric: float) -> bool:
        """Handles the response when the training and test datasets follow different underlying distributions.

        Args:
            metric (float): The mean ROC AUC score.

        Returns:
            bool: Always returns False.
        """
        if self._verbose:
            print(
                f"INFO: The training and test datasets are similar [mean ROC AUC: {round(metric, 3)}]."
            )

            if metric < 0.4:
                print(
                    f"INFO: The reported ROC AUC value is very low, which may indicate a class confusion problem."
                )
        return False

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

    def _prune_features(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Prunes features from the design matrix, based on a feature importance scheme.

        Args:
            X (pd.DataFrame): The design matrix.
            y (pd.Series): The target variable.

        Returns:
            pd.DataFrame: The pruned design matrix.
        """
        clf = self._classifier()

        model = clf.fit(X, y)

        prunable_features = [
            feature
            for feature, importance in zip(X.columns, model.feature_importances_)
            if importance >= self._feature_importance_threshold
        ]

        if self._verbose:
            print(
                f"INFO: The following features were pruned, based on feature importance: {prunable_features}"
            )

        return X.drop(prunable_features, axis=1)

    @staticmethod
    def __keep_numeric_types(df: pd.DataFrame) -> pd.DataFrame:
        """Keeps only numeric columns in the `df`.

        Args:
            df (pd.DataFrame): A DataFrame.

        Returns:
            pd.DataFrame: A DataFrame with only numeric-type columns.
        """
        return df.select_dtypes(include=np.number).copy()

    @staticmethod
    def _classifier(**params):
        """Returns a classifier instance.

        Args:
            **params: Arbitrary keyword arguments.

        Returns:
            xgb.XGBClassifier: An XGBoost classifier instance.
        """
        return xgb.XGBClassifier(**params)
