from typing import Union

import numpy as np
import pandas as pd

from .core import AdversarialValidation


def validate(
    trainset: pd.DataFrame,
    testset: pd.DataFrame,
    target: str,
    smart: bool = True,
    n_splits: int = 5,
    verbose: bool = True,
    random_state: Union[int, np.random.RandomState] = None,
) -> dict:
    """Performs adversarial validation on the train & test datasets provided.

    Args:
        trainset (pd.DataFrame): The training dataset.
        testset (pd.DataFrame): The test dataset.
        target (str): The target column name.
        smart (bool, optional): Whether to prune features with strongly identifiable properties.
        n_splits (int, optional): The number of splits to perform.
        verbose (bool, optional): Whether to print informative messages to the standard output.
        random_state (Union[int, np.random.RandomState], optional): If you wish to ensure reproducible output across \
        multiple function calls.

    Returns:
        dict: An informative key-valued response.

    Raises:
        ValueError: If a validation error occurs, based on the provided parameters.

    Examples:
        >>> from advertion import validate
        >>>
        >>> train = pd.read_csv("...")
        >>> test = pd.read_csv("...")
        >>>
        >>> validate(
        >>>     trainset=train,
        >>>     testset=test,
        >>>     target="label",
        >>> )

        >>> // {
        >>> //     "datasets_follow_same_distribution": True,
        >>> //     'mean_roc_auc': 0.5021320833333334,
        >>> //     "adversarial_features': ['id'],
        >>> // }

    """
    return AdversarialValidation(
        smart=smart,
        n_splits=n_splits,
        verbose=verbose,
        random_state=random_state,
    ).perform(trainset=trainset, testset=testset, target=target)
