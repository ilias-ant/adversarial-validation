from typing import Union

import numpy as np
import pandas as pd

from .core import AdversarialValidation


def validate(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
    smart: bool = True,
    n_splits: int = 5,
    verbose: bool = True,
    random_state: Union[int, np.random.RandomState] = None,
) -> bool:
    """Performs adversarial validation on the train & test datasets provided.

    Args:
        train (pd.DataFrame): The training dataset.
        test (pd.DataFrame): The testing dataset.
        target (str): The target column name.
        smart (bool, optional): Whether to prune features with strongly identifiable properties.
        n_splits (int, optional): The number of splits to perform.
        verbose (bool, optional): Whether to print informative messages to the standard output.
        random_state (Union[int, np.random.RandomState], optional): If you wish to ensure reproducible output across \
        multiple function calls.

    Returns:
        bool: Whether the train & test datasets follow the same underlying distribution.

    Raises:
        ValueError: If a validation error occurs, based on the provided parameters.

    Examples:
        >>> from advertion import validate
        >>>
        >>> train = pd.read_csv("...")
        >>> test = pd.read_csv("...")
        >>>
        >>> are_similar = validate(
        >>>     train=train,
        >>>     test=test,
        >>>     target="label",
        >>> )
        >>> # are_similar = True: train and test are following the same
        >>> # underlying distribution.
        >>> # are_similar = False: test dataset exhibits a different
        >>> # underlying distribution than train dataset.

    """
    return AdversarialValidation(
        train=train,
        test=test,
        target=target,
        smart=smart,
        n_splits=n_splits,
        verbose=verbose,
        random_state=random_state,
    ).perform()
