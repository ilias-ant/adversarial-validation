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
):
    """Performs adversarial validation on the train & test datasets provided.

    Args:
        train (pd.DataFrame): The training dataset.
        test (pd.DataFrame): The testing dataset.
        target (str): The target column name.
        smart (bool, optional): Whether to prune features strong identification properties. Defaults to True.
        n_splits (int, optional): The number of splits to perform. Defaults to 5.
        verbose (bool, optional): Whether to print informative messages. Defaults to True.
        random_state (int, optional): The random state. Defaults to None.

    Returns:
        None

    Raises:
        ValueError: If a validation error occurs, based on the provided parameters.

    Example:

        >>> from advertion import validate
        >>>
        >>> train = pd.read_csv("...")
        >>> test = pd.read_csv("...")
        >>>
        >>> validate(
        >>>     train=train,
        >>>     test=test,
        >>>     target="label",
        >>> )

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
