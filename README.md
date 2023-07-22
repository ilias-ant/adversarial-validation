# adversarial-validation

[![PyPI](https://img.shields.io/pypi/v/advertion?color=blue&label=PyPI&logo=PyPI&logoColor=white)](https://pypi.org/project/advertion/) 
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/advertion?logo=python&logoColor=white)](https://www.python.org/)
[![codecov](https://codecov.io/gh/ilias-ant/adversarial-validation/branch/main/graph/badge.svg?token=WXJ66ACKTA)](https://codecov.io/gh/ilias-ant/adversarial-validation)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) 
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/ilias-ant/adversarial-validation/ci.yml?branch=main)](https://github.com/ilias-ant/adversarial-validation/actions/workflows/ci.yml)
[![Documentation Status](https://readthedocs.org/projects/advertion/badge/?version=latest)](https://advertion.readthedocs.io/en/latest/?badge=latest)
[![PyPI - Wheel](https://img.shields.io/pypi/wheel/advertion?color=orange)](https://www.python.org/dev/peps/pep-0427/)

A tiny framework to perform adversarial validation of your training and test data.

<img src="https://raw.githubusercontent.com/ilias-ant/adversarial-validation/main/static/logo.png" width="95%" text="figjam">

**What is adversarial validation?**
A common workflow in machine learning projects (especially in Kaggle competitions) is:

1. train your ML model in a training dataset.
2. tune and validate your ML model in a validation dataset (typically is a discrete fraction of the training dataset).
3. finally, assess the actual generalization ability of your ML model in a “held-out” test dataset.

This strategy is widely accepted, but it heavily relies on the assumption that the training and test datasets are drawn 
from the same underlying distribution. This is often referred to as the “*identically distributed*” property in the 
literature.

This package helps you easily assert whether the "*identically distributed*" property holds true for your training and 
test datasets or equivalently whether your validation dataset is a good proxy for your model's performance on the unseen 
test instances.

If you are a person of details, feel free to take a deep dive to the following companion article:

[adversarial validation: can i trust my validation dataset?](https://ilias-ant.github.io/blog/adversarial-validation/)

## Install

The recommended installation is via `pip`:

```bash
pip install advertion
```

(*advertion stands for **adver**sarial valida**tion***)

## Usage

```python
from advertion import validate

train = pd.read_csv("...")
test = pd.read_csv("...")

validate(
    trainset=train,
    testset=test,
    target="label",
)

# // {
# //     "datasets_follow_same_distribution": True,
# //     'mean_roc_auc': 0.5021320833333334,
# //     "adversarial_features': ['id'],
# // }
```

## How to contribute

If you wish to contribute, [this](CONTRIBUTING.md) is a great place to start!

## License

Distributed under the [Apache License 2.0](LICENSE).
