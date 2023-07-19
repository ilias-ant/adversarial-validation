# adversarial-validation

[![PyPI](https://img.shields.io/pypi/v/advertion?color=blue&label=PyPI&logo=PyPI&logoColor=white)](https://pypi.org/project/advertion/) 
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/advertion?logo=python&logoColor=white)](https://www.python.org/)
[![codecov](https://codecov.io/gh/ilias-ant/adversarial-validation/branch/main/graph/badge.svg?token=WXJ66ACKTA)](https://codecov.io/gh/ilias-ant/adversarial-validation)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) 
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/ilias-ant/adversarial-validation/ci.yml?branch=main)](https://github.com/ilias-ant/adversarial-validation/actions/workflows/ci.yml)
[![Documentation Status](https://readthedocs.org/projects/advertion/badge/?version=latest)](https://advertion.readthedocs.io/en/latest/?badge=latest)
[![PyPI - Wheel](https://img.shields.io/pypi/wheel/advertion?color=orange)](https://www.python.org/dev/peps/pep-0427/)

A tiny framework to perform adversarial validation of your training and test data.

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