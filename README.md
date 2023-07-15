# adversarial-validation

[![PyPI](https://img.shields.io/pypi/v/advertion?color=blue&label=PyPI&logo=PyPI&logoColor=white)](https://pypi.org/project/advertion/) 
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/advertion?logo=python&logoColor=white)](https://www.python.org/) 
[![codecov](https://codecov.io/gh/ilias-ant/advertion/branch/main/graph/badge.svg?token=2H0VB8I8IH)](https://codecov.io/gh/ilias-ant/advertion) 
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) 
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/ilias-ant/advertion/ci.yml?branch=main)](https://github.com/ilias-ant/advertion/actions/workflows/ci.yml) 
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
from advertion import validation

train = pd.read_csv("...")
test = pd.read_csv("...")

are_similar = validate(
    train=train,
    test=test,
    target="label",
)
# are_similar = True: train and test are following 
# the same underlying distribution.
# are_similar = False: test dataset exhibits a different
# underlying distribution than train dataset.
```