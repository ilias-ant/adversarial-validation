# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) (*strictly from v0.1.1 and onwards - 
before v0.1.1, format was a bit freestyle*),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [0.1.1] - 2023-07-22

### Fixed

- wrap preprocessing INFO statement, printed to the stdout, under `verbose` functionality - as expected. This particular
statement got printed even when `verbose=False` was passed to the `validate` function.

    ```shell
    INFO: Working only with available numerical features, 
    categorical features are not yet supported.
    ```

## [0.1.0] - 2023-07-22

The first non pre-release of the package. 🎉

`v0.1.0` is still considered a *beta* release, as the API has not been tested extensively across many and diverse datasets. I have tested it with 3 different Kaggle datasets up to this point.

No changes to the functionality are introduced, only the article [https://ilias-ant.github.io/blog/adversarial-validation/](https://ilias-ant.github.io/blog/adversarial-validation/) is referenced in the README, meant to serve as additional contextual documentation.

## [0.1.0-beta] - 2023-07-20

This is considered the **beta** pre-release version, introducing some minor additions after a bit of personal testing on 2-3 kaggle datasets.

**Features:**

Passing  explicitly a `random_state` is now propagated to the underlying classifier as well.

**Documentation:**

Added short README/homepage introduction on the concept of adversarial validation and where this package stands.

Also, added a homemade package logo (available in README + homepage [https://advertion.readthedocs.io/en/latest/](https://advertion.readthedocs.io/en/latest/))


## [0.1.0-alpha] - 2023-07-19

This is considered the **alpha** pre-release version, introducing some backwards-incompatible changes w.r.t. the previous release.

**Features:**

Response of the main public object, `advertion.validate`, has changed from `bool` to `dict`:

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

Also, upon selecting `smart=True` (*is actually the default case*), an improved identification logic of adversarial features has been introduced, based on the [Kolmogorov–Smirnov test](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test). Having `verbose=True` prints to the standard output the statistic value and the p-value of the test for every feature that is deemed as adversarial.

**Documentation:**

New page on adversarial features: [https://advertion.readthedocs.io/en/latest/adversarial-features/](https://advertion.readthedocs.io/en/latest/adversarial-features/). It is also referenced on the standard output when `smart=True` and `verbose=True`.

**Tests:**

Tests have been developed for the package's public interface, reaching `100%` test coverage on the project.

**CI/CD:**

Continuous Integration - enabled through Github Actions - enriched with 2 additional linters:

- [autoflake](https://pypi.org/project/autoflake/) (*detects unused imports*)
- [bandit](https://pypi.org/project/bandit/) (*detects common software security issues*)

Also, test suite now runs against the following combinations:

```yaml
python-version: ['3.8', '3.9', '3.10', '3.11']
os: [ubuntu-latest, macos-latest, windows-latest]
```

Last but not least, [codecov](https://about.codecov.io/) has been introduced.

For more details, see:

- `.github/workflows/ci.yml`


## [0.1.0-alpha2] - 2023-07-16

A follow-up, pre-alpha release that introduces continuous documentation capabilities to the project, through [MkDocs](https://www.mkdocs.org/) + [readthedocs](https://readthedocs.org/). [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) has been utilized as the theme.

URL: [https://advertion.readthedocs.io/en/latest/](https://advertion.readthedocs.io/en/latest/)

No change to the functionality since inaugural pre-release `v0.1.0-alpha1`.

## [0.1.0-alpha1] - 2023-07-16

This inaugural pre-alpha release introduces the core functionality of adversarial validation, exposed to the end user through the following method:

```python
from advertion import validate

train = pd.read_csv("...")   # let's say target variable is "label"
test = pd.read_csv("...")

are_similar = validate(
    train=train,
    test=test,
    target="label",
)
# are_similar = True: train and test are following the same underlying distribution.
# are_similar = False: test dataset exhibits a different underlying distribution than train dataset.
```
At the same time:

- passing `smart=True` employs a pruning strategy of design matrix features based on feature importance - this helps remove featutes with strongly identifiable properties such as IDs, timestamps etc.
- passing an `n_splits` value controls the number of cross-validation folds that take place internally.
- passing `verbose=True` prints to the standard output informative messages on the adversarial validation strategy.
- passing a `random_state` value ensures reproducible output across multiple function calls.