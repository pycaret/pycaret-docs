---
description: Learn how to install PyCaret
---

# 💻 Installation

{% hint style="info" %}
**Help us improve the documentation!** If you found a broken link or a typo or would like to contribute to this documentation, please submit a pull request on [pycaret-docs](https://github.com/pycaret/pycaret-docs) repo.
{% endhint %}

## Install

PyCaret is tested and supported on the following 64-bit systems:

* Python 3.6 – 3.8
* Python 3.9 for Ubuntu only
* Ubuntu 16.04 or later
* Windows 7 or later

Install PyCaret with Python's pip package manager.

```
pip install pycaret
```

To install the full version (see dependencies below):

```
pip install pycaret[full]
```

{% hint style="info" %}
If you want to try our nightly build (unstable) you can install **pycaret-nightly** from pip. `pip install pycaret-nightly`
{% endhint %}

## Environment

In order to avoid potential conflicts with other packages, it is strongly recommended to use a virtual environment, e.g. python3 virtualenv (see [python3 virtualenv documentation](https://docs.python.org/3/tutorial/venv.html)) or [conda environments](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html). Using an isolated environment makes it possible to install a specific version of pycaret and its dependencies independently of any previously installed Python packages.&#x20;

```
# create a conda environment
conda create --name yourenvname python=3.8

# activate conda environment
conda activate yourenvname

# install pycaret
pip install pycaret

# create notebook kernel
python -m ipykernel install --user --name yourenvname --display-name "display-name"
```

{% hint style="warning" %}
PyCaret is **not** yet compatible with sklearn>=0.23.2.
{% endhint %}

## GPU

With PyCaret, you can train models on GPU and speed up your workflow by 10x. To train models on GPU simply pass `use_gpu = True` in the setup function. There is no change in the use of the API, however, in some cases, additional libraries have to be installed as they are not installed with the default version or the full version. As of the latest release, the following models can be trained on GPU:

* Extreme Gradient Boosting (requires no further installation)
* Catboost (requires no further installation)
* Light Gradient Boosting Machine requires [GPU installation](https://lightgbm.readthedocs.io/en/latest/GPU-Tutorial.html)
* Logistic Regression, Ridge Classifier, Random Forest, K Neighbors Classifier, K Neighbors Regressor, Support Vector Machine, Linear Regression, Ridge Regression, Lasso Regression requires [cuML >= 0.15](https://github.com/rapidsai/cuml)

## Dependencies

* Default dependencies that are installed with `pip install pycaret` are [listed here](https://github.com/pycaret/pycaret/blob/master/requirements.txt).
* Optional dependencies that are installed with `pycaret[full]` are [listed here](installation.md#install-from-pip).
* Testing requirements are [listed here](https://github.com/pycaret/pycaret/blob/master/requirements-test.txt).

## Building from source

To install the package directly from GitHub (latest source), use the following command:

```
pip install git+https://github.com/pycaret/pycaret.git#egg=pycaret
```

Don't forget to include the `#egg=pycaret` part to explicitly name the project, this way pip can track metadata for it without having to have run the `setup.py` script.

#### Run the tests:

To launch the test suite, run the following command from outside the source directory:

```
pytest pycaret
```

## Docker

Docker uses containers to create virtual environments that isolate a PyCaret installation from the rest of the system. PyCaret docker comes pre-installed with a Notebook environment. that can share resources with its host machine (access directories, use the GPU, connect to the Internet, etc.). The PyCaret Docker images are tested for each release.

```
docker run -p 8888:8888 pycaret/slim
```

For docker image with full version:

```
docker run -p 8888:8888 pycaret/full
```

To learn more, check out [this documentation](https://hub.docker.com/r/pycaret/full).
