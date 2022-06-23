---
description: All other setup related parameters
---

# Other setup parameters

### Required Parameters

There are only two non-optional parameters in the setup function.

#### PARAMETERS

* **data: pandas.DataFrame**\
  ****Shape (n\_samples, n\_features), where n\_samples is the number of samples and n\_features is the number of features.
* **target: str**\
  ****Name of the target column to be passed in as a string.&#x20;

### Experiment Logging

PyCaret can automatically log entire experiments including setup parameters, model hyperparameters, performance metrics, and pipeline artifacts. The default settings use [MLflow](https://mlflow.org/) as the logging backend. [wandb](https://wandb.ai/) is also available as an option for logging backend. A parameter in the setup can be enabled to automatically track all the metrics, hyperparameters, and other important information about your machine learning model.&#x20;

#### PARAMETERS

* **log\_experiment: bool, default = bool or string 'mlflow' or 'wandb'**\
  A (list of) PyCaret `BaseLogger` or str (one of `mlflow`, `wandb`) corresponding to a logger to determine which experiment loggers to use. Setting to True will use the MLFlow backend by default.
* **experiment\_name: str, default = None**\
  Name of the experiment for logging. When set to `None`, a default name is used.
* **experiment\_custom\_tags: dict, default = None**\
  ****Dictionary of tag\_name: String -> value: (String, but will be string-ified if not) passed to the mlflow.set\_tags to add new custom tags for the experiment.
* **log\_plots: bool, default = False**\
  When set to `True`, applicable analysis plots are logged as an image file.
* **log\_profile: bool, default = False**\
  When set to `True`, the data profile is logged as an HTML file.&#x20;
* **log\_data: bool, default = False**\
  When set to `True`, train and test dataset are logged as a CSV file.

#### Example

```
# load dataset
from pycaret.datasets import get_data
data = get_data('diabetes')

# init setup
from pycaret.classification import *
clf1 = setup(data, target = 'Class variable', log_experiment = True, experiment_name = 'diabetes1')

# model training
best_model = compare_models() 
```

To initialize `MLflow` server you must run the following command from within the notebook or from the command line. Once the server is initialized, you can track your experiment on `https://localhost:5000`.

```
# init server
!mlflow ui
```

![](<../../.gitbook/assets/image (58).png>)

#### Configure MLflow tracking server

When no backend is configured Data is stored locally at the provided file (or ./mlruns if empty). To configure the backend use `mlflow.set_tracking_uri` before executing the setup function.

* An empty string, or a local file path, prefixed with file:/. Data is stored locally at the provided file (or ./mlruns if empty).
* An HTTP URI like https://my-tracking-server:5000.
* A Databricks workspace, provided as the string “databricks” or, to use a Databricks CLI profile, “databricks://\<profileName>”.

```
# set tracking uri 
import mlflow 
mlflow.set_tracking_uri('file:/c:/users/mlflow-server')

# load dataset
from pycaret.datasets import get_data
data = get_data('diabetes')

# init setup
from pycaret.classification import *
clf1 = setup(data, target = 'Class variable', log_experiment = True, experiment_name = 'diabetes1')
```

#### PyCaret on Databricks

When using PyCaret on Databricks `experiment_name` parameter in the setup must include complete path to storage.  See example below on how to log experiments when using Databricks:

```
# load dataset
from pycaret.datasets import get_data
data = get_data('diabetes')

# init setup
from pycaret.classification import *
clf1 = setup(data, target = 'Class variable', log_experiment = True, experiment_name = '/Users/username@domain.com/experiment-name-here')
```

### Model Selection

Following parameters in the setup can be used for setting parameters for model selection process. These are not related to data preprocessing but can influence your model selection process.

#### PARAMETERS

* **train\_size: float, default = 0.7**\
  ****The proportion of the dataset to be used for training and validation.&#x20;
* **test\_data: pandas.DataFrame, default = None**\
  ****If not `None`, the `test_data` is used as a hold-out set and the `train_size` is ignored. `test_data` must be labeled and the shape of the `data` and `test_data` must match.
* **data\_split\_shuffle: bool, default = True**\
  When set to `False`, prevents shuffling of rows during `train_test_split`.
* **data\_split\_stratify: bool or list, default = False**\
  Controls stratification during the `train_test_split`. When set to `True`, it will stratify by target column. To stratify on any other columns, pass a list of column names. Ignored when `data_split_shuffle` is `False`.
* **fold\_strategy: str or scikit-learn** **CV generator object, default = ‘stratifiedkfold’**\
  Choice of cross-validation strategy. Possible values are:
  * ‘kfold’
  * ‘stratifiedkfold’
  * ‘groupkfold’
  * ‘timeseries’
  * a custom CV generator object compatible with `scikit-learn`.
* **fold: int, default = 10**\
  ****The number of folds to be used in cross-validation. Must be at least 2. This is a global setting that can be over-written at the function level by using the `fold` parameter. Ignored when `fold_strategy` is a custom object.
* **fold\_shuffle: bool, default = False**\
  ****Controls the shuffle parameter of CV. Only applicable when `fold_strategy` is `kfold` or `stratifiedkfold`. Ignored when `fold_strategy` is a custom object.
*   **fold\_groups: str or array-like, with shape (n\_samples,), default = None**

    Optional group labels when ‘GroupKFold’ is used for the cross-validation. It takes an array with shape (n\_samples, ) where n\_samples is the number of rows in the training dataset. When the string is passed, it is interpreted as the column name in the dataset containing group labels.

### Other Miscellaneous

Following parameters in the setup can be used for controlling other experiment settings such as using GPU for training or setting verbosity of the experiment. They do not affect the data in any way.

#### PARAMETERS

* **n\_jobs: int, default = -1**\
  ****The number of jobs to run in parallel (for functions that support parallel processing) -1 means using all processors. To run all functions on single processor set n\_jobs to None.
* **use\_gpu: bool or str, default = False**\
  ****When set to `True`, it will use GPU for training with algorithms that support it and fall back to CPU if they are unavailable. When set to `force` it will only use GPU-enabled algorithms and raise exceptions when they are unavailable. When `False` all algorithms are trained using CPU only.
* **html: bool, default = True**\
  When set to `False`, prevents the runtime display of the monitor. This must be set to `False` when the environment does not support IPython. For example, command line terminal, Databricks, PyCharm, Spyder, and other similar IDEs.
* **session\_id: int, default = None**\
  ****Controls the randomness of the experiment. It is equivalent to `random_state` in `scikit-learn`. When `None`, a pseudo-random number is generated. This can be used for later reproducibility of the entire experiment.
* **silent: bool, default = False**\
  Controls the confirmation input of data types when `setup` is executed. When executing in completely automated mode or on a remote kernel, this must be `True`.
* **verbose: bool, default = True**\
  When set to `False`, Information grid is not printed.
* **profile: bool, default = False**\
  When set to `True`, an interactive EDA report is displayed.
* **profile\_kwargs: dict, default = {} (empty dict)**\
  Dictionary of arguments passed to the ProfileReport method used to create the EDA report. Ignored if `profile` is False.
* **custom\_pipeline: (str, transformer) or list of (str, transformer), default = None**\
  ****When passed, will append the custom transformers in the preprocessing pipeline and are applied on each CV fold separately and on the final fit. All the custom transformations are applied after `train_test_split` and before PyCaret's internal transformations.
* **preprocess: bool, default = True**\
  When set to `False`, no transformations are applied except for `train_test_split` and custom transformations passed in `custom_pipeline` parameter. Data must be ready for modeling (no missing values, no dates, categorical data encoding) when preprocess is set to `False`.
