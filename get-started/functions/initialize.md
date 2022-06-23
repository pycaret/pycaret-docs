---
description: Functions that initialize experiment in PyCaret
---

# Initialize

## setup

This function initializes the experiment in PyCaret and creates the transformation pipeline based on all the parameters passed in the function. Setup function must be called before executing any other function. It takes two mandatory parameters: `data` and `target`. All the other parameters are optional.

#### Example

```
# load dataset
from pycaret.datasets import get_data
diabetes = get_data('diabetes')

# init setup
from pycaret.classification import *
clf1 = setup(data = diabetes, target = 'Class variable')
```

As soon as you run the setup function, PyCaret will automatically infer the data types of all the variables in the dataset. If these are correctly inferred, you can press enter to continue.

![](<../../.gitbook/assets/image (390).png>)

Once you press enter to continue, you will see the output like this:

![Output truncated for display](<../../.gitbook/assets/image (362).png>)

All the preprocessing and data transformations are configured within the setup function. There are many options to choose from, from data cleaning to feature engineering. To learn more about all the available Preprocessing, [see this page](../preprocessing/).

{% hint style="info" %}
**NOTE:** If you do not want to see the data type confirmation, you can pass `silent=True` in the setup to run it without any interruption.&#x20;
{% endhint %}

### Required Parameters

There are many parameters in the setup function but only two are non-optional.

* **data: pandas.DataFrame**\
  ****Shape (n\_samples, n\_features), where n\_samples is the number of samples and n\_features is the number of features.
* **target: str**\
  ****Name of the target column to be passed in as a string.&#x20;

{% hint style="info" %}
**NOTE:** target parameter is not required for unsupervised modules such as `clustering`, `anomaly detection` or `NLP`.
{% endhint %}

### Default Transformations

All the preprocessing steps in setup are simply a flag of `True` or `False` . For example, if you want to scale your features, you will have to pass `normalize=True` in the setup function. However, there are three things that will happen by default:

* [Missing Value Imputation](../preprocessing/data-preparation.md#missing-values)
* [One-Hot Encoding](../preprocessing/data-preparation.md#one-hot-encoding)
* [Train-Test Split](../preprocessing/other-setup-parameters.md)

### Experiment Logging

PyCaret uses [MLflow](https://mlflow.org/) for experiment tracking. A parameter in the setup can be enabled to automatically track all the metrics, hyperparameters, and other important information about your machine learning model.&#x20;

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

To initialize the `MLflow` server you must run the following command from within the notebook or from the command line. Once the server is initialized, you can track your experiment on `https://localhost:5000`.

```
# init server
!mlflow ui
```

![](<../../.gitbook/assets/image (65).png>)

To learn more about experiment tracking in PyCaret, [see this page](../preprocessing/other-setup-parameters.md#experiment-logging).

### Model Validation

There are quite a few parameters in the setup function that are not directly related to preprocessing or data transformation but they are used as part of model validation and selection strategy such as `train_size`, `fold_strategy`, or number of `fold` for cross-validation. To learn more about all the model validation and selection settings in the setup, see [this page](../preprocessing/other-setup-parameters.md#model-selection).

### GPU Support&#x20;

With PyCaret, you can train models on GPU and speed up your workflow by 10x. To train models on GPU simply pass `use_gpu = True` in the setup function. There is no change in the use of the API, however, in some cases, additional libraries have to be installed as they are not installed with the default version or the full version. To learn more about GPU support, see [this page](../installation.md#gpu).

### Examples

To see the use of the `setup` in other modules of PyCaret, see below:

* [Classification](../quickstart.md#classification)
* [Regression](../quickstart.md#regression)
* [Clustering](../quickstart.md#clustering)
* [Anomaly Detection](../quickstart.md#anomaly-detection)
* [Natural Language Processing](../quickstart.md#natural-language-processing)
* [Association Rules Mining](../quickstart.md#association-rules-mining)

{% hint style="warning" %}
**NOTE:** setup function uses global environment variables in Python. Therefore, if you run the `setup` function twice in the same script, it will overwrite the previous experiment. PyCaret next major release will include a new object-oriented API that will make it possible to create multiple instances through class instances.&#x20;
{% endhint %}
