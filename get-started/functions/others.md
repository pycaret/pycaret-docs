---
description: Other Functions in PyCaret
---

# Others

## pull

Returns the last printed scoring grid. Use `pull` function after any training function to store the scoring grid in `pandas.DataFrame`.

#### Example

```
# loading dataset
from pycaret.datasets import get_data
data = get_data('diabetes')

# init setup
from pycaret.classification import *
clf1 = setup(data, target = 'Class variable')

# compare models
best_model = compare_models()

# get the scoring grid
results = pull()
```

![Output from pull()](<../../.gitbook/assets/image (290).png>)

```
type(results)
>>> pandas.core.frame.DataFrame
```

## models

Return a table containing all the models available in the imported module of the model library.

#### Example

```
# loading dataset
from pycaret.datasets import get_data
data = get_data('diabetes')

# init setup
from pycaret.classification import *
clf1 = setup(data, target = 'Class variable')

# check model library
models()
```

![Output from models()](<../../.gitbook/assets/image (241).png>)

If you want to see a little more information than this, you can pass `internal=True`.

```
# loading dataset
from pycaret.datasets import get_data
data = get_data('diabetes')

# init setup
from pycaret.classification import *
clf1 = setup(data, target = 'Class variable')

# check model library
models(internal = True)
```

![Output from models(internal = True)](<../../.gitbook/assets/image (34).png>)

## get\_config

This function retrieves the global variables created when initializing the [setup](initialize.md#setup) function.&#x20;

#### **Example**

```
# load dataset
from pycaret.datasets import get_data
data = get_data('diabetes')

# init setup
from pycaret.classification import *
clf1 = setup(data, target = 'Class variable')

# get X_train
get_config('X_train')
```

![Output from get\_config('X\_train')](<../../.gitbook/assets/image (137).png>)

Variables accessible by `get_config` function:

* X: Transformed dataset (X)
* y: Transformed dataset (y)
* X\_train: Transformed train dataset (X)
* X\_test: Transformed test/holdout dataset (X)
* y\_train: Transformed train dataset (y)
* y\_test: Transformed test/holdout dataset (y)
* seed: random state set through session\_id
* prep\_pipe: Transformation pipeline
* fold\_shuffle\_param: shuffle parameter used in Kfolds
* n\_jobs\_param: n\_jobs parameter used in model training
* html\_param: html\_param configured through setup
* create\_model\_container: results grid storage container
* master\_model\_container: model storage container
* display\_container: results display container
* exp\_name\_log: Name of experiment
* logging\_param: log\_experiment param
* log\_plots\_param: log\_plots param
* USI: Unique session ID parameter
* fix\_imbalance\_param: fix\_imbalance param
* fix\_imbalance\_method\_param: fix\_imbalance\_method param
* data\_before\_preprocess: data before preprocessing
* target\_param: name of target variable
* gpu\_param: use\_gpu param configured through setup
* fold\_generator: CV splitter configured in fold\_strategy
* fold\_param: fold params defined in the setup
* fold\_groups\_param: fold groups defined in the setup
* stratify\_param: stratify parameter defined in the setup

## set\_config

This function resets the global variables.

#### **Example**

```
# load dataset
from pycaret.datasets import get_data
data = get_data('diabetes')

# init setup
from pycaret.classification import *
clf1 = setup(data, target = 'Class variable', session_id = 123)

# reset environment seed
set_config('seed', 999) 
```

## get\_metrics

Returns the table of all the available metrics in the metric container. All these metrics are used for cross-validation.

```
# load dataset
from pycaret.datasets import get_data
data = get_data('diabetes')

# init setup
from pycaret.classification import *
clf1 = setup(data, target = 'Class variable', session_id = 123)

# get metrics
get_metrics()
```

![Output from get\_metrics()](<../../.gitbook/assets/image (153).png>)

## add\_metric

Adds a custom metric to the metric container.&#x20;

```
# load dataset
from pycaret.datasets import get_data
data = get_data('diabetes')

# init setup
from pycaret.classification import *
clf1 = setup(data, target = 'Class variable', session_id = 123)

# add metric
from sklearn.metrics import log_loss
add_metric('logloss', 'Log Loss', log_loss, greater_is_better = False)
```

![Output from add\_metric('logloss', 'Log Loss', log\_loss, greater\_is\_better = False)](<../../.gitbook/assets/image (269).png>)

Now if you check metric container:

```
get_metrics()
```

![Output from get\_metrics() (after adding log loss metric)](<../../.gitbook/assets/image (442).png>)

## remove\_metric

Removes a metric from the metric container.

```
# remove metric
remove_metric('logloss')
```

No Output. Let's check the metric container again.

```
get_metrics()
```

![Output from get\_metrics() (after removing log loss metric)](<../../.gitbook/assets/image (535).png>)

## automl

This function returns the best model out of all trained models in the current setup based on the `optimize` parameter. Metrics evaluated can be accessed using the `get_metrics` function.

#### Example

```
# load dataset 
from pycaret.datasets import get_data 
data = get_data('diabetes') 

# init setup 
from pycaret.classification import *
clf1 = setup(data, target = 'Class variable') 

# compare models
top5 = compare_models(n_select = 5) 

# tune models
tuned_top5 = [tune_model(i) for i in top5]

# ensemble models
bagged_top5 = [ensemble_model(i) for i in tuned_top5]

# blend models
blender = blend_models(estimator_list = top5) 

# stack models
stacker = stack_models(estimator_list = top5) 

# automl 
best = automl(optimize = 'Recall')
print(best)
```

![Output from print(best)](<../../.gitbook/assets/image (251).png>)

## get\_logs

Returns a table of experiment logs. Only works when `log_experiment = True` when initializing the [setup](initialize.md#setup) function.

#### Example

```
# load dataset
from pycaret.datasets import get_data
data = get_data('diabetes')

# init setup
from pycaret.classification import *
clf1 = setup(data, target = 'Class variable', log_experiment = True, experiment_name = 'diabetes1')

# compare models
top5 = compare_models()

# check ML logs
get_logs()
```

![Output from get\_logs()](<../../.gitbook/assets/image (83).png>)

## **get\_system\_logs**

Read and print `logs.log` file from current active directory.

#### Example

```
# loading dataset
from pycaret.datasets import get_data
data = get_data('diabetes')

# init setup
from pycaret.classification import *
clf1 = setup(data, target = 'Class variable', session_id = 123)

# check system logs
from pycaret.utils import get_system_logs
get_system_logs()
```

![](<../../.gitbook/assets/image (227).png>)
