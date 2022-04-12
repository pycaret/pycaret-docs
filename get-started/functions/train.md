---
description: Training functions in PyCaret
---

# Train

## compare\_models

This function trains and evaluates the performance of all estimators available in the model library using cross-validation. The output of this function is a scoring grid with average cross-validated scores. Metrics evaluated during CV can be accessed using the `get_metrics` function. Custom metrics can be added or removed using `add_metric` and `remove_metric` function.

### Example

```
# load dataset
from pycaret.datasets import get_data
diabetes = get_data('diabetes')

# init setup
from pycaret.classification import *
clf1 = setup(data = diabetes, target = 'Class variable')

# compare models
best = compare_models()
```

![Output from compare\_models](<../../.gitbook/assets/image (245).png>)

The `compare_models` returns only the top-performing model based on the criteria defined in `sort` parameter. It is `Accuracy` for classification experiments and `R2` for regression. You can change the `sort` order by passing the name of the metric based on which you want to do model selection.&#x20;

### Change the sort order

```
# load dataset
from pycaret.datasets import get_data
diabetes = get_data('diabetes')

# init setup
from pycaret.classification import *
clf1 = setup(data = diabetes, target = 'Class variable')

# compare models
best = compare_models(sort = 'F1')
```

![Output from compare\_models(sort = 'F1')](<../../.gitbook/assets/image (213).png>)

Notice that the sort order of scoring grid is changed now and also the best model returned by this function is selected based on `F1`.

```
print(best)
```

![Output from print(best)](<../../.gitbook/assets/image (71).png>)

### Compare only a few models

If you don't want to do horse racing on the entire model library, you can only compare a few models of your choice by using the `include` parameter.

```
# load dataset
from pycaret.datasets import get_data
diabetes = get_data('diabetes')

# init setup
from pycaret.classification import *
clf1 = setup(data = diabetes, target = 'Class variable')

# compare models
best = compare_models(include = ['lr', 'dt', 'lightgbm'])
```

![Output from compare\_models(include = \['lr', 'dt', 'lightgbm'\])](<../../.gitbook/assets/image (393).png>)

Alternatively, you can also use exclude parameter. This will compare all models except for the ones that are passed in the `exclude` parameter.

```
# load dataset
from pycaret.datasets import get_data
diabetes = get_data('diabetes')

# init setup
from pycaret.classification import *
clf1 = setup(data = diabetes, target = 'Class variable')

# compare models
best = compare_models(exclude = ['lr', 'dt', 'lightgbm'])
```

![Output from compare\_models(exclude = \['lr', 'dt', 'lightgbm'\])](<../../.gitbook/assets/image (545).png>)

### Return more than one model

By default, `compare_models` only return the top-performing model but if you want you can get the Top N models instead of just one model.&#x20;

```
# load dataset
from pycaret.datasets import get_data
diabetes = get_data('diabetes')

# init setup
from pycaret.classification import *
clf1 = setup(data = diabetes, target = 'Class variable')

# compare models
best = compare_models(n_select = 3)
```

![Output from compare\_models(n\_select = 3)](<../../.gitbook/assets/image (342).png>)

Notice that there is no change in the results, however, if you check the variable `best` , it will now contain a list of the top 3 models instead of just one model as seen previously.&#x20;

```
type(best)
>>> list

print(best)
```

![Output from print(best)](<../../.gitbook/assets/image (79).png>)

### Set the budget time

If you are running short on time and would like to set a fixed budget time for this function to run, you can do that by setting the `budget_time` parameter.

```
# load dataset
from pycaret.datasets import get_data
diabetes = get_data('diabetes')

# init setup
from pycaret.classification import *
clf1 = setup(data = diabetes, target = 'Class variable')

# compare models
best = compare_models(budget_time = 0.5)
```

![Output from compare\_models(budget\_time = 0.5)](<../../.gitbook/assets/image (262).png>)

### Set the probability threshold

When performing binary classification, you can change the probability threshold or cut-off value for hard labels. By default, all classifiers use `0.5` as a default threshold.&#x20;

```
# load dataset
from pycaret.datasets import get_data
diabetes = get_data('diabetes')

# init setup
from pycaret.classification import *
clf1 = setup(data = diabetes, target = 'Class variable')

# compare models
best = compare_models(probability_threshold = 0.25)
```

![Output from compare\_models(probability\_threshold = 0.25)](<../../.gitbook/assets/image (92) (1).png>)

Notice that all metrics except for `AUC` are now different. AUC doesn't change because it's not dependent on the hard labels, everything else is dependent on the hard label which is now obtained using `probability_threshold=0.25` .

{% hint style="info" %}
**NOTE:** This parameter is only available in the [Classification](../modules.md) module of PyCaret.
{% endhint %}

### Disable cross-validation

If you don't want to evaluate models using cross-validation and rather just train them and see the metrics on the test/hold-out set you can set the `cross_validation=False.`

```
# load dataset
from pycaret.datasets import get_data
diabetes = get_data('diabetes')

# init setup
from pycaret.classification import *
clf1 = setup(data = diabetes, target = 'Class variable')

# compare models
best = compare_models(cross_validation=False)
```

![Output from compare\_models(cross\_validation=False)](<../../.gitbook/assets/image (291).png>)

The output looks pretty similar but if you focus, the metrics are now different and that's because instead of average cross-validated scores, these are now the metrics on the test/hold-out set.

{% hint style="info" %}
**NOTE:** This function is only available in [Classification](../modules.md) and [Regression](../modules.md) modules.
{% endhint %}

### Distributed training on a cluster

To scale on large datasets you can run `compare_models` function on a cluster in distributed mode using a parameter called `parallel`. It leverages the [Fugue](https://github.com/fugue-project/fugue/) abstraction layer to run `compare_models` on Spark or Dask clusters.&#x20;

```
# load dataset
from pycaret.datasets import get_data
diabetes = get_data('diabetes')

# init setup
from pycaret.classification import *
clf1 = setup(data = diabetes, target = 'Class variable', n_jobs = 1)

# create pyspark session
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()

# import parallel back-end
from pycaret.parallel import FugueBackend

# compare models
best = compare_models(parallel = FugueBackend(spark))
```

![Output from compare\_models(parallel = FugueBackend(spark))](<../../.gitbook/assets/image (378).png>)

{% hint style="info" %}
Note that we need to set `n_jobs = 1` in the setup for testing with local Spark because some models will already try to use all available cores, and running such models in parallel can cause deadlocks from resource contention.&#x20;
{% endhint %}

For Dask, we can specify the `"dask"` inside `FugueBackend` and it will pull the available Dask client.

```
# load dataset
from pycaret.datasets import get_data
diabetes = get_data('diabetes')

# init setup
from pycaret.classification import *
clf1 = setup(data = diabetes, target = 'Class variable', n_jobs = 1)

# import parallel back-end
from pycaret.parallel import FugueBackend

# compare models
best = compare_models(parallel = FugueBackend("dask"))
```

For the complete example and other features related to distributed execution, check [this example](https://github.com/pycaret/pycaret/blob/master/examples/PyCaret%202%20Fugue%20Integration.ipynb). This example also shows how to get the leaderboard in real-time. In a distributed setting, this involves setting up an RPCClient, but Fugue simplifies that.

## create\_model

This function trains and evaluates the performance of a given estimator using cross-validation. The output of this function is a scoring grid with CV scores by fold. Metrics evaluated during CV can be accessed using the `get_metrics` function. Custom metrics can be added or removed using `add_metric` and `remove_metric` function. All the available models can be accessed using the `models` function.

### **Example**&#x20;

```
# load dataset 
from pycaret.datasets import get_data 
diabetes = get_data('diabetes') 

# init setup
from pycaret.classification import * 
clf1 = setup(data = diabetes, target = 'Class variable')

# train logistic regression
lr = create_model('lr')
```

![Output from create\_model('lr')](<../../.gitbook/assets/image (85).png>)

This function displays the performance metrics by fold and the average and standard deviation for each metric and returns the trained model. By default, it uses the `10` fold that can either be changed globally in the [setup](initialize.md#setup) function or locally within `create_model`.

### Changing the fold param

```
# load dataset 
from pycaret.datasets import get_data 
diabetes = get_data('diabetes') 

# init setup
from pycaret.classification import * 
clf1 = setup(data = diabetes, target = 'Class variable')

# train logistic regression
lr = create_model('lr', fold = 5)
```

![Output from create\_model('lr', fold = 5)](<../../.gitbook/assets/image (507).png>)

The model returned by this is the same as above, however, the performance evaluation is done using 5 fold cross-validation.&#x20;

### Model library

To check the list of available models in any module, you can use `models` function.

```
# load dataset 
from pycaret.datasets import get_data 
diabetes = get_data('diabetes') 

# init setup
from pycaret.classification import * 
clf1 = setup(data = diabetes, target = 'Class variable')

# check available models
models()
```

![Output from models()](<../../.gitbook/assets/image (304).png>)

### Models with custom param

When you just run `create_model('dt')`, it will train Decision Tree with all default hyperparameter settings. If you would like to change that, simply pass the attributes in the `create_model` function.&#x20;

```
# load dataset 
from pycaret.datasets import get_data 
diabetes = get_data('diabetes') 

# init setup
from pycaret.classification import * 
clf1 = setup(data = diabetes, target = 'Class variable')

# train decision tree
dt = create_model('dt', max_depth = 5)
```

![Output from create\_model('dt', max\_depth = 5)](<../../.gitbook/assets/image (90).png>)

```
# see models params
print(dt)
```

![](<../../.gitbook/assets/image (52).png>)

### Access the scoring grid

The performance metrics/scoring grid you see after the `create_model` is only displayed and is not returned. As such, if you want to access that grid as `pandas.DataFrame` you will have to use `pull` command after `create_model`.

```
# load dataset 
from pycaret.datasets import get_data 
diabetes = get_data('diabetes') 

# init setup
from pycaret.classification import * 
clf1 = setup(data = diabetes, target = 'Class variable')

# train decision tree
dt = create_model('dt', max_depth = 5)

# access the scoring grid
dt_results = pull()
print(dt_results)
```

![Output from print(dt\_results)](<../../.gitbook/assets/image (26).png>)

```
# check type
type(dt_results)
>>> pandas.core.frame.DataFrame

# select only Mean and SD
dt_results.loc[['Mean', 'SD']]
```

![Output from dt\_results.loc\[\['Mean', 'SD'\]\]](<../../.gitbook/assets/image (150).png>)

### Disable cross-validation

If you don't want to evaluate models using cross-validation and rather just train them and see the metrics on the test/hold-out set you can set the `cross_validation=False.`

```
# load dataset 
from pycaret.datasets import get_data 
diabetes = get_data('diabetes') 

# init setup
from pycaret.classification import * 
clf1 = setup(data = diabetes, target = 'Class variable')

# train model without cv
lr = create_model('lr', cross_validation = False)
```

![Output from create\_model('lr', cross\_validation = False)](<../../.gitbook/assets/image (5).png>)

These are the metrics on the test/hold-out set. That's why you only see one row as opposed to the 12 rows in the original output. When you disable `cross_validation`, the model is only trained one time, on the entire training dataset and scored using the test/hold-out set.

{% hint style="info" %}
**NOTE:** This function is only available in [Classification](../modules.md) and [Regression](../modules.md) modules.
{% endhint %}

### Return train score

The default scoring grid shows the performance metrics on the validation set by fold. If you want to see the performance metrics on the training set by fold as well to examine the over-fitting/under-fitting you can use `return_train_score` parameter.

```
# load dataset 
from pycaret.datasets import get_data 
diabetes = get_data('diabetes') 

# init setup
from pycaret.classification import * 
clf1 = setup(data = diabetes, target = 'Class variable')

# train model without cv
lr = create_model('lr', return_train_score = True)
```

![Output from createmodel('lr', return\_train\_score = True)](<../../.gitbook/assets/image (379).png>)

### Set the probability threshold

When performing binary classification, you can change the probability threshold or cut-off value for hard labels. By default, all classifiers use `0.5` as a default threshold.&#x20;

```
# load dataset 
from pycaret.datasets import get_data 
diabetes = get_data('diabetes') 

# init setup
from pycaret.classification import * 
clf1 = setup(data = diabetes, target = 'Class variable')

# train model with 0.25 threshold
lr = create_model('lr', probability_threshold = 0.25)
```

![Output from create\_model('lr', probability\_threshold = 0.25)](<../../.gitbook/assets/image (172).png>)

```
# see the model
print(lr)
```

![Output from print(lr)](<../../.gitbook/assets/image (349).png>)

### Train models in a loop

You can use the `create_model` function in a loop to train multiple models or even the same model with different configurations and compare their results.

```
# load dataset 
from pycaret.datasets import get_data 
diabetes = get_data('diabetes') 

# init setup
from pycaret.classification import * 
clf1 = setup(data = diabetes, target = 'Class variable')

# train models in a loop
lgbs  = [create_model('lightgbm', learning_rate = i) for i in np.arange(0,1,0.1)]
```

![](<../../.gitbook/assets/image (476) (1).png>)

```
type(lgbs)
>>> list

len(lgbs)
>>> 9
```

If you want to keep track of metrics as well, as in most cases, this is how you can do it.

```
# load dataset 
from pycaret.datasets import get_data 
diabetes = get_data('diabetes') 

# init setup
from pycaret.classification import * 
clf1 = setup(data = diabetes, target = 'Class variable')

# start a loop
models = []
results = []

for i in np.arange(0.1,1,0.1):
    model = create_model('lightgbm', learning_rate = i)
    model_results = pull().loc[['Mean']]
    models.append(model)
    results.append(model_results)
    
results = pd.concat(results, axis=0)
results.index = np.arange(0.1,1,0.1)
results.plot()
```

![Output from results.plot()](<../../.gitbook/assets/image (46).png>)

### Train custom models

You can use your own custom models for training or models from other libraries which are not part of pycaret. As long as their API is consistent with `sklearn`, it will work like a breeze.

```
# load dataset 
from pycaret.datasets import get_data 
diabetes = get_data('diabetes') 

# init setup
from pycaret.classification import * 
clf1 = setup(data = diabetes, target = 'Class variable')

# import custom model
from gplearn.genetic import SymbolicClassifier
sc = SymbolicClassifier()

# train custom model
sc_trained = create_model(sc)
```

![Output from create\_model(sc)](<../../.gitbook/assets/image (300).png>)

```
type(sc_trained)
>>> gplearn.genetic.SymbolicClassifier

print(sc_trained)
```

![Output from print(sc\_trained)](<../../.gitbook/assets/image (426).png>)

### Write your own models

You can also write your own class with `fit` and `predict` function. PyCaret will be compatible with that. Here is a simple example:

```
# load dataset 
from pycaret.datasets import get_data 
insurance= get_data('insurance') 

# init setup
from pycaret.regression import * 
reg1 = setup(data = insurance, target = 'charges')

# create custom estimator
import numpy as np
from sklearn.base import BaseEstimator
class MyOwnModel(BaseEstimator):
    
    def __init__(self):
        self.mean = 0
        
    def fit(self, X, y):
        self.mean = y.mean()
        return self
    
    def predict(self, X):
        return np.array(X.shape[0]*[self.mean])
        
# create an instance
my_own_model = MyOwnModel()

# train model
my_model_trained = create_model(my_own_model)
```

![Output from create\_model(my\_own\_model)](<../../.gitbook/assets/image (39).png>)
