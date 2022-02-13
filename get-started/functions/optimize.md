---
description: Optimization functions in PyCaret
---

# Optimize

## tune\_model

This function tunes the hyperparameters of the model. The output of this function is a scoring grid with cross-validated scores by fold. The best model is selected based on the metric defined in `optimize` parameter. Metrics evaluated during cross-validation can be accessed using the `get_metrics` function. Custom metrics can be added or removed using `add_metric` and `remove_metric` function.

### **Example**

```
# load dataset
from pycaret.datasets import get_data 
boston = get_data('boston') 

# init setup
from pycaret.regression import * 
reg1 = setup(data = boston, target = 'medv')

# train model
dt = create_model('dt')

# tune model
tuned_dt = tune_model(dt)
```

![Output from tune\_model(dt)](<../../.gitbook/assets/image (427).png>)

To compare the hyperparameters.

```
# default model
print(dt)

# tuned model
print(tuned_dt)
```

![Model hyperparameters before and after tuning](<../../.gitbook/assets/image (166).png>)

### Increasing the iteration

Hyperparameter tuning at the end of the day is an optimization that is constrained by the number of iterations, which eventually depends on how much time and resources you have available. The number of iterations is defined by `n_iter`. By default, it is set to `10`.

```
# load dataset
from pycaret.datasets import get_data 
boston = get_data('boston') 

# init setup
from pycaret.regression import * 
reg1 = setup(data = boston, target = 'medv')

# train model
dt = create_model('dt')

# tune model
tuned_dt = tune_model(dt, n_iter = 50)
```

![Output from tune\_model(dt, n\_iter = 50)](<../../.gitbook/assets/image (244).png>)

#### Comparison of 10 and 50 iterations

{% tabs %}
{% tab title="n_iter = 10" %}
![](<../../.gitbook/assets/image (139).png>)
{% endtab %}

{% tab title="n_iter = 50" %}
![](<../../.gitbook/assets/image (540).png>)
{% endtab %}
{% endtabs %}

### Choosing the metric

When you are tuning the hyperparameters of the model, you must know which metric to optimize for. That can be defined under `optimize` parameter. By default, it is set to `Accuracy` for classification experiments and `R2` for regression.

```
# load dataset
from pycaret.datasets import get_data 
boston = get_data('boston') 

# init setup
from pycaret.regression import * 
reg1 = setup(data = boston, target = 'medv')

# train model
dt = create_model('dt')

# tune model
tuned_dt = tune_model(dt, optimize = 'MAE')
```

![Output from tune\_model(dt, optimize = 'MAE')](<../../.gitbook/assets/image (246).png>)

### Passing custom grid

The tuning grid for hyperparameters is already defined by PyCaret for all the models in the library. However, if you wish you can define your own search space by passing a custom grid using `custom_grid` parameter.&#x20;

```
# load dataset
from pycaret.datasets import get_data 
boston = get_data('boston') 

# init setup
from pycaret.regression import * 
reg1 = setup(boston, target = 'medv')

# train model
dt = create_model('dt')

# define search space
params = {"max_depth": np.random.randint(1, (len(boston.columns)*.85),20),
          "max_features": np.random.randint(1, len(boston.columns),20),
          "min_samples_leaf": [2,3,4,5,6]}
          
# tune model
tuned_dt = tune_model(dt, custom_grid = params)
```

![Output from tune\_model(dt, custom\_grid = params)](<../../.gitbook/assets/image (177).png>)

### Changing the search algorithm

PyCaret integrates seamlessly with many different libraries for hyperparameter tuning. This gives you access to many different types of search algorithms including random, bayesian, optuna, TPE, and a few others. All of this just by changing a parameter. By default, PyCaret using `RandomGridSearch` from the sklearn and you can change that by using `search_library` and `search_algorithm` parameter in the `tune_model` function.

```
# load dataset
from pycaret.datasets import get_data 
boston = get_data('boston') 

# init setup
from pycaret.regression import * 
reg1 = setup(boston, target = 'medv')

# train model
dt = create_model('dt')

# tune model sklearn
tune_model(dt)

# tune model optuna
tune_model(dt, search_library = 'optuna')

# tune model scikit-optimize
tune_model(dt, search_library = 'scikit-optimize')

# tune model tune-sklearn
tune_model(dt, search_library = 'tune-sklearn', search_algorithm = 'hyperopt')
```

{% tabs %}
{% tab title="scikit-learn" %}
![](<../../.gitbook/assets/image (324).png>)
{% endtab %}

{% tab title="optuna" %}
![](<../../.gitbook/assets/image (57) (1).png>)
{% endtab %}

{% tab title="scikit-optimize" %}
![](<../../.gitbook/assets/image (233).png>)
{% endtab %}

{% tab title="tune-sklearn" %}
![](<../../.gitbook/assets/image (252).png>)
{% endtab %}
{% endtabs %}

### Access the tuner

By default PyCaret's `tune_model` function only returns the best model as selected by the tuner. Sometimes you may need access to the tuner object as it may contain important attributes, you can use `return_tuner` parameter.

```
# load dataset
from pycaret.datasets import get_data 
boston = get_data('boston') 

# init setup
from pycaret.regression import * 
reg1 = setup(boston, target = 'medv')

# train model
dt = create_model('dt')

# tune model and return tuner
tuned_model, tuner = tune_model(dt, return_tuner=True)
```

![Output from tune\_model(dt, return\_tuner=True)](<../../.gitbook/assets/image (329).png>)

```
type(tuned_model), type(tuner)
```

![Output from type(tuned\_model), type(tuner)](<../../.gitbook/assets/image (225).png>)

```
print(tuner)
```

![Output from print(tuner)](<../../.gitbook/assets/image (463).png>)

### Automatically choose better

Often times the `tune_model` will not improve the model performance. In fact, it may end up making performance worst than the model with default hyperparameters. This may be problematic when you are not actively experimenting in the Notebook rather you have a python script that runs a workflow of `create_model` --> `tune_model` or `compare_models` --> `tune_model`. To overcome this issue, you can use `choose_better`. When set to `True` it will always return a better performing model meaning that if hyperparameter tuning doesn't improve the performance, it will return the input model.

```
# load dataset
from pycaret.datasets import get_data 
boston = get_data('boston') 

# init setup
from pycaret.regression import * 
reg1 = setup(boston, target = 'medv')

# train model
dt = create_model('dt')

# tune model
dt = tune_model(dt, choose_better = True)
```

![Output from tune\_model(dt, choose\_better = True)](<../../.gitbook/assets/image (261).png>)

{% hint style="info" %}
**NOTE:** `choose_better` doesn't affect the scoring grid that is displayed on the screen. The scoring grid will always present the performance of the best model as selected by the tuner, regardless of the fact that output performance < input performance.
{% endhint %}

## ensemble\_model

This function ensembles a given estimator. The output of this function is a scoring grid with CV scores by fold. Metrics evaluated during CV can be accessed using the `get_metrics` function. Custom metrics can be added or removed using `add_metric` and `remove_metric` function.

### **Example**

```
# load dataset
from pycaret.datasets import get_data 
boston = get_data('boston') 

# init setup
from pycaret.regression import * 
reg1 = setup(boston, target = 'medv')

# train model
dt = create_model('dt')

# ensemble model
bagged_dt = ensemble_model(dt)
```

![Output from ensemble\_model(dt)](<../../.gitbook/assets/image (376).png>)

```
type(bagged_dt)
>>> sklearn.ensemble._bagging.BaggingRegressor

print(bagged_dt)
```

![Output from print(bagged\_dt)](<../../.gitbook/assets/image (33).png>)

### **Changing the fold param**

```
# load dataset
from pycaret.datasets import get_data 
boston = get_data('boston') 

# init setup
from pycaret.regression import * 
reg1 = setup(boston, target = 'medv')

# train model
dt = create_model('dt')

# ensemble model
bagged_dt = ensemble_model(dt, fold = 5)
```

![Output from ensemble\_model(dt, fold = 5)](<../../.gitbook/assets/image (536).png>)

The model returned by this is the same as above, however, the performance evaluation is done using 5 fold cross-validation.&#x20;

### **Method: Bagging**

Bagging, also known as _Bootstrap aggregating_, is a machine learning ensemble meta-algorithm designed to improve the stability and accuracy of machine learning algorithms used in statistical classification and regression. It also reduces variance and helps to avoid overfitting. Although it is usually applied to decision tree methods, it can be used with any type of method. Bagging is a special case of the model averaging approach.

![](<../../.gitbook/assets/image (178).png>)

### **Method: Boosting**

Boosting is an ensemble meta-algorithm for primarily reducing bias and variance in supervised learning. Boosting is in the family of machine learning algorithms that convert weak learners to strong ones. A weak learner is defined to be a classifier that is only slightly correlated with the true classification (it can label examples better than random guessing). In contrast, a strong learner is a classifier that is arbitrarily well-correlated with the true classification.

![](<../../.gitbook/assets/image (17).png>)

### Choosing the method

There are two possible ways you can ensemble your machine learning model with `ensemble_model`. You can define this in the `method` parameter.

```
# load dataset
from pycaret.datasets import get_data 
boston = get_data('boston') 

# init setup
from pycaret.regression import * 
reg1 = setup(boston, target = 'medv')

# train model
dt = create_model('dt')

# ensemble model
boosted_dt = ensemble_model(dt, method = 'Boosting')
```

![Output from ensemble\_model(dt, method = 'Boosting')](<../../.gitbook/assets/image (347).png>)

```
type(boosted_dt)
>>> sklearn.ensemble._weight_boosting.AdaBoostRegressor

print(boosted_dt)
```

![Output from print(boosted\_dt)](<../../.gitbook/assets/image (152).png>)

### Increasing the estimators

By default, PyCaret uses 10 estimators for both `Bagging` or `Boosting`. You can increase that by changing `n_estimators` parameter.

```
# load dataset
from pycaret.datasets import get_data 
boston = get_data('boston') 

# init setup
from pycaret.regression import * 
reg1 = setup(boston, target = 'medv')

# train model
dt = create_model('dt')

# ensemble model
ensemble_model(dt, n_estimators = 100)
```

![Output from ensemble\_model(dt, n\_estimators = 100)](<../../.gitbook/assets/image (441).png>)

### Automatically choose better

Often times the `ensemble_model` will not improve the model performance. In fact, it may end up making performance worst than the model with ensembling. This may be problematic when you are not actively experimenting in the Notebook rather you have a python script that runs a workflow of `create_model` --> `ensemble_model` or `compare_models` --> `ensemble_model`. To overcome this issue, you can use `choose_better`. When set to `True` it will always return a better performing model meaning that if hyperparameter tuning doesn't improve the performance, it will return the input model.

```
# load dataset
from pycaret.datasets import get_data 
boston = get_data('boston') 

# init setup
from pycaret.regression import * 
reg1 = setup(boston, target = 'medv')

# train model
lr = create_model('lr')

# ensemble model
ensemble_model(lr, choose_better = True)
```

![Output from ensemble\_model(lr, choose\_better = True)](<../../.gitbook/assets/image (299).png>)

Notice that with `choose_better = True` the model returned from the `ensemble_model` is a simple `LinearRegression` instead of `BaggedRegressor`. This is because the performance of the model didn't improve after ensembling and hence input model is returned.&#x20;

## blend\_models

This function trains a Soft Voting / Majority Rule classifier for select models passed in the `estimator_list` parameter. The output of this function is a scoring grid with CV scores by fold. Metrics evaluated during CV can be accessed using the `get_metrics` function. Custom metrics can be added or removed using `add_metric` and `remove_metric` function.

### Example

```
# load dataset
from pycaret.datasets import get_data
diabetes = get_data('diabetes')

# init setup
from pycaret.classification import *
clf1 = setup(data = diabetes, target = 'Class variable')

# train a few models
lr = create_model('lr')
dt = create_model('dt')
knn = create_model('knn')

# blend models
blender = blend_models([lr, dt, knn])
```

![Output from blend\_models(\[lr, dt, knn\])](<../../.gitbook/assets/image (479).png>)

```
type(blender)
>>> sklearn.ensemble._voting.VotingClassifier

print(blender)
```

![Output from print(blender)](<../../.gitbook/assets/image (276).png>)

### Changing the fold param

```
# load dataset
from pycaret.datasets import get_data
diabetes = get_data('diabetes')

# init setup
from pycaret.classification import *
clf1 = setup(data = diabetes, target = 'Class variable')

# train a few models
lr = create_model('lr')
dt = create_model('dt')
knn = create_model('knn')

# blend models
blender = blend_models([lr, dt, knn], fold = 5)
```

![Output from blend\_models(\[lr, dt, knn\], fold = 5)](<../../.gitbook/assets/image (32).png>)

The model returned by this is the same as above, however, the performance evaluation is done using 5 fold cross-validation.&#x20;

### Dynamic input estimators

You can also automatically generate the list of input estimators using the [compare\_models](train.md#compare\_models) function. The benefit of this is that you do not have the change your script at all. Every time the top N models are used as an input list.

```
# load dataset
from pycaret.datasets import get_data
diabetes = get_data('diabetes')

# init setup
from pycaret.classification import *
clf1 = setup(data = diabetes, target = 'Class variable')

# blend models
blender = blend_models(compare_models(n_select = 3))
```

![Output from blend\_models(compare\_models(n\_select = 3))](<../../.gitbook/assets/image (157).png>)

Notice here what happens. We passed `compare_models(n_select = 3` as an input to `blend_models`. What happened internally is that the `compare_models` function got executed first and the top 3 models are then passed as an input to the `blend_models` function.&#x20;

```
print(blender)
```

![Output from print(blender)](<../../.gitbook/assets/image (464).png>)

In this example, the top 3 models as evaluated by the `compare_models` are `LogisticRegression`, `LinearDiscriminantAnalysis`, and `RandomForestClassifier`.

### Changing the method

When `method = 'soft'`, it predicts the class label based on the argmax of the sums of the predicted probabilities, which is recommended for an ensemble of well-calibrated classifiers.

```
# load dataset
from pycaret.datasets import get_data
diabetes = get_data('diabetes')

# init setup
from pycaret.classification import *
clf1 = setup(data = diabetes, target = 'Class variable')

# train a few models
lr = create_model('lr')
dt = create_model('dt')
knn = create_model('knn')

# blend models
blender_soft = blend_models([lr,dt,knn], method = 'soft')
```

![Output from blend\_models(\[lr,dt,knn\], method = 'soft')](<../../.gitbook/assets/image (223).png>)

When the `method = 'hard'` , it uses the predictions (hard labels) from input models instead of probabilities.

```
# load dataset
from pycaret.datasets import get_data
diabetes = get_data('diabetes')

# init setup
from pycaret.classification import *
clf1 = setup(data = diabetes, target = 'Class variable')

# train a few models
lr = create_model('lr')
dt = create_model('dt')
knn = create_model('knn')

# blend models
blender_hard = blend_models([lr,dt,knn], method = 'hard')
```

![Output from blend\_models(\[lr,dt,knn\], method = 'hard')](<../../.gitbook/assets/image (81).png>)

The default method is set to `auto` which means it will try to use `soft` method and fall back to `hard` if the former is not supported, this may happen when one of your input models does not support `predict_proba` attribute.

{% hint style="info" %}
**NOTE:** Method parameter is only available in [Classification](../modules.md) module.
{% endhint %}

### Changing the weights

By default, all the input models are given equal weight when blending them but you can explicitly pass the weights to be given to each input model.

```
# load dataset
from pycaret.datasets import get_data
diabetes = get_data('diabetes')

# init setup
from pycaret.classification import *
clf1 = setup(data = diabetes, target = 'Class variable')

# train a few models
lr = create_model('lr')
dt = create_model('dt')
knn = create_model('knn')

# blend models
blender_weighted = blend_models([lr,dt,knn], weights = [0.5,0.2,0.3])
```

![Output from blend\_models(\[lr,dt,knn\], weights = \[0.5,0.2,0.3\])](../../.gitbook/assets/image.png)

You can also tune the weights of the blender using the `tune_model`.

```
# load dataset
from pycaret.datasets import get_data
diabetes = get_data('diabetes')

# init setup
from pycaret.classification import *
clf1 = setup(data = diabetes, target = 'Class variable')

# train a few models
lr = create_model('lr')
dt = create_model('dt')
knn = create_model('knn')

# blend models
blender_weighted = blend_models([lr,dt,knn], weights = [0.5,0.2,0.3])

# tune blender
tuned_blender = tune_model(blender_weighted)
```

![Output from tune\_model(blender\_weighted)](<../../.gitbook/assets/image (234).png>)

```
print(tuned_blender)
```

![Output from print(tuned\_blender)](<../../.gitbook/assets/image (196).png>)

### Automatically choose better

Often times the `blend_models` will not improve the model performance. In fact, it may end up making performance worst than the model with blending. This may be problematic when you are not actively experimenting in the Notebook rather you have a python script that runs a workflow of `compare_models` --> `blend_models`. To overcome this issue, you can use `choose_better`. When set to `True` it will always return a better performing model meaning that if blending the models doesn't improve the performance, it will return the single best performing input model.

```
# load dataset
from pycaret.datasets import get_data
diabetes = get_data('diabetes')

# init setup
from pycaret.classification import *
clf1 = setup(data = diabetes, target = 'Class variable')

# train a few models
lr = create_model('lr')
dt = create_model('dt')
knn = create_model('knn')

# blend models
blend_models([lr,dt,knn], choose_better = True)
```

![Output from blend\_models(\[lr,dt,knn\], choose\_better = True)](<../../.gitbook/assets/image (141).png>)

Notice that because `choose_better=True` the final model returned by this function is `LogisticRegression` instead of `VotingClassifier` because the performance of Logistic Regression was most optimized out of all the given input models plus the blender.

## stack\_models

This function trains a meta-model over select estimators passed in the `estimator_list` parameter. The output of this function is a scoring grid with CV scores by fold. Metrics evaluated during CV can be accessed using the `get_metrics` function. Custom metrics can be added or removed using `add_metric` and `remove_metric` function.

### Example

```
# load dataset
from pycaret.datasets import get_data
diabetes = get_data('diabetes')

# init setup
from pycaret.classification import *
clf1 = setup(data = diabetes, target = 'Class variable')

# train a few models
lr = create_model('lr')
dt = create_model('dt')
knn = create_model('knn')

# stack models
stacker = stack_models([lr, dt, knn])
```

![Output from stack\_models(\[lr, dt, knn\])](<../../.gitbook/assets/image (436).png>)

### Changing the fold param

```
# load dataset
from pycaret.datasets import get_data
diabetes = get_data('diabetes')

# init setup
from pycaret.classification import *
clf1 = setup(data = diabetes, target = 'Class variable')

# train a few models
lr = create_model('lr')
dt = create_model('dt')
knn = create_model('knn')

# stack models
stacker = stack_models([lr, dt, knn], fold = 5)
```

![Output from stack\_models(\[lr, dt, knn\], fold = 5)](<../../.gitbook/assets/image (363).png>)

The model returned by this is the same as above, however, the performance evaluation is done using 5 fold cross-validation.&#x20;

### Dynamic input estimators

You can also automatically generate the list of input estimators using the [compare\_models](train.md#compare\_models) function. The benefit of this is that you do not have the change your script at all. Every time the top N models are used as an input list.

```
# load dataset
from pycaret.datasets import get_data
diabetes = get_data('diabetes')

# init setup
from pycaret.classification import *
clf1 = setup(data = diabetes, target = 'Class variable')

# stack models
stacker = stack_models(compare_models(n_select = 3))
```

![Output from stack\_models(compare\_models(n\_select = 3))](<../../.gitbook/assets/image (440).png>)

Notice here what happens. We passed `compare_models(n_select = 3` as an input to `stack_models`. What happened internally is that the `compare_models` function got executed first and the top 3 models are then passed as an input to the `stack_models` function.&#x20;

```
print(stacker)
```

![Output from print(stacker)](<../../.gitbook/assets/image (236).png>)



In this example, the top 3 models as evaluated by the `compare_models` are `LogisticRegression`, `RandomForestClassifier`, and `LGBMClassifier`.

### Changing the method

There are a few different methods you can explicitly choose for stacking or pass `auto` to be automatically determined. When set to `auto`, it will invoke, for each model, `predict_proba`, `decision_function` or `predict` function in that order. Alternatively, you can define the method explicitly.

```
# load dataset
from pycaret.datasets import get_data
diabetes = get_data('diabetes')

# init setup
from pycaret.classification import *
clf1 = setup(data = diabetes, target = 'Class variable')

# train a few models
lr = create_model('lr')
dt = create_model('dt')
knn = create_model('knn')

# stack models
stacker = stack_models([lr, dt, knn], method = 'predict')
```

![Output from stack\_models(\[lr, dt, knn\], method = 'predict')](<../../.gitbook/assets/image (49).png>)

### Changing the meta-model

When no `meta_model` is passed explicitly, `LogisticRegression` is used for Classification experiments and `LinearRegression` is used for Regression experiments. You can also pass a specific model to be used as a meta-model.

```
# load dataset
from pycaret.datasets import get_data
diabetes = get_data('diabetes')

# init setup
from pycaret.classification import *
clf1 = setup(data = diabetes, target = 'Class variable')

# train a few models
lr = create_model('lr')
dt = create_model('dt')
knn = create_model('knn')

# train meta-model
lightgbm = create_model('lightgbm')

# stack models
stacker = stack_models([lr, dt, knn], meta_model = lightgbm)
```

![Output from stack\_models(\[lr, dt, knn\], meta\_model = lightgbm)](<../../.gitbook/assets/image (254).png>)

```
print(stacker.final_estimator_)
```

![Output from print(stacker.final\_estimator\_)](<../../.gitbook/assets/image (439).png>)

### Restacking

There are two ways you can stack models. (i) only the predictions of input models will be used as training data for meta-model, (ii) predictions as well as the original training data is used for training meta-model.

```
# load dataset
from pycaret.datasets import get_data
diabetes = get_data('diabetes')

# init setup
from pycaret.classification import *
clf1 = setup(data = diabetes, target = 'Class variable')

# train a few models
lr = create_model('lr')
dt = create_model('dt')
knn = create_model('knn')

# stack models
stacker = stack_models([lr, dt, knn], restack = False)
```

![Output from stack\_models(\[lr, dt, knn\], restack = False)](<../../.gitbook/assets/image (397).png>)

## optimize\_threshold

This function optimizes the probability threshold for a trained model. It iterates over performance metrics at different `probability_threshold` with a step size defined in `grid_interval` parameter. This function will display a plot of the performance metrics at each probability threshold and returns the best model based on the metric defined under `optimize` parameter.

### Example

```
# load dataset
from pycaret.datasets import get_data
diabetes = get_data('diabetes')

# init setup
from pycaret.classification import *
clf1 = setup(data = diabetes, target = 'Class variable')

# train a model
knn = create_model('knn')

# optimize threshold
optimized_knn = optimize_threshold(knn)
```

![Output from optimize\_threshold(knn)](<../../.gitbook/assets/image (184).png>)

```
print(optimized_knn)
```

![Output from print(optimized\_knn)](<../../.gitbook/assets/image (66).png>)

## calibrate\_model

This function calibrates the probability of a given model using isotonic or logistic regression. The output of this function is a scoring grid with CV scores by fold. Metrics evaluated during CV can be accessed using the `get_metrics` function. Custom metrics can be added or removed using `add_metric` and `remove_metric` function.

### Example

```
# load dataset
from pycaret.datasets import get_data
diabetes = get_data('diabetes')

# init setup
from pycaret.classification import *
clf1 = setup(data = diabetes, target = 'Class variable')

# train a model
dt = create_model('dt')

# calibrate model
calibrated_dt = calibrate_model(dt)
```

![Output from calibrate\_model(dt)](<../../.gitbook/assets/image (127).png>)

```
print(calibrated_dt)
```

![Output from print(calibrated\_dt)](<../../.gitbook/assets/image (481).png>)

### Before and after calibration

{% tabs %}
{% tab title="Before Calibration" %}
![](<../../.gitbook/assets/image (135).png>)
{% endtab %}

{% tab title="After Calibration" %}
![](<../../.gitbook/assets/image (320).png>)
{% endtab %}
{% endtabs %}
