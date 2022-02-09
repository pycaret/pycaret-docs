---
description: Quick start guide for beginners
---

# 🚀 Quickstart

{% hint style="warning" %}
**NOTE:** We are working to continuously improve the documentation. If you encounter a broken link, please report it on our [GitHub](https://github.com/pycaret/pycaret/issues).
{% endhint %}

## Introduction

Select your use case:

* [Classification](quickstart.md#classification)
* [Regression](quickstart.md#regression)
* [Clustering](quickstart.md#clustering)
* [Anomaly Detection](quickstart.md#anomaly-detection)
* [Natural Language Processing](quickstart.md#natural-language-processing)
* [Association Rules Mining](quickstart.md#association-rules-mining)
* [Time Series (beta)](quickstart.md#time-series-beta)

## Classification

PyCaret’s **Classification Module** is a supervised machine learning module that is used for classifying elements into groups. The goal is to predict the categorical class **labels** which are discrete and unordered. Some common use cases include predicting customer default (Yes or No), predicting customer churn (customer will leave or stay), the disease found (positive or negative). This module can be used for **binary** or **multiclass** problems. It provides several [pre-processing](preprocessing/) features that prepare the data for modeling through the [setup](functions/#setting-up-environment) function. It has over 18 ready-to-use algorithms and [several plots](functions/#plot-model) to analyze the performance of trained models.&#x20;

### Initialize Setup

This function initializes the training environment and creates the transformation pipeline. Setup function must be called before executing any other function. It takes two mandatory parameters: `data` and `target`. All the other parameters are optional.

```
from pycaret.datasets import get_data
data = get_data('diabetes')
```

![](<../.gitbook/assets/image (494).png>)

```
from pycaret.classification import *
s = setup(data, target = 'Class variable')
```

![](<../.gitbook/assets/image (530).png>)

When the `setup` is executed, PyCaret's inference algorithm will automatically infer the data types for all features based on certain properties. The data type should be inferred correctly but this is not always the case. To handle this, PyCaret displays a prompt, asking for data types confirmation, once you execute the `setup`. You can press enter if all data types are correct or type `quit` to exit the setup.

Ensuring that the data types are correct is really important in PyCaret as it automatically performs multiple type-specific preprocessing tasks which are imperative for machine learning models.

Alternatively, you can also use `numeric_features` and `categorical_features` parameters in the `setup` to pre-define the data types.

![Output truncated for display](<../.gitbook/assets/image (288).png>)

### Compare Models

This function trains and evaluates the performance of all the estimators available in the model library using cross-validation. The output of this function is a scoring grid with average cross-validated scores. Metrics evaluated during CV can be accessed using the `get_metrics` function. Custom metrics can be added or removed using `add_metric` and `remove_metric` function.

```
best = compare_models()
```

![](<../.gitbook/assets/image (208).png>)

```
print(best)
```

![](<../.gitbook/assets/image (239).png>)

### Analyze Model

This function analyzes the performance of a trained model on the test set. It may require re-training the model in certain cases.

```
evaluate_model(best)
```

![](<../.gitbook/assets/image (430).png>)

`evaluate_model` can only be used in Notebook since it uses `ipywidget` . You can also use the `plot_model` function to generate plots individually.

```
plot_model(best, plot = 'auc')
```

![](<../.gitbook/assets/image (80).png>)

```
plot_model(best, plot = 'confusion_matrix')
```

![](<../.gitbook/assets/image (117).png>)

### Predictions

This function predicts the `Label` and the `Score` (probability of predicted class) columns using a trained model. When `data` is None, it predicts label and score on the test set (created during the `setup` function).

```
predict_model(best)
```

![](<../.gitbook/assets/image (263).png>)

The evaluation metrics are calculated on the test set. The second output is the `pd.DataFrame` with predictions on the test set (see the last two columns). To generate labels on the unseen (new) dataset, simply pass the dataset in the `predict_model` function

```
predictions = predict_model(best, data=data)
predictions.head()
```

![](<../.gitbook/assets/image (528).png>)

{% hint style="info" %}
`Score` means the probability of the predicted class (NOT the positive class). If Label is 0 and Score is 0.90, it means 90% probability of class 0. If you want to see the probability of both the classes, simply pass `raw_score=True` in the `predict_model` function.
{% endhint %}

```
predictions = predict_model(best, data=data, raw_score=True)
predictions.head()
```

![](<../.gitbook/assets/image (36).png>)

### Save the model

```
save_model(best, 'my_best_pipeline')
```

![](<../.gitbook/assets/image (201).png>)

#### To load the model back in environment:

```
loaded_model = load_model('my_best_pipeline')
print(loaded_model)
```

![](<../.gitbook/assets/image (521).png>)

## Regression

PyCaret’s **Regression Module** is a supervised machine learning module that is used for estimating the relationships between a **dependent variable** (often called the ‘outcome variable’, or ‘target’) and one or more **independent variables** (often called ‘features’, ‘predictors’, or ‘covariates’). The objective of regression is to predict continuous values such as predicting sales amount, predicting quantity, predicting temperature, etc. It provides several [pre-processing](preprocessing/) features that prepare the data for modeling through the [setup](functions/#setting-up-environment) function. It has over 25 ready-to-use algorithms and [several plots](functions/#plot-model) to analyze the performance of trained models.&#x20;

### Initialize Setup

This function initializes the training environment and creates the transformation pipeline. Setup function must be called before executing any other function. It takes two mandatory parameters: `data` and `target`. All the other parameters are optional.

```
from pycaret.datasets import get_data
data = get_data('insurance')
```

![](<../.gitbook/assets/image (121).png>)

```
from pycaret.regression import *
s = setup(data, target = 'charges')
```

![](<../.gitbook/assets/image (130).png>)

When the `setup` is executed, PyCaret's inference algorithm will automatically infer the data types for all features based on certain properties. The data type should be inferred correctly but this is not always the case. To handle this, PyCaret displays a prompt, asking for data types confirmation, once you execute the `setup`. You can press enter if all data types are correct or type `quit` to exit the setup.

Ensuring that the data types are correct is really important in PyCaret as it automatically performs multiple type-specific preprocessing tasks which are imperative for machine learning models.

Alternatively, you can also use `numeric_features` and `categorical_features` parameters in the `setup` to pre-define the data types.

![Output truncated for display](<../.gitbook/assets/image (475).png>)

### Compare Models

This function trains and evaluates the performance of all the estimators available in the model library using cross-validation. The output of this function is a scoring grid with average cross-validated scores. Metrics evaluated during CV can be accessed using the `get_metrics` function. Custom metrics can be added or removed using `add_metric` and `remove_metric` function.

```
best = compare_models()
```

![](<../.gitbook/assets/image (375).png>)

```
print(best)
```

![](<../.gitbook/assets/image (538).png>)

### Analyze Model

This function analyzes the performance of a trained model on the test set. It may require re-training the model in certain cases.

```
evaluate_model(best)
```

![](<../.gitbook/assets/image (91).png>)

`evaluate_model` can only be used in Notebook since it uses `ipywidget` . You can also use the `plot_model` function to generate plots individually.

```
plot_model(best, plot = 'residuals')
```

![](<../.gitbook/assets/image (453).png>)

```
plot_model(best, plot = 'feature')
```

![](<../.gitbook/assets/image (18).png>)

### Predictions

This function predicts `Label` using the trained model. When `data` is None, it predicts label and score on the test set (created during the `setup` function).

```
predict_model(best)
```

![](<../.gitbook/assets/image (465).png>)

The evaluation metrics are calculated on the test set. The second output is the `pd.DataFrame` with predictions on the test set (see the last two columns). To generate labels on the unseen (new) dataset, simply pass the dataset in the `predict_model` function.

```
predictions = predict_model(best, data=data)
predictions.head()
```

![](<../.gitbook/assets/image (143).png>)

### Save the model

```
save_model(best, 'my_best_pipeline')
```

![](<../.gitbook/assets/image (171).png>)

#### To load the model back in the environment:

```
loaded_model = load_model('my_best_pipeline')
print(loaded_model)
```

![](<../.gitbook/assets/image (70).png>)

## Clustering

PyCaret’s **Clustering Module** is an unsupervised machine learning module that performs the task of **grouping** a set of objects in such a way that objects in the same group (also known as a **cluster**) are more similar to each other than to those in other groups. It provides several [pre-processing](preprocessing/) features that prepare the data for modeling through the [setup](functions/#setting-up-environment) function. It has over 10 ready-to-use algorithms and [several plots](functions/#plot-model) to analyze the performance of trained models.&#x20;

### Initialize Setup

This function initializes the training environment and creates the transformation pipeline. Setup function must be called before executing any other function. It takes one mandatory parameter: `data`. All the other parameters are optional.

```
from pycaret.datasets import get_data
data = get_data('jewellery')
```

![](<../.gitbook/assets/image (322).png>)

```
from pycaret.clustering import *
s = setup(data, normalize = True)
```

![](<../.gitbook/assets/image (13).png>)

When the `setup` is executed, PyCaret's inference algorithm will automatically infer the data types for all features based on certain properties. The data type should be inferred correctly but this is not always the case. To handle this, PyCaret displays a prompt, asking for data types confirmation, once you execute the `setup`. You can press enter if all data types are correct or type `quit` to exit the setup.

Ensuring that the data types are correct is really important in PyCaret as it automatically performs multiple type-specific preprocessing tasks which are imperative for machine learning models.

Alternatively, you can also use `numeric_features` and `categorical_features` parameters in the `setup` to pre-define the data types.

![Output truncated for display](<../.gitbook/assets/image (9).png>)

### Create Model

This function trains and evaluates the performance of a given model. Metrics evaluated can be accessed using the `get_metrics` function. Custom metrics can be added or removed using the `add_metric` and `remove_metric` function. All the available models can be accessed using the `models` function.

```
kmeans = create_model('kmeans')
```

![](<../.gitbook/assets/image (113).png>)

```
print(kmeans)
```

![](<../.gitbook/assets/image (108).png>)

### Analyze Model

This function analyzes the performance of a trained model.

```
evaluate_model(kmeans)
```

![](<../.gitbook/assets/image (173).png>)

`evaluate_model` can only be used in Notebook since it uses `ipywidget` . You can also use the `plot_model` function to generate plots individually.

```
plot_model(kmeans, plot = 'elbow')
```

![](<../.gitbook/assets/image (428).png>)

```
plot_model(kmeans, plot = 'silhouette')
```

![](<../.gitbook/assets/image (206).png>)

### Assign Model

This function assigns cluster labels to the training data, given a trained model.

```
result = assign_model(kmeans)
result.head()
```

![](<../.gitbook/assets/image (531).png>)

### Predictions

This function generates cluster labels using a trained model on the new/unseen dataset.

```
predictions = predict_model(kmeans, data = data)
predictions.head()
```

![](<../.gitbook/assets/image (325).png>)

### Save the model

```
save_model(kmeans, 'kmeans_pipeline')
```

![](<../.gitbook/assets/image (516).png>)

#### To load the model back in the environment:

```
loaded_model = load_model('kmeans_pipeline')
print(loaded_model)
```

![](<../.gitbook/assets/image (491).png>)

## Anomaly Detection

PyCaret’s **Anomaly Detection** Module is an unsupervised machine learning module that is used for identifying **rare items**, **events**, or **observations** that raise suspicions by differing significantly from the majority of the data. Typically, the anomalous items will translate to some kind of problems such as bank fraud, a structural defect, medical problems, or errors. It provides several [pre-processing](preprocessing/) features that prepare the data for modeling through the [setup](functions/#setting-up-environment) function. It has over 10 ready-to-use algorithms and [several plots](functions/#plot-model) to analyze the performance of trained models.&#x20;

### Initialize Setup

This function initializes the training environment and creates the transformation pipeline. The `setup` function must be called before executing any other function. It takes one mandatory parameter only: `data`. All the other parameters are optional.

```
from pycaret.datasets import get_data
data = get_data('anomaly')
```

![](<../.gitbook/assets/image (505).png>)

```
from pycaret.anomaly import *
s = setup(data)
```

![](<../.gitbook/assets/image (93).png>)

When the `setup` is executed, PyCaret's inference algorithm will automatically infer the data types for all features based on certain properties. The data type should be inferred correctly but this is not always the case. To handle this, PyCaret displays a prompt, asking for data types confirmation, once you execute the `setup`. You can press enter if all data types are correct or type `quit` to exit the setup.

Ensuring that the data types are correct is really important in PyCaret as it automatically performs multiple type-specific preprocessing tasks which are imperative for machine learning models.

Alternatively, you can also use `numeric_features` and `categorical_features` parameters in the `setup` to pre-define the data types.

![Output truncated for display](<../.gitbook/assets/image (232).png>)

### Create Model

This function trains an unsupervised anomaly detection model. All the available models can be accessed using the `models` function.

```
iforest = create_model('iforest')
print(iforest)
```

![](<../.gitbook/assets/image (451).png>)

```
models()
```

![](<../.gitbook/assets/image (402).png>)

### Analyze Model

```
plot_model(iforest, plot = 'tsne')
```

![](<../.gitbook/assets/image (146).png>)

```
plot_model(iforest, plot = 'umap')
```

![](<../.gitbook/assets/image (486).png>)

### Assign Model

This function assigns anomaly labels to the dataset for a given model. (1 = outlier, 0 = inlier).

```
result = assign_model(iforest)
result.head()
```

![](<../.gitbook/assets/image (94).png>)

### Predictions

work-in-progress.

### Save the model

work-in-progress.

## Natural Language Processing

work-in-progress.

## Association Rules Mining

work-in-progress.

## Time Series (beta)

work-in-progress.
