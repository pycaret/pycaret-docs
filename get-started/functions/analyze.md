---
description: Analysis and model explainability functions in PyCaret
---

# Analyze

## plot\_model

This function analyzes the performance of a trained model on the hold-out set. It may require re-training the model in certain cases.

### **Example**

```
# load dataset
from pycaret.datasets import get_data
diabetes = get_data('diabetes')

# init setup
from pycaret.classification import *
clf1 = setup(data = diabetes, target = 'Class variable')

# creating a model
lr = create_model('lr')

# plot model
plot_model(lr, plot = 'auc')
```

![Output from plot\_model(lr, plot = 'auc')](<../../.gitbook/assets/image (235).png>)

### **Change the scale**

The resolution scale of the figure can be changed with `scale` parameter.

```
# load dataset
from pycaret.datasets import get_data
diabetes = get_data('diabetes')

# init setup
from pycaret.classification import *
clf1 = setup(data = diabetes, target = 'Class variable')

# creating a model
lr = create_model('lr')

# plot model
plot_model(lr, plot = 'auc', scale = 3)
```

![Output from plot\_model(lr, plot = 'auc', scale = 3)](<../../.gitbook/assets/image (307).png>)

### Save the plot

You can save the plot as a `png` file using the `save` parameter.

```
# load dataset
from pycaret.datasets import get_data
diabetes = get_data('diabetes')

# init setup
from pycaret.classification import *
clf1 = setup(data = diabetes, target = 'Class variable')

# creating a model
lr = create_model('lr')

# plot model
plot_model(lr, plot = 'auc', save = True)
```

![Output from plot\_model(lr, plot = 'auc', save = True)](<../../.gitbook/assets/image (467).png>)

### Customize the plot

PyCaret uses [Yellowbrick](https://www.scikit-yb.org/en/latest/) for most of the plotting work. Any argument that is acceptable for Yellowbrick visualizers can be passed as `plot_kwargs` parameter.

```
# load dataset
from pycaret.datasets import get_data
diabetes = get_data('diabetes')

# init setup
from pycaret.classification import *
clf1 = setup(data = diabetes, target = 'Class variable')

# creating a model
lr = create_model('lr')

# plot model
plot_model(lr, plot = 'confusion_matrix', plot_kwargs = {'percent' : True})
```

![Output from plot\_model(lr, plot = 'confusion\_matrix', plot\_kwargs = {'percent' : True})](<../../.gitbook/assets/image (27).png>)

{% tabs %}
{% tab title="Before Customization" %}
![](<../../.gitbook/assets/image (11).png>)
{% endtab %}

{% tab title="After Customization" %}
![](<../../.gitbook/assets/image (295).png>)
{% endtab %}
{% endtabs %}

### Use train data

If you want to assess the model plot on the train data, you can pass `use_train_data=True` in the `plot_model` function.

```
# load dataset
from pycaret.datasets import get_data
diabetes = get_data('diabetes')

# init setup
from pycaret.classification import *
clf1 = setup(data = diabetes, target = 'Class variable')

# creating a model
lr = create_model('lr')

# plot model
plot_model(lr, plot = 'auc', use_train_data = True)
```

![Output from plot\_model(lr, plot = 'auc', use\_train\_data = True)](<../../.gitbook/assets/image (323).png>)

#### Plot on train data vs. hold-out data

{% tabs %}
{% tab title="Train Data" %}
![](<../../.gitbook/assets/image (321).png>)
{% endtab %}

{% tab title="Hold-out Data" %}
![](<../../.gitbook/assets/image (492).png>)
{% endtab %}
{% endtabs %}

### **Examples by module**

#### **Classification**

| **Plot Name**               | **Plot**            |
| --------------------------- | ------------------- |
| Area Under the Curve        | ‘auc’               |
| Discrimination Threshold    | ‘threshold’         |
| Precision Recall Curve      | ‘pr’                |
| Confusion Matrix            | ‘confusion\_matrix’ |
| Class Prediction Error      | ‘error’             |
| Classification Report       | ‘class\_report’     |
| Decision Boundary           | ‘boundary’          |
| Recursive Feature Selection | ‘rfe’               |
| Learning Curve              | ‘learning’          |
| Manifold Learning           | ‘manifold’          |
| Calibration Curve           | ‘calibration’       |
| Validation Curve            | ‘vc’                |
| Dimension Learning          | ‘dimension’         |
| Feature Importance (Top 10) | ‘feature’           |
| Feature IImportance (all)   | 'feature\_all'      |
| Model Hyperparameter        | ‘parameter’         |
| Lift Curve                  | 'lift'              |
| Gain Curve                  | 'gain'              |
| KS Statistic Plot           | 'ks'                |



{% tabs %}
{% tab title="auc" %}
![](<../../.gitbook/assets/image (339).png>)
{% endtab %}

{% tab title="confusion_matrix" %}
![](<../../.gitbook/assets/image (446).png>)
{% endtab %}

{% tab title="threshold" %}
![](<../../.gitbook/assets/image (132).png>)
{% endtab %}

{% tab title="pr" %}
![](<../../.gitbook/assets/image (207).png>)
{% endtab %}

{% tab title="error" %}
![](<../../.gitbook/assets/image (369).png>)
{% endtab %}

{% tab title="class_report" %}
![](<../../.gitbook/assets/image (386).png>)
{% endtab %}

{% tab title="rfe" %}
![](<../../.gitbook/assets/image (354).png>)
{% endtab %}

{% tab title="learning" %}
![](<../../.gitbook/assets/image (95).png>)
{% endtab %}

{% tab title="vc" %}
![](<../../.gitbook/assets/image (503).png>)
{% endtab %}
{% endtabs %}

{% tabs %}
{% tab title="feature" %}
![](<../../.gitbook/assets/image (97).png>)
{% endtab %}

{% tab title="manifold" %}
![](<../../.gitbook/assets/image (466).png>)
{% endtab %}

{% tab title="calibration" %}
![](<../../.gitbook/assets/image (337).png>)
{% endtab %}

{% tab title="dimension" %}
![](<../../.gitbook/assets/image (199).png>)
{% endtab %}

{% tab title="boundary" %}
![](<../../.gitbook/assets/image (112).png>)
{% endtab %}

{% tab title="lift" %}
![](<../../.gitbook/assets/image (277).png>)
{% endtab %}

{% tab title="gain" %}
![](<../../.gitbook/assets/image (340).png>)
{% endtab %}

{% tab title="ks" %}
![](<../../.gitbook/assets/image (148).png>)
{% endtab %}

{% tab title="parameter" %}
![](<../../.gitbook/assets/image (7).png>)
{% endtab %}
{% endtabs %}

#### Regression

| **Name**                    | **Plot**       |
| --------------------------- | -------------- |
| Residuals Plot              | ‘residuals’    |
| Prediction Error Plot       | ‘error’        |
| Cooks Distance Plot         | ‘cooks’        |
| Recursive Feature Selection | ‘rfe’          |
| Learning Curve              | ‘learning’     |
| Validation Curve            | ‘vc’           |
| Manifold Learning           | ‘manifold’     |
| Feature Importance (top 10) | ‘feature’      |
| Feature Importance (all)    | 'feature\_all' |
| Model Hyperparameter        | ‘parameter’    |

{% tabs %}
{% tab title="residuals" %}
![](<../../.gitbook/assets/image (140).png>)
{% endtab %}

{% tab title="error" %}
![](<../../.gitbook/assets/image (60).png>)
{% endtab %}

{% tab title="cooks" %}
![](<../../.gitbook/assets/image (55).png>)
{% endtab %}

{% tab title="rfe" %}
![](<../../.gitbook/assets/image (381).png>)
{% endtab %}

{% tab title="feature" %}
![](<../../.gitbook/assets/image (470).png>)
{% endtab %}

{% tab title="learning" %}
![](<../../.gitbook/assets/image (110) (1).png>)
{% endtab %}

{% tab title="vc" %}
![](<../../.gitbook/assets/image (214).png>)
{% endtab %}

{% tab title="manifold" %}
![](<../../.gitbook/assets/image (496).png>)
{% endtab %}
{% endtabs %}

#### Clustering

| **Name**              | **Plot**       |
| --------------------- | -------------- |
| Cluster PCA Plot (2d) | ‘cluster’      |
| Cluster TSnE (3d)     | ‘tsne’         |
| Elbow Plot            | ‘elbow’        |
| Silhouette Plot       | ‘silhouette’   |
| Distance Plot         | ‘distance’     |
| Distribution Plot     | ‘distribution’ |

{% tabs %}
{% tab title="cluster" %}
![](<../../.gitbook/assets/image (371).png>)
{% endtab %}

{% tab title="tsne" %}
![](<../../.gitbook/assets/image (433).png>)
{% endtab %}

{% tab title="elbow" %}
![](<../../.gitbook/assets/image (410).png>)
{% endtab %}

{% tab title="silhouette" %}
![](<../../.gitbook/assets/image (149).png>)
{% endtab %}

{% tab title="distance" %}
![](<../../.gitbook/assets/image (25).png>)
{% endtab %}

{% tab title="distribution" %}
![](<../../.gitbook/assets/image (517).png>)
{% endtab %}
{% endtabs %}

#### Anomaly Detection

| **Name**                  | **Plot** |
| ------------------------- | -------- |
| t-SNE (3d) Dimension Plot | ‘tsne’   |
| UMAP Dimensionality Plot  | ‘umap’   |

{% tabs %}
{% tab title="tsne" %}
![](<../../.gitbook/assets/image (378).png>)
{% endtab %}

{% tab title="umap" %}
![](<../../.gitbook/assets/image (396).png>)
{% endtab %}
{% endtabs %}

#### Natural Language Processing

| **Name**                  | **Plot**              |
| ------------------------- | --------------------- |
| Word Token Frequency      | ‘frequency’           |
| Word Distribution Plot    | ‘distribution’        |
| Bigram Frequency Plot     | ‘bigram’              |
| Trigram Frequency Plot    | ‘trigram’             |
| Sentiment Polarity Plot   | ‘sentiment’           |
| Part of Speech Frequency  | ‘pos’                 |
| t-SNE (3d) Dimension Plot | ‘tsne’                |
| Topic Model (pyLDAvis)    | ‘topic\_model’        |
| Topic Infer Distribution  | ‘topic\_distribution’ |
| Word cloud                | ‘wordcloud’           |
| UMAP Dimensionality Plot  | ‘umap’                |

{% tabs %}
{% tab title="frequency" %}
![](<../../.gitbook/assets/image (490).png>)
{% endtab %}

{% tab title="distribution" %}
![](<../../.gitbook/assets/image (169).png>)
{% endtab %}

{% tab title="bigram" %}
![](<../../.gitbook/assets/image (422).png>)
{% endtab %}

{% tab title="trigram" %}
![](<../../.gitbook/assets/image (175).png>)
{% endtab %}

{% tab title="sentiment" %}
![](<../../.gitbook/assets/image (435).png>)
{% endtab %}

{% tab title="pos" %}
![](<../../.gitbook/assets/image (194).png>)
{% endtab %}
{% endtabs %}

{% tabs %}
{% tab title="tsne" %}
![](<../../.gitbook/assets/image (460).png>)
{% endtab %}

{% tab title="umap" %}
![](<../../.gitbook/assets/image (431).png>)
{% endtab %}

{% tab title="wordcloud" %}
![](<../../.gitbook/assets/image (187).png>)
{% endtab %}

{% tab title="topic_distribution" %}
![](<../../.gitbook/assets/image (298).png>)
{% endtab %}

{% tab title="topic_model" %}
![](<../../.gitbook/assets/image (249).png>)
{% endtab %}
{% endtabs %}

#### **Association Rule Mining**

{% tabs %}
{% tab title="2d" %}
![](<../../.gitbook/assets/image (158).png>)
{% endtab %}

{% tab title="3d" %}
![](<../../.gitbook/assets/image (218).png>)
{% endtab %}
{% endtabs %}

## evaluate\_model

The `evaluate_model` displays a user interface for analyzing the performance of a trained model. It calls the [plot\_model](analyze.md#plot\_model) function internally.

```
# load dataset
from pycaret.datasets import get_data
juice = get_data('juice')

# init setup
from pycaret.classification import *
exp_name = setup(data = juice,  target = 'Purchase')

# create model
lr = create_model('lr')

# launch evaluate widget
evaluate_model(lr)
```

![Output from evaluate\_model(lr)](<../../.gitbook/assets/image (72).png>)

{% hint style="info" %}
**NOTE:** This function only works in Jupyter Notebook or an equivalent environment.
{% endhint %}

## interpret\_model

This function analyzes the predictions generated from a trained model. Most plots in this function are implemented based on the SHAP (Shapley Additive exPlanations). For more info on this, please see [https://shap.readthedocs.io/en/latest/](https://shap.readthedocs.io/en/latest/)

### Example

```
# load dataset
from pycaret.datasets import get_data
diabetes = get_data('diabetes')

# init setup
from pycaret.classification import *
clf1 = setup(data = diabetes, target = 'Class variable')

# creating a model
xgboost = create_model('xgboost')

# interpret model
interpret_model(xgboost)
```

![Output from interpret\_model(xgboost)](<../../.gitbook/assets/image (303).png>)

### Save the plot

You can save the plot as a `png` file using the `save` parameter.

```
# load dataset
from pycaret.datasets import get_data
diabetes = get_data('diabetes')

# init setup
from pycaret.classification import *
clf1 = setup(data = diabetes, target = 'Class variable')

# creating a model
xgboost = create_model('xgboost')

# interpret model
interpret_model(xgboost, save = True)
```

{% hint style="info" %}
**NOTE:** When `save=True` no plot is displayed in the Notebook.&#x20;
{% endhint %}

### Change plot type

There are a few different plot types available that can be changed by the `plot` parameter.

#### Correlation

```
# load dataset
from pycaret.datasets import get_data
diabetes = get_data('diabetes')

# init setup
from pycaret.classification import *
clf1 = setup(data = diabetes, target = 'Class variable')

# creating a model
xgboost = create_model('xgboost')

# interpret model
interpret_model(xgboost, plot = 'correlation')
```

![Output from interpret\_model(xgboost, plot = 'correlation')](<../../.gitbook/assets/image (502).png>)

By default, PyCaret uses the first feature in the dataset but that can be changed using `feature` parameter.

```
# load dataset
from pycaret.datasets import get_data
diabetes = get_data('diabetes')

# init setup
from pycaret.classification import *
clf1 = setup(data = diabetes, target = 'Class variable')

# creating a model
xgboost = create_model('xgboost')

# interpret model
interpret_model(xgboost, plot = 'correlation', feature = 'Age (years)')
```

![Output from interpret\_model(xgboost, plot = 'correlation', feature = 'Age (years)')](<../../.gitbook/assets/image (271).png>)

#### Partial Dependence Plot

```
# load dataset
from pycaret.datasets import get_data
diabetes = get_data('diabetes')

# init setup
from pycaret.classification import *
clf1 = setup(data = diabetes, target = 'Class variable')

# creating a model
xgboost = create_model('xgboost')

# interpret model
interpret_model(xgboost, plot = 'pdp')
```

![Output from interpret\_model(xgboost, plot = 'pdp')](<../../.gitbook/assets/image (544).png>)

By default, PyCaret uses the first available feature in the dataset but this can be changed using the `feature` parameter.

```
# load dataset
from pycaret.datasets import get_data
diabetes = get_data('diabetes')

# init setup
from pycaret.classification import *
clf1 = setup(data = diabetes, target = 'Class variable')

# creating a model
xgboost = create_model('xgboost')

# interpret model
interpret_model(xgboost, plot = 'pdp', feature = 'Age (years)')
```

![Output from interpret\_model(xgboost, plot = 'pdp', feature = 'Age (years)')](<../../.gitbook/assets/image (8).png>)

#### Morris Sensitivity Analysis

```
# load dataset
from pycaret.datasets import get_data
diabetes = get_data('diabetes')

# init setup
from pycaret.classification import *
clf1 = setup(data = diabetes, target = 'Class variable')

# creating a model
xgboost = create_model('xgboost')

# interpret model
interpret_model(xgboost, plot = 'msa')
```

![Output from interpret\_model(xgboost, plot = 'msa')](<../../.gitbook/assets/image (370).png>)

#### Permutation Feature Importance

```
# load dataset
from pycaret.datasets import get_data
diabetes = get_data('diabetes')

# init setup
from pycaret.classification import *
clf1 = setup(data = diabetes, target = 'Class variable')

# creating a model
xgboost = create_model('xgboost')

# interpret model
interpret_model(xgboost, plot = 'pfi')
```

![Output from interpret\_model(xgboost, plot = 'pfi')](<../../.gitbook/assets/image (500).png>)

#### Reason Plot

```
# load dataset
from pycaret.datasets import get_data
diabetes = get_data('diabetes')

# init setup
from pycaret.classification import *
clf1 = setup(data = diabetes, target = 'Class variable')

# creating a model
xgboost = create_model('xgboost')

# interpret model
interpret_model(xgboost, plot = 'reason')
```

![Output from interpret\_model(xgboost, plot = 'reason')](<../../.gitbook/assets/image (131).png>)

When you generate `reason` plot without passing the specific index of test data, you will get the interactive plot displayed with the ability to select the x and y-axis. This will only be possible if you are using Jupyter Notebook or an equivalent environment. If you want to see this plot for a specific observation, you will have to pass the index in the `observation` parameter.

```
# load dataset
from pycaret.datasets import get_data
diabetes = get_data('diabetes')

# init setup
from pycaret.classification import *
clf1 = setup(data = diabetes, target = 'Class variable')

# creating a model
xgboost = create_model('xgboost')

# interpret model
interpret_model(xgboost, plot = 'reason', observation = 1)
```

![](<../../.gitbook/assets/image (268).png>)

Here the `observation = 1` means index 1 from the test set.

### Use train data

By default, all the plots are generated on the test dataset. If you want to generate plots using a train data set (not recommended) you can use `use_train_data` parameter.

```
# load dataset
from pycaret.datasets import get_data
diabetes = get_data('diabetes')

# init setup
from pycaret.classification import *
clf1 = setup(data = diabetes, target = 'Class variable')

# creating a model
xgboost = create_model('xgboost')

# interpret model
interpret_model(xgboost, use_train_data = True)
```

![Output from interpret\_model(xgboost, use\_train\_data = True)](<../../.gitbook/assets/image (136).png>)

## dashboard

The `dashboard` function generates the interactive dashboard for a trained model. The dashboard is implemented using ExplainerDashboard ([explainerdashboard.readthedocs.io](https://explainerdashboard.readthedocs.io))

#### Dashboard Example

```
# load dataset
from pycaret.datasets import get_data
juice = get_data('juice')

# init setup
from pycaret.classification import *
exp_name = setup(data = juice,  target = 'Purchase')

# train model
lr = create_model('lr')

# launch dashboard
dashboard(lr)
```

![Dashboard (Classification Metrics)](<../../.gitbook/assets/image (64).png>)

![Dashboard (Individual Predictions)](<../../.gitbook/assets/image (501).png>)

![Dashboard (What-if analysis)](<../../.gitbook/assets/image (293).png>)

#### Video:

{% embed url="https://www.youtube.com/watch?v=FZ5-GtdYez0" %}

## eda

This function generates automated Exploratory Data Analysis (EDA) using the AutoViz library. You must install Autoviz separately `pip install autoviz` to use this function.

#### EDA Example

```
# load dataset
from pycaret.datasets import get_data
juice = get_data('juice')

# init setup
from pycaret.classification import *
exp_name = setup(data = juice,  target = 'Purchase')

# launch eda
eda(display_format = 'bokeh')
```

![Output with display\_format = 'bokeh'](<../../.gitbook/assets/image (346).png>)

You can also run this function with `display_format = 'svg'`.

```
# load dataset
from pycaret.datasets import get_data
juice = get_data('juice')

# init setup
from pycaret.classification import *
exp_name = setup(data = juice,  target = 'Purchase')

# launch eda
eda(display_format = 'svg')
```

![Output with display\_format = 'svg'](<../../.gitbook/assets/image (231).png>)

#### Video:

{% embed url="https://www.youtube.com/watch?v=Pm5VOuYqU4Q" %}

## check\_fairness

There are many approaches to conceptualizing fairness. The `check_fairness` function follows the approach known as group fairness, which asks: which groups of individuals are at risk for experiencing harm. `check_fairness` provides fairness-related metrics between different groups (also called sub-population).

#### Check Fairness Example

```
# load dataset
from pycaret.datasets import get_data
income = get_data('income')

# init setup
from pycaret.classification import *
exp_name = setup(data = income,  target = 'income >50K')

# train model
lr = create_model('lr')

# check model fairness
lr_fairness = check_fairness(lr, sensitive_features = ['sex', 'race'])
```

![](<../../.gitbook/assets/image (317).png>)

![](<../../.gitbook/assets/image (509).png>)

#### Video:

{% embed url="https://www.youtube.com/watch?v=mjhDKuLRpM0" %}

## get\_leaderboard

This function returns the leaderboard of all models trained in the current setup.

```
# load dataset
from pycaret.datasets import get_data
diabetes = get_data('diabetes')

# init setup
from pycaret.classification import *
clf1 = setup(data = diabetes, target = 'Class variable')

# compare models
top3 = compare_models(n_select = 3)

# tune top 3 models
tuned_top3 = [tune_model(i) for i in top3]

# ensemble top 3 tuned models
ensembled_top3 = [ensemble_model(i) for i in tuned_top3]

# blender
blender = blend_models(tuned_top3)

# stacker
stacker = stack_models(tuned_top3)

# check leaderboard
get_leaderboard()
```

![Output from get\_leaderboard()](<../../.gitbook/assets/image (287).png>)

You can also access the trained Pipeline with this.&#x20;

```
# check leaderboard
lb = get_leaderboard()

# select top model
lb.iloc[0]['Model']
```

![Output from lb.iloc\[0\]\['Model'\]](<../../.gitbook/assets/image (534).png>)

## assign\_model

This function assigns labels to the training dataset using the trained model. It is available for [Clustering](../modules.md), [Anomaly Detection](../modules.md), and [NLP](../modules.md) modules.

#### Clustering

```
# load dataset
from pycaret.datasets import get_data
jewellery = get_data('jewellery')

# init setup
from pycaret.clustering import *
clu1 = setup(data = jewellery)

# train a model
kmeans = create_model('kmeans')

# assign model
assign_model(kmeans)
```

![Output from assign\_model(kmeans)](<../../.gitbook/assets/image (164).png>)

#### Anomaly Detection

```
# load dataset
from pycaret.datasets import get_data
anomaly = get_data('anomaly')

# init setup
from pycaret.anomaly import *
ano1 = setup(data = anomaly)

# train a model
iforest = create_model('iforest')

# assign model
assign_model(iforest)
```

![Output from assign\_model(iforest)](<../../.gitbook/assets/image (403).png>)

#### Natural Language Processing

```
# load dataset
from pycaret.datasets import get_data
kiva = get_data('kiva')

# init setup
from pycaret.nlp import *
nlp1 = setup(data = kiva, target = 'en')

# train a model
lda = create_model('lda')

# assign model
assign_model(lda)
```

![Output from assign\_model(lda)](<../../.gitbook/assets/image (138).png>)
