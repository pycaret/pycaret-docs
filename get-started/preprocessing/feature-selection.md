# Feature Selection

### Feature Selection

**Feature Importance** is a process **** used to select features in the dataset that contributes the most in predicting the target variable. Working with selected features instead of all the features reduces the risk of over-fitting, improves accuracy, and decreases the training time. In PyCaret, this can be achieved using `feature_selection` parameter. It uses a combination of several supervised feature selection techniques to select the subset of features that are most important for modeling. The size of the subset can be controlled using `feature_selection_threshold` parameter within [setup](https://www.pycaret.org/setup).

#### **PARAMETERS**

* **feature\_selection: bool, default = False**\
  When set to True, a subset of features are selected using a combination of various permutation importance techniques including Random Forest, Adaboost and Linear correlation with target variable. The size of the subset is dependent on the feature\_selection\_param. Generally, this is used to constrain the feature space in order to improve efficiency in modeling. When polynomial\_features and feature\_interaction are used, it is highly recommended to define the feature\_selection\_threshold param with a lower value.
* **feature\_selection\_threshold: float, default = 0.8**\
  Threshold used for feature selection (including newly created polynomial features). A higher value will result in a higher feature space. It is recommended to do multiple trials with different values of feature\_selection\_threshold specially in cases where polynomial\_features and feature\_interaction are used. Setting a very low value may be efficient but could result in under-fitting.

#### **Example**

```
# load dataset
from pycaret.datasets import get_data
diabetes = get_data('diabetes')

# init setup
from pycaret.regression import *
clf1 = setup(data = diabetes, target = 'Class variable', feature_selection = True)
```

#### **Before**

![Dataframe before feature importance](<../../.gitbook/assets/image (372).png>)

#### **After** 

![Dataframe after feature importance](<../../.gitbook/assets/image (388).png>)

### Remove Multicollinearity

**Multicollinearity** (also called _collinearity_) is a phenomenon in which one feature variable in the dataset is highly linearly correlated with another feature variable in the same dataset. Multicollinearity increases the variance of the coefficients, thus making them unstable and noisy for linear models. One such way to deal with Multicollinearity is to drop one of the two features that are highly correlated with each other. This can be achieved in PyCaret using `remove_multicollinearity` parameter within [setup](https://www.pycaret.org/setup).

#### PARAMETERS

* **remove\_multicollinearity: bool, default = False**\
  When set to True, the variables with inter-correlations higher than the threshold defined under the multicollinearity\_threshold param are dropped. When two features are highly correlated with each other, the feature that is less correlated with the target variable is dropped.
* **multicollinearity\_threshold: float, default = 0.9**\
  Threshold used for dropping the correlated features. Only comes into effect when remove\_multicollinearity is set to True.

#### **Example**

```
# load dataset
from pycaret.datasets import get_data
concrete = get_data('concrete')

# init setup
from pycaret.regression import *
reg1 = setup(data = concrete, target = 'strength', remove_multicollinearity = True, multicollinearity_threshold = 0.6)
```

#### **Before**

![Dataframe view before remove multicollinearity](<../../.gitbook/assets/image (387).png>)

#### **After**

![Dataframe view after remove multicollinearity](<../../.gitbook/assets/image (168).png>)

### Principal Component Analysis

**Principal Component Analysis (PCA)** is an unsupervised technique used in machine learning to reduce the dimensionality of a data. It does so by compressing the feature space by identifying a subspace that captures most of the information in the complete feature matrix. It projects the original feature space into lower dimensionality. This can be achieved in PyCaret using `pca` **** parameter within [setup](https://www.pycaret.org/setup).

**PARAMETERS**

* **pca: bool, default = False**\
  When set to True, dimensionality reduction is applied to project the data into a lower dimensional space using the method defined in pca\_method param. In supervised learning pca is generally performed when dealing with high feature space and memory is a constraint. Note that not all datasets can be decomposed efficiently using a linear PCA technique and that applying PCA may result in loss of information. As such, it is advised to run multiple experiments with different pca\_methods to evaluate the impact.
* **pca\_method: string, default = ‘linear’**\
  The ‘linear’ method performs Linear dimensionality reduction using Singular Value Decomposition. The other available options are:
  * **kernel :** dimensionality reduction through the use of RVF kernel.
  * **incremental :** replacement for ‘linear’ pca when the dataset to be decomposed is too large to fit in memory
* **pca\_components: int/float, default = 0.99**\
  Number of components to keep. if pca\_components is a float, it is treated as a target percentage for information retention. When pca\_components is an integer it is treated as the number of features to be kept. pca\_components must be strictly less than the original number of features in the dataset.

#### **Example**

```
# load dataset
from pycaret.datasets import get_data
income = get_data('income')

# init setup
from pycaret.classification import *
clf1 = setup(data = income, target = 'income >50K', pca = True, pca_components = 10)
```

#### **Before**

![Dataframe view before pca](<../../.gitbook/assets/image (493).png>)

#### **After**

![Dataframe view after pca](<../../.gitbook/assets/image (472).png>)

### Ignore Low Variance

Sometimes a dataset may have a **categorical feature** with multiple levels, where distribution of such levels are skewed and one level may dominate over other levels. This means there is not much variation in the information provided by such feature.  For a ML model, such feature may not add a lot of information and thus can be ignored for modeling. This can be achieved in PyCaret using _**ignore\_low\_variance**_ parameter within [setup](https://www.pycaret.org/setup). Both conditions below must be met for a feature to be considered a low variance feature.

Count of unique values in a feature  / sample size < 10%

Count of most common value / Count of second most common value > 20 times.

#### **PARAMETERS**

* **ignore\_low\_variance: bool, default = False**\
  When set to True, all categorical features with statistically insignificant variances are removed from the dataset. The variance is calculated using the ratio of unique  values to the number of samples, and the ratio of the most common value to the frequency of the second most common value.

#### **Example**

```
# load dataset
from pycaret.datasets import get_data
mice = get_data('mice')

# filter dataset
mice = mice[mice['Genotype']] = 'Control'

# init setup
from pycaret.classification import *
clf1 = setup(data = mice, target = 'class', ignore_low_variance = True)
```

#### **Before**

![Dataframe view before ignore low variance](<../../.gitbook/assets/image (105).png>)

#### **After**

![Dataframe view after ignore low variance](<../../.gitbook/assets/image (28).png>)
