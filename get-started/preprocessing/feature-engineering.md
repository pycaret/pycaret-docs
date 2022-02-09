# Feature Engineering

### Feature Interaction

It is often seen in machine learning experiments when two features combined through an **arithmetic operation** becomes more significant in explaining variances in the data, than the same two features separately. Creating a new feature through interaction of existing features is known as **feature interaction**. It can achieved in PyCaret using `feature_interaction` and `feature_ratio` __ parameters within [setup](https://www.pycaret.org/setup). Feature interaction creates new features by multiplying two variables (a \* b), while feature ratios create new features but by calculating the ratios of existing features (a / b).

#### **PARAMETERS**

* **feature\_interaction: bool, default = False**\
  When set to True, it will create new features by interacting (a \* b) for all numeric variables in the dataset including polynomial and trigonometric features (if created). This feature is not scalable and may not work as expected on datasets with large feature space.
* **feature\_ratio: bool, default = False**\
  When set to True, it will create new features by calculating the ratios (a / b) of all numeric variables in the dataset. This feature is not scalable and may not work as expected on datasets with large feature space.
* **interaction\_threshold: bool, default = 0.01**\
  Similar to polynomial\_threshold, It is used to compress a sparse matrix of newly created features through interaction. Features whose importance based on the combination of Random Forest, AdaBoost and Linear correlation falls within the percentile of the defined threshold are kept in the dataset. Remaining features are dropped before further processing.

#### **Example**

```
# load dataset
from pycaret.datasets import get_data
insurance = get_data('insurance')

# init setup
from pycaret.regression import *
reg1 = setup(data = insurance, target = 'charges', feature_interaction = True, feature_ratio = True)
```

#### **Before**

![Dataframe view before feature interaction](<../../.gitbook/assets/image (351).png>)

#### **After**

![Dataframe view after feature interaction](<../../.gitbook/assets/image (76).png>)

### Polynomial Features

In machine learning experiments the relationship between the dependent and independent variable is often assumed as linear, however this is not always the case. Sometimes the relationship between dependent and independent variables is more complex. Creating new polynomial features sometimes might help in capturing that relationship which otherwise may go unnoticed. PyCaret can create polynomial features from existing features using `polynomial_features` parameter within [setup](https://www.pycaret.org/setup).

#### **PARAMETERS**

* **polynomial\_features: bool, default = False**\
  When set to True, new features are created based on all polynomial combinations that exist within the numeric features in a dataset to the degree defined in polynomial\_degree param.
* **polynomial\_degree: int, default = 2**\
  Degree of polynomial features. For example, if an input sample is two dimensional and of the form \[a, b], the polynomial features with degree = 2 are: \[1, a, b, a^2, ab, b^2].
* **polynomial\_threshold: float, default = 0.1**\
  This is used to compress a sparse matrix of polynomial and trigonometric features. Polynomial and trigonometric features whose feature importance based on the combination of Random Forest, AdaBoost and Linear correlation falls within the percentile of the defined threshold are kept in the dataset. Remaining features are dropped before further processing.

#### **Example**

```
# load dataset
from pycaret.datasets import get_data
juice = get_data('juice')

# init setup
from pycaret.classification import *
clf1 = setup(data = juice, target = 'Purchase', polynomial_features = True)
```

#### **Before**

![Dataframe view before polynomial features](<../../.gitbook/assets/image (99).png>)

#### **After**

![Dataframe view after polynomial features](<../../.gitbook/assets/image (301).png>)

Notice that new features were created from the existing feature space. To expand or compress polynomial feature space, you can use `polynomial_threshold` parameter which uses feature importance based on the combination of Random Forest, AdaBoost and Linear correlation to filter out the non important polynomial features. `polynomial_degree` _****_ can be used for defining number of degrees to be considered in feature creation.

### Trigonometry Features

Similar to [**Polynomial Features**](https://www.pycaret.org/polynomial-features)**,** PyCaret also allows creating new **trigonometry features** from the existing features. It is achieved using `trigonometry_features` parameter within [setup](https://www.pycaret.org/setup).

**PARAMETERS**

* **trigonometry\_features: bool, default = False**\
  When set to True, new features are created based on all trigonometric combinations that exist within the numeric features in a dataset to the degree defined in the polynomial\_degree par

#### Example

```
# load dataset
from pycaret.datasets import get_data
insurance = get_data('insurance')

# init setup
from pycaret.regression import *
reg1 = setup(data = insurance, target = 'charges', trigonometry_features = True)
```

#### Before

![Dataframe view before trigonometry features](<../../.gitbook/assets/image (250).png>)

#### After

![Dataframe view after trigonometry features](<../../.gitbook/assets/image (1).png>)

### Group Features

When dataset contains features that are related to each other in someway, for example: features recorded at some fixed time intervals, then new statistical features such as **mean**, **median**, **variance** and **standard deviation** for a group of such features can be created from existing features using `group_features` parameter within [setup](https://www.pycaret.org/setup).

#### **PARAMETERS**

* **group\_features: list or list of list, default = None**\
  When a dataset contains features that have related characteristics, the group\_features param can be used for statistical feature extraction. For example, if a dataset has numeric features that are related with each other (i.e ‘Col1’, ‘Col2’, ‘Col3’), a list containing the column names can be passed under group\_features to extract statistical information such as the mean, median, mode and standard deviation.
* **group\_names: list, default = None**\
  When group\_features is passed, a name of the group can be passed into the group\_names param as a list containing strings. The length of a group\_names list must equal to the length of group\_features. When the length doesn’t match or the name is not passed, new features are sequentially named such as group\_1, group\_2 etc.

#### **Example**

```
# load dataset
from pycaret.datasets import get_data
credit = get_data('credit')

# init setup
from pycaret.classification import *
clf1 = setup(data = credit, target = 'default', group_features = ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6'])
```

#### **Before**

![Dataframe before group features](<../../.gitbook/assets/image (343).png>)

#### **After**

![Dataframe after group features](<../../.gitbook/assets/image (508).png>)

### Bin Numeric Features

Feature binning is a method of turning continuous variables into categorical values using pre-defined number of **bins.** It is effective when a continuous feature has too many unique values or few extreme values outside the expected range. Such extreme values influence on the trained model, thereby affecting the prediction accuracy of the model. In PyCaret, continuous numeric features can be binned into intervals using `bin_numeric_features` parameter within [setup. ](https://www.pycaret.org/setup)PyCaret uses the _‘sturges’_ rule to determine the number of bins and also uses K-Means clustering to convert continuous numeric features into categorical features.

#### **PARAMETERS**

* **bin\_numeric\_features: list, default = None**\
  When a list of numeric features is passed they are transformed into categorical features using K-Means, where values in each bin have the same nearest center of a 1D k-means cluster. The number of clusters are determined based on the ‘sturges’ method. It is only optimal for gaussian data and underestimates the number of bins for large non-gaussian datasets.

#### **Example**

```
# load dataset
from pycaret.datasets import get_data
income = get_data('income')

# init setup
from pycaret.classification import *
clf1 = setup(data = income, target = 'income >50K', bin_numeric_features = ['age'])
```

#### Before

![Dataframe view before bin numeric bin features](<../../.gitbook/assets/image (160).png>)

#### After

![Dataframe view after numeric bin features](<../../.gitbook/assets/image (456).png>)

### Combine Rare Levels

Sometimes a dataset can have a categorical feature (or multiple categorical features) that has a very high number of levels (i.e. high cardinality features). If such feature (or features) are encoded into numeric values, then the resultant matrix is a **sparse matrix.** This not only makes experiment slow due to manifold increment in the number of features and hence the size of the dataset, but also introduces noise in the experiment. Sparse matrix can be avoided by combining the rare levels in the feature(or features) having high cardinality. This can be achieved in PyCaret using `combine_rare_levels` parameter within [setup](https://www.pycaret.org/setup).

#### **PARAMETERS**

* **combine\_rare\_levels: bool, default = False**\
  When set to True, all levels in categorical features below the threshold defined in rare\_level\_threshold param are combined together as a single level. There must be at least two levels under the threshold for this to take effect. rare\_level\_threshold represents the percentile distribution of level frequency. Generally, this technique is applied to limit a sparse matrix caused by high numbers of levels in categorical features.
* **rare\_level\_threshold: float, default = 0.1**\
  Percentile distribution below which rare categories are combined. Only comes into effect when combine\_rare\_levels is set to True.

#### **Example**

```
# load dataset
from pycaret.datasets import get_data
income = get_data('income')

# init setup
from pycaret.classification import *
clf1 = setup(data = income, target = 'income >50K', combine_rare_levels = True)
```

#### **Before**

![Dataframe view before combine rare levels](<../../.gitbook/assets/image (438).png>)

#### After

![Dataframe view after combine rare levels](<../../.gitbook/assets/image (524).png>)

#### Effect of combining rare levels

![](<../../.gitbook/assets/image (417).png>)

### Create Clusters

**Creating Clusters** using the existing features from the data is an unsupervised ML technique to engineer and create new features. It uses iterative approach to determine the number of clusters using combination of Calinski-Harabasz and Silhouette criterion. Each data point with the original features is assigned to a cluster. The assigned cluster label is then used as a **new feature** in predicting target variable. This can be achieved in PyCaret using `create_clusters` parameter within [setup](https://www.pycaret.org/setup).

#### PARAMETERS

* **create\_clusters: bool, default = False**\
  When set to True, an additional feature is created where each instance is assigned to a cluster. The number of clusters is determined using a combination of Calinski-Harabasz and Silhouette criterion.
* **cluster\_iter: int, default = 20**\
  Number of iterations used to create a cluster. Each iteration represents cluster size. Only comes into effect when create\_clusters param is set to True.

#### **Example**

```
# load dataset
from pycaret.datasets import get_data
insurance = get_data('insurance')

# init setup
from pycaret.regression import *
reg1 = setup(data = insurance, target = 'charges', create_clusters = True)
```

#### **Before**

![Dataframe view before create clusters](<../../.gitbook/assets/image (257).png>)

#### **After**

![Dataframe view after create clusters](<../../.gitbook/assets/image (75).png>)

