# Data Preparation

### **Missing Values**

Datasets for various reasons may have missing values or empty records, often encoded as blanks or `NaN`. Most of the machine learning algorithms are not capable of dealing with missing or blank values. Removing samples with missing values is a basic strategy that is sometimes used but it comes with a cost of losing probable valuable data and the associated information or patterns. A better strategy is to impute the missing values. PyCaret by default imputes the missing value in the dataset by `mean` for numeric features and `constant` **** for categorical features. To change the imputation method, `numeric_imputation` and `categorical_imputation` parameters can be used within **** the setup.&#x20;

#### **PARAMETERS**

* **imputation\_type: string, default = 'simple'**\
  ****The type of imputation to use. It can be either `simple` or `iterative`
* **numeric\_imputation: string, default = ‘mean’**\
  Missing values in numeric features are imputed with the `mean` value of the feature in the training dataset. The other available option is `median` or `zero`.
* **categorical\_imputation: string, default = ‘constant’**\
  Missing values in categorical features are imputed with a constant `not_available` value. The other available option is `mode`.
* **iterative\_imputation\_iters: int = 5**\
  ****The number of iterations. Ignored when `imputation_type` is not `iterative`.
* **numeric\_iterative\_imputer: Union\[str, Any] = 'lightgbm'**\
  ****Estimator for iterative imputation of missing values in numeric features. Ignored when `imputation_type` is set to `simple`.
* **categorical\_iterative\_imputer: Union\[str, Any] = 'lightgbm**'\
  Estimator for iterative imputation of missing values in categorical features. Ignored when `imputation_type` is not `iterative`.

#### **Example**

```
# load dataset
from pycaret.datasets import get_data
hepatitis = get_data('hepatitis')

# init setup
from pycaret.classification import *
clf1 = setup(data = hepatitis, target = 'Class')
```

#### Before

![](<../../.gitbook/assets/image (483).png>)

#### **After**

![](<../../.gitbook/assets/image (211).png>)

#### Comparison of Simple imputer vs. Iterative imputer

![](<../../.gitbook/assets/image (413).png>)

To learn more about this experiment, you read [this article](https://www.linkedin.com/pulse/iterative-imputation-pycaret-22-antoni-baum/).&#x20;

{% hint style="info" %}
**NOTE:** No explicit parameters for imputing missing values are required in the `setup` function as PyCaret handles this task by default.&#x20;
{% endhint %}

### Data Types

Each feature in the dataset has an associated data type such as numeric, categorical, or Datetime. PyCaret’s inference algorithm automatically detects the data type of each feature. However, sometimes the data types inferred by PyCaret are incorrect. Ensuring data types are correct is important as several downstream processes depend on the data type of the features. One example could be that [Missing Values](data-preparation.md#missing-values) for numeric and categorical features in the dataset are imputed differently. To overwrite the inferred data types, `numeric_features`, `categorical_features` __ and `date_features` parameters can be used in the setup function. You can also use `ignore_features` to ignore certain features for model training.

**PARAMETERS**

* **numeric\_features: list of string, default = None**\
  If the inferred data types are not correct, `numeric_features` can be used to overwrite the inferred data types.&#x20;
* **categorical\_features: list of string, default = None**\
  If the inferred data types are not correct, `categorical_features` can be used to overwrite the inferred data types.&#x20;
* **date\_features: list of string, default = None**\
  If the data has a `Datetime` column that is not automatically inferred when running the setup, `date_features` can be used to force the data type. It can work with multiple date columns. Datetime related features are not used in modeling. Instead, feature extraction is performed and original `Datetime` columns are ignored during model training. If the `Datetime` column includes a timestamp, features related to time will also be extracted.
* **ignore\_features: list of string, default = None**\
  `ignore_features` can be used to ignore features during model training. It takes a list of strings with column names that are to be ignored.

#### **Example 1 - Categorical Features**

```
# load dataset
from pycaret.datasets import get_data
hepatitis = get_data('hepatitis')

# init setup
from pycaret.classification import *
clf1 = setup(data = hepatitis, target = 'Class', categorical_features = ['AGE'])
```

#### **Before**

![](<../../.gitbook/assets/image (353).png>)

#### **After**

![](<../../.gitbook/assets/image (373).png>)

#### Example 2 - Ignore Features

```
# load dataset
from pycaret.datasets import get_data
pokemon = get_data('pokemon')

# init setup
from pycaret.classification import *
clf1 = setup(data = pokemon, target = 'Legendary', ignore_features = ['#', 'Name'])
```

#### Before

![](<../../.gitbook/assets/image (151).png>)

#### After

![](<../../.gitbook/assets/image (255).png>)

### One-Hot Encoding

Categorical features in the dataset contain the label values (ordinal or nominal) rather than continuous numbers. The majority of the machine learning algorithms cannot directly deal with categorical features and they must be transformed into numeric values before training a model. The most common type of categorical encoding is One-Hot Encoding (also known as _dummy encoding_) where each categorical level becomes a separate feature in the dataset containing binary values (1 or 0).&#x20;

Since this is an imperative step to perform an ML experiment, PyCaret will transform all categorical features in the dataset using one-hot encoding. This is ideal for features having nominal categorical data i.e. data cannot be ordered. In other different scenarios, other methods of encoding must be used. For example, when the data is ordinal i.e. data has intrinsic levels, [Ordinal Encoding](data-preparation.md#ordinal-encoding) **** must be used. One-Hot Encoding works on all features that are either inferred as categorical or are forced as categorical using `categorical_features` in the setup function.&#x20;

#### **Example**

```
# load dataset
from pycaret.datasets import get_data
pokemon = get_data('pokemon')

# init setup
from pycaret.classification import *
clf1 = setup(data = pokemon, target = 'Legendary')
```

#### **Before**

![](<../../.gitbook/assets/image (50).png>)

#### **After**

![](<../../.gitbook/assets/image (129).png>)

{% hint style="info" %}
NOTE: There is no additional parameter need to be passed in the `setup` function for one-hot-encoding. By default, it is applied to all `categorical_features`, unless otherwise, you define `ordinal_encoding` or `high_cardinality_features` explicitly.
{% endhint %}

### Ordinal Encoding

When the categorical features in the dataset contain variables with intrinsic natural order such as _Low, Medium, and High_, these must be encoded differently than nominal variables (where there is no intrinsic order for e.g. Male or Female). This can be achieved using  the `ordinal_features` parameter in the setup function that accepts a dictionary with feature names and the levels in the increasing order from lowest to highest.

#### **PARAMETERS**

* **ordinal\_features: dictionary, default = None**\
  When the data contains ordinal features, they must be encoded differently using the `ordinal_features`. If the data has a categorical variable with values of `low`, `medium`, `high` and it is known that low < medium < high, then it can be passed as `ordinal_features = { ‘column_name’ : [‘low’, ‘medium’, ‘high’] }`. The list sequence must be in increasing order from lowest to highest.

#### **Example**

```
# load dataset
from pycaret.datasets import get_data
employee = get_data('employee')

# init setup
from pycaret.classification import *
clf1 = setup(data = employee, target = 'left', ordinal_features = {'salary' : ['low', 'medium', 'high']})
```

#### **Before**

![](<../../.gitbook/assets/image (395).png>)

#### **After**

![](<../../.gitbook/assets/image (526).png>)

### Cardinal Encoding

When categorical features in the dataset contain variables with many levels (also known as high cardinality features), then typical One-Hot Encoding leads to the creation of a very large number of new features, thereby making the experiment slow. Features with high cardinality can be handled using `high_cardinality_features` in the setup. It supports two methods for cardinal encoding  (1) Frequency and (2) Clustering. These methods can be defined in the setup function.

#### **PARAMETERS**

* **high\_cardinality\_features: string, default = None**\
  When the data contains features with high cardinality, they can be compressed into fewer levels by passing them as a list of column names with high cardinality. Features are compressed using the method defined in the `high_cardinality_method` parameter.
* **high\_cardinality\_method: string, default = ‘frequency’**\
  When the method is set to `frequency`, it will replace the original value of the feature with the frequency distribution and convert the feature into numeric. The other available method is `clustering` that clusters the statistical attributes of data and replaces the original value of the feature with the cluster label. The number of clusters is determined using a combination of Calinski-Harabasz and Silhouette criteria.

#### **Example**

```
# load dataset
from pycaret.datasets import get_data
income = get_data('income')

# init setup
from pycaret.classification import *
clf1 = setup(data = income, target = 'income >50K', high_cardinality_features = ['native-country'])
```

#### **Before**

![](<../../.gitbook/assets/image (315).png>)

#### **After**

![](<../../.gitbook/assets/image (247).png>)

### Handle Unknown Levels

When categorical features in the dataset contain unseen variables at the time of predictions, it may cause problems for the trained model as those levels were not present at the time of training. One way to deal with this is to reassign such levels. This can be achieved in PyCaret using `handle_unknown_categorical` _****_ and `unknown_categorical_method` parameters in the setup function.

#### **PARAMETERS**

* **handle\_unknown\_categorical: bool, default = True**\
  When set to `True`, unknown categorical levels are replaced by the most or least frequent level as learned in the training dataset.
* **unknown\_categorical\_method: string, default = ‘least\_frequent’**\
  This can be set to `least_frequent` or `most_frequent`.

#### Example

```
# load dataset
from pycaret.datasets import get_data
insurance = get_data('insurance')

# init setup
from pycaret.regression import *
reg1 = setup(data = insurance, target = 'charges', handle_unknown_categorical = True, unknown_categorical_method = 'most_frequent')
```

### Target Imbalance

When the training dataset has an unequal distribution of target class it can be fixed using the `fix_imbalance` parameter in the setup. When set to `True`, SMOTE (Synthetic Minority Over-sampling Technique) is used as a default method for resampling. The method for resampling can be changed using the `fix_imbalance_method` within the setup.&#x20;

#### **PARAMETERS**

* **fix\_imbalance: bool, default = False**\
  When set to `True`, the training dataset is resampled using the algorithm defined in `fix_imbalance_method` . When `None`, SMOTE is used by default.
* **fix\_imbalance\_method: obj, default = None**\
  This parameter accepts any algorithm from [imblearn](https://imbalanced-learn.org/stable/) that supports `fit_resample` method.

#### **Example**

```
# load dataset
from pycaret.datasets import get_data
credit = get_data('credit')

# init setup
from pycaret.classification import *
clf1 = setup(data = credit, target = 'default', fix_imbalance = True)
```

#### Before and After SMOTE&#x20;

![](<../../.gitbook/assets/image (547).png>)

### Remove Outliers

The **** `remove_outliers` **** function in PyCaret allows you to identify and remove outliers from the dataset before training the model. Outliers are identified through PCA linear dimensionality reduction using the Singular Value Decomposition technique. It can be achieved using `remove_outliers` parameter within [setup](https://www.pycaret.org/setup). The proportion of outliers are controlled through `outliers_threshold` parameter.

#### **PARAMETERS**

**remove\_outliers: bool, default = False**\
When set to True, outliers from the training data are removed using PCA linear dimensionality reduction using the Singular Value Decomposition technique.

**outliers\_threshold: float, default = 0.05**\
The percentage/proportion of outliers in the dataset can be defined using the outliers\_threshold param. By default, 0.05 is used which means 0.025 of the values on each side of the distribution’s tail are dropped from training data.

#### **Example**

```
# load dataset
from pycaret.datasets import get_data
insurance = get_data('insurance')

# init setup
from pycaret.regression import *
reg1 = setup(data = insurance, target = 'charges', remove_outliers = True)
```

![](<../../.gitbook/assets/image (399).png>)

#### Before and After removing outliers

![](<../../.gitbook/assets/image (29).png>)
