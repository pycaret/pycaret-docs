---
description: >-
  This page lists all the data preprocessing and transformation parameters in
  the setup
---

# ⚙ Preprocessing

## Select the tab :point\_down:

{% tabs %}
{% tab title="Data Preparation" %}
#### [Missing Values](data-preparation.md#missing-values)

Datasets for various reasons may have missing values or empty records, often encoded as blanks or `NaN`. Most of the machine learning algorithms are not capable of dealing with the missing values. [Learn More.](data-preparation.md#missing-values)



#### [Data Types](data-preparation.md#data-types)

Each feature in the dataset has an associated data type such as numeric, categorical, or Datetime. PyCaret automatically detects the data type of each feature. [Learn More.](data-preparation.md#data-types)



#### [One-Hot Encoding](data-preparation.md#one-hot-encoding)

Categorical features in the dataset contain the label values (ordinal or nominal) rather than continuous numbers. Most of the machine learning algorithms are not capable of handling categorical data without encoding. [Learn More.](data-preparation.md#one-hot-encoding)



#### [Ordinal Encoding](data-preparation.md#ordinal-encoding)

When the categorical features in the dataset contain variables with intrinsic natural order such as _Low, Medium, and High_, these must be encoded differently than nominal variables (where there is no intrinsic order for e.g. Male or Female). [Learn More.](data-preparation.md#ordinal-encoding)



#### [Cardinal Encoding](data-preparation.md#cardinal-encoding)

When categorical features in the dataset contain variables with many levels (also known as high cardinality features), then typical One-Hot Encoding leads to the creation of a very large number of new features. [Learn More.](data-preparation.md#cardinal-encoding)



#### [Handle Unknown Levels](data-preparation.md#handle-unknown-levels)

When categorical features in the dataset contain unseen variables at the time of predictions, it may cause problems for the trained model as those levels were not present at the time of training. [Learn More.](data-preparation.md#handle-unknown-levels)



#### [Target Imbalance](data-preparation.md#target-imbalance)

When the training dataset has an unequal distribution of target class it can be fixed using the `fix_imbalance` parameter in the setup. [Learn More.](data-preparation.md#target-imbalance)



#### [Remove Outliers](data-preparation.md#remove-outliers)

The **** `remove_outliers` **** function in PyCaret allows you to identify and remove outliers from the dataset before training the model. [Learn More.](data-preparation.md#remove-outliers)
{% endtab %}

{% tab title="Scale and Transform" %}
#### [Normalize](scale-and-transform.md#normalize)

Normalization is a technique often applied as part of data preparation for machine learning. The goal of normalization is to rescale the values of numeric columns in the dataset without distorting the differences in the ranges of values. [Learn More.](scale-and-transform.md#normalize)



#### [Feature Transform](scale-and-transform.md#feature-transform)

While normalization rescales the data within new limits to reduce the impact of magnitude in the variance, Feature transformation is a more radical technique. Transformation changes the shape of the distribution. [Learn More.](scale-and-transform.md#feature-transform)



#### [Target Transform](scale-and-transform.md#target-transform)

Target Transformation is similar to feature transformation as it will change the shape of the distribution of the target variable instead of the features. [Learn More.](scale-and-transform.md#target-transform)
{% endtab %}

{% tab title="Feature Engineering" %}
#### [Feature Interaction](feature-engineering.md#feature-interaction)

It is often seen in machine learning experiments when two features combined through an arithmetic operation become more significant in explaining variances in the data than the same two features separately. [Learn More.](feature-engineering.md#feature-interaction)



#### [Polynomial Features](feature-engineering.md#polynomial-features)

In machine learning experiments the relationship between the dependent and independent variable is often assumed as linear, however, this is not always the case. Sometimes the relationship between dependent and independent variables is more complex. [Learn More.](feature-engineering.md#polynomial-features)



#### [Group Features](feature-engineering.md#group-features)

When a dataset contains features that are related to each other in some way, for example, features recorded at some fixed time intervals, then new statistical features such as mean, median, variance, and standard deviation **** for a group of such features. [Learn More.](feature-engineering.md#group-features)



#### [Bin Numeric Features](feature-engineering.md#bin-numeric-features)

Feature binning is a method of turning continuous variables into categorical values using the pre-defined number of bins. **** It is effective when a continuous feature has too many unique values or few extreme values outside the expected range. [Learn More.](feature-engineering.md#bin-numeric-features)



#### [Combine Rare Levels](feature-engineering.md#combine-rare-levels)

Sometimes a dataset can have a categorical feature (or multiple categorical features) that has a very high number of levels (i.e. high cardinality features). If such feature (or features) are encoded into numeric values, then the resultant matrix is a sparse matrix. **** [Learn More.](feature-engineering.md#combine-rare-levels)

****

#### [Create Clusters](feature-engineering.md#create-clusters)

Creating Clusters using the existing features from the data is an unsupervised ML technique to engineer and create new features. [Learn More.](feature-engineering.md#create-clusters)
{% endtab %}

{% tab title="Feature Selection" %}
#### [Feature Selection](feature-selection.md#feature-selection)

Feature Selection is a process **** used to select features in the dataset that contributes the most in predicting the target variable. Working with selected features instead of all the features reduces the risk of over-fitting, improves accuracy, and decreases the training time. [Learn More.](feature-selection.md#feature-selection)



#### [Remove Multicollinearity](feature-selection.md#remove-multicollinearity)

Multicollinearity (also called _collinearity_) is a phenomenon in which one feature variable in the dataset is highly linearly correlated with another feature variable in the same dataset. [Learn More.](feature-selection.md#remove-multicollinearity)



#### [Principal Component Analysis](feature-selection.md#principal-component-analysis)

Principal Component Analysis (PCA) is an unsupervised technique used in machine learning to reduce the dimensionality of the data. It does so by compressing the feature space. [Learn More.](feature-selection.md#principal-component-analysis)



#### [Ignore Low Variance](feature-selection.md#ignore-low-variance)

Sometimes a dataset may have a categorical feature with multiple levels, where the distribution of such levels is skewed and one level may dominate over other levels. [Learn More.](feature-selection.md#ignore-low-variance)
{% endtab %}

{% tab title="Other setup parameters" %}
#### [Required Parameters](other-setup-parameters.md#mandatory-parameters)

There are only two non-optional parameters in the setup function i.e. data and name of the target variable. [Learn More.](other-setup-parameters.md#mandatory-parameters)

####

#### [Experiment Logging](other-setup-parameters.md#experiment-logging)

PyCaret uses MLflow for experiment tracking. A parameter in the setup can be set to automatically track all the metrics, hyperparameters, and other model artifacts. [Learn More.](other-setup-parameters.md#experiment-logging)



#### [Model Selection](other-setup-parameters.md#model-selection)

Parameters in the setup can be used for setting parameters for the model selection process. These are not related to data preprocessing but can influence your model selection process. [Learn More.](other-setup-parameters.md#model-selection)



#### [Other Miscellaneous](other-setup-parameters.md#other-miscellaneous)&#x20;

Other miscellaneous parameters in the setup that are used for controlling experiment settings such as using GPU for training or setting verbosity of the experiment. [Learn More.](other-setup-parameters.md#other-miscellaneous)
{% endtab %}
{% endtabs %}



