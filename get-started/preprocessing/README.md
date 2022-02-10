---
description: Data preprocessing and transformations in PyCaret
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

####

#### [Polynomial Features](feature-engineering.md#polynomial-features)

####

#### [Group Features](feature-engineering.md#group-features)

####

#### [Bin Numeric Features](feature-engineering.md#bin-numeric-features)

####

#### [Combine Rare Levels](feature-engineering.md#combine-rare-levels)

####

#### [Create Clusters](feature-engineering.md#create-clusters)
{% endtab %}

{% tab title="Feature Selection" %}
#### [Feature Selection](feature-selection.md#feature-selection)

####

#### [Remove Multicollinearity](feature-selection.md#remove-multicollinearity)

####

#### [Principal Component Analysis](feature-selection.md#principal-component-analysis)

####

#### [Ignore Low Variance](feature-selection.md#ignore-low-variance)

####
{% endtab %}

{% tab title="Other setup parameters" %}
#### [Required Parameters](other-setup-parameters.md#mandatory-parameters)

####

#### [Experiment Logging](other-setup-parameters.md#experiment-logging)

####

#### [Model Selection](other-setup-parameters.md#model-selection)

####

#### [Other Miscellaneous](other-setup-parameters.md#other-miscellaneous)&#x20;

####
{% endtab %}
{% endtabs %}
