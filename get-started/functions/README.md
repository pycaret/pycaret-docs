---
description: This page lists all the functions of PyCaret
---

# 💡 Functions

## Select the tab :point\_down:

{% tabs %}
{% tab title="Initialize" %}
#### [setup](initialize.md#setting-up-environment)

This function initializes the experiment in PyCaret and prepares the transformation pipeline based on all the parameters passed in the function. The setup function must be called before executing any other function. It only requires two parameters: `data` and `target`. All the other parameters are optional. [Learn More.](initialize.md#setting-up-environment)
{% endtab %}

{% tab title="Train" %}
#### [compare\_models](train.md#compare\_models)

This function trains and evaluates the performance of all the models available in the model library using cross-validation. The output of this function is a scoring grid with average cross-validated scores. [Learn More.](train.md#compare\_models)



#### [create\_model](train.md#create\_model)&#x20;

This function trains and evaluates the performance of a given model using cross-validation. The output of this function is a scoring grid with cross-validated scores along with mean and standard deviation. [Learn More.](train.md#create\_model)
{% endtab %}

{% tab title="Optimize" %}
#### [tune\_model](optimize.md#tune\_model)

This function tunes the hyperparameters of a given model. The output of this function is a scoring grid with cross-validated scores of the best model. Search spaces are pre-defined with the flexibility to provide your own. The search algorithm can be random, bayesian, and a few others with the ability to scale on large clusters. [Learn More. ](optimize.md#tune\_model)



#### [ensemble\_model](optimize.md#ensemble\_model)

This function ensembles a given model. The output of this function is a scoring grid with cross-validated scores of the ensembled model. Two methods `Bagging` or `Boosting` can be used for ensembling. [Learn More](optimize.md#ensemble\_model).



#### [blend\_models](optimize.md#blend\_models)

This function trains a Soft Voting / Majority Rule classifier for given models in a list. The output of this function is a scoring grid with cross-validated scores of a Voting Classifier or Regressor. [Learn More.](optimize.md#blend\_models)



#### [stack\_models](optimize.md#stack\_models)

This function trains a meta-model over given models in a list. The output of this function is a scoring grid with cross-validated scores of a Stacking Classifier or Regressor. [Learn More.](optimize.md#stack\_models)



#### [optimize\_threshold](optimize.md#optimize\_threshold)

This function optimizes the probability threshold for a given model. It iterates over performance metrics at different probability thresholds and returns a plot with performance metrics on the y-axis and threshold on the x-axis. [Learn More.](optimize.md#optimize\_threshold)



#### [calibrate\_model](optimize.md#calibrate\_model)

This function calibrates the probability of a given model using isotonic or logistic regression. The output of this function is a scoring grid with cross-validated scores of calibrated classifier. [Learn More.](optimize.md#calibrate\_model)
{% endtab %}

{% tab title="Analyze" %}
#### [plot\_model](analyze.md#plot\_model)

This function analyzes the performance of a trained model on the hold-out set. It may require re-training the model in certain cases. [Learn More.](analyze.md#plot\_model)



#### [evaluate\_model](analyze.md#evaluate\_model)

This function uses `ipywidgets` to display a basic user interface for analyzing the performance of a trained model. [Learn More.](analyze.md#evaluate\_model)



#### [interpret\_model](analyze.md#interpret\_model)

This function analyzes the predictions generated from a trained model. Most plots in this function are implemented based on the SHAP (Shapley Additive exPlanations). [Learn More.](analyze.md#interpret\_model)



#### [dashboard](analyze.md#dashboard)

This function generates the interactive dashboard for a trained model. The dashboard is implemented using the ExplainerDashboard project. [Learn More.](analyze.md#dashboard)



#### [deep\_check](analyze.md#deep\_check)

This function runs a full suite check over a trained model using the deepchecks library. This function is in experimental mode. [Learn More](analyze.md#deep\_check).



#### [eda](analyze.md#eda)

This function generates automated Exploratory Data Analysis (EDA) using the AutoViz project. Fully interactive and exportable. [Learn More.](analyze.md#eda)



#### [check\_fairness](analyze.md#check\_fairness)

This function provides fairness-related metrics between different groups in the dataset for a given model. There are many approaches to evaluate fairness but this function uses the approach known as group fairness, which asks: which groups of individuals are at risk for experiencing harm. [Learn More.](analyze.md#check\_fairness)

####

#### [get\_leaderboard](analyze.md#get\_leaderboard)

This function returns the leaderboard of all models trained in the current setup. [Learn More.](analyze.md#get\_leaderboard)



#### [assign\_model](analyze.md#assign\_model)

This function assigns labels to the training dataset using the trained model. It is only available for unsupervised modules. [Learn More.](analyze.md#assign\_model)
{% endtab %}

{% tab title="Deploy" %}
#### [predict\_model](deploy.md#predict\_model)

This function generates the label using a trained model.  When unseen data is not passed, it predicts the label and score on the holdout set. [Learn More.](deploy.md#predict\_model)



#### [finalize\_model](deploy.md#finalize\_model)

This function refits a given model on the entire dataset. [Learn More. ](deploy.md#finalize\_model)

####

#### [save\_model](deploy.md#save\_model)

This function saves the ML pipeline as a pickle file for later use. [Learn More.](deploy.md#save\_model)

####

#### [load\_model](deploy.md#load\_model)

This function loads a previously saved pipeline. [Learn More.](deploy.md#load\_model)



#### [save\_config](deploy.md#save\_config)

This function saves all the global variables to a pickle file, allowing to later resume without rerunning the setup function. [Learn More.](deploy.md#save\_config)



#### [load\_config](deploy.md#load\_config)&#x20;

This function loads global variables from a pickle file into Python. [Learn More.](deploy.md#load\_config)



#### [deploy\_model](deploy.md#deploy\_model)

This function deploys the entire ML pipeline on the cloud. [Learn More.](deploy.md#deploy\_model)

####

#### [convert\_model](deploy.md#convert\_model)

This function transpiles the trained machine learning model's decision function in different programming languages such as Python, C, Java, Go, C#, etc. [Learn More.](deploy.md#convert\_model)

####

#### [create\_api](deploy.md#create\_api)

This function takes an input model and creates a POST API for inference. It only creates the API and doesn't run it automatically. To run the API, you must run the Python file using `!python`. [Learn More.](deploy.md#create\_api)

####

#### [create\_docker](deploy.md#create\_docker)

This function creates a Dockerfile and requirements.txt for deploying API. [Learn More.](deploy.md#create\_docker)

####

#### [create\_app](deploy.md#create\_app)

This function creates a basic gradio app for inference. [Learn More.](deploy.md#create\_app)
{% endtab %}

{% tab title="Others" %}
#### [pull](others.md#pull)

Returns the last printed scoring grid. Use `pull` function after any training function to get the metrics in `pandas.DataFrame`. [Learn More.](others.md#pull)



#### [models](others.md#models)

Return a table containing all the models available in the imported module of the model library. [Learn More.](others.md#models)

####

#### [get\_config](others.md#get\_config)

This function retrieves the global variables created by the [setup](initialize.md#setup) function.  [Learn More.](others.md#get\_config)

####

#### [set\_config](others.md#set\_config)

This function resets the global variables. [Learn More.](others.md#set\_config)



#### [get\_metrics](others.md#get\_metrics)

Returns the table of all available metrics used for cross-validation. [Learn More.](others.md#get\_metrics)



#### [add\_metric](others.md#add\_metric)

Adds a custom metric to the metric container for cross-validation. [Learn More.](others.md#add\_metric)



#### [remove\_metric](others.md#remove\_metric)

Removes a custom metric from the metric container. [Learn More.](others.md#remove\_metric)



#### [automl](others.md#automl)

This function returns the best model from all the models in the current setup. [Learn More.](others.md#automl)

####

#### [get\_logs](others.md#get\_logs)

Returns a table of experiment logs. Only works when log\_experiment = True when initializing the setup function. [Learn More.](others.md#get\_logs)

####

#### [get\__system\_logs_](others.md#get\_system\_logs)

Read and print logs.log file from the currently active directory. [Learn More.](others.md#get\_system\_logs)
{% endtab %}
{% endtabs %}

