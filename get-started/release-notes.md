---
description: All release notes of historic releases since 2.0
---

# ⚒ Release Notes

### **PyCaret 2.3.6**

**Release Date: January 11th, 2022 (NEW FEATURES, BUG FIXES)**

* Added new function `create_docker` ([https://github.com/pycaret/pycaret/pull/2005](https://github.com/pycaret/pycaret/pull/2005))
* Added new function `create_api` ([https://github.com/pycaret/pycaret/pull/2000](https://github.com/pycaret/pycaret/pull/2000))
* Added new function `check_fairness` ([https://github.com/pycaret/pycaret/pull/1997](https://github.com/pycaret/pycaret/pull/1997))
* Added new function `eda` ([https://github.com/pycaret/pycaret/pull/1983](https://github.com/pycaret/pycaret/pull/1983))
* Added new function `convert_model` ([https://github.com/pycaret/pycaret/pull/1959](https://github.com/pycaret/pycaret/pull/1959))
* Added an ability to pass kwargs to plots in `plot_model` ([https://github.com/pycaret/pycaret/pull/19400](https://github.com/pycaret/pycaret/pull/19400))
* Added `drift_report` functionality to `predict_model` ([https://github.com/pycaret/pycaret/pull/1935](https://github.com/pycaret/pycaret/pull/1935))
* Added new function `dashboard` ([https://github.com/pycaret/pycaret/pull/1925](https://github.com/pycaret/pycaret/pull/1925))
* Added `grid_interval` parameter to `optimize_threshold` - thanks to @wolfryu ([https://github.com/pycaret/pycaret/pull/1938](https://github.com/pycaret/pycaret/pull/1938))
* Made logging level configurable by environment variable ([https://github.com/pycaret/pycaret/pull/2026](https://github.com/pycaret/pycaret/pull/2026))
* Made the optional path in AWS configurable ([https://github.com/pycaret/pycaret/pull/2045](https://github.com/pycaret/pycaret/pull/2045))
* Fixed TSNE plot with PCA ([https://github.com/pycaret/pycaret/pull/2032](https://github.com/pycaret/pycaret/pull/2032))
* Fixed rendering of streamlit plots ([https://github.com/pycaret/pycaret/pull/2008](https://github.com/pycaret/pycaret/pull/2008))
* Fixed class names in `tree` plot - thanks to @yamasakih ([https://github.com/pycaret/pycaret/pull/1982](https://github.com/pycaret/pycaret/pull/1982))
* Fixed NearZeroVariance preprocessor not being configurable - thanks to @Flyfoxs ([https://github.com/pycaret/pycaret/pull/1952](https://github.com/pycaret/pycaret/pull/1952))
* Removed duplicated code - thanks to @Flyfoxs ([https://github.com/pycaret/pycaret/pull/1882](https://github.com/pycaret/pycaret/pull/1882))
* Documentation improvements - thanks to @harsh204016, @khrapovs ([https://github.com/pycaret/pycaret/pull/1931/files](https://github.com/pycaret/pycaret/pull/1931/files), [https://github.com/pycaret/pycaret/pull/1956](https://github.com/pycaret/pycaret/pull/1956), [https://github.com/pycaret/pycaret/pull/1946](https://github.com/pycaret/pycaret/pull/1946), [https://github.com/pycaret/pycaret/pull/1949](https://github.com/pycaret/pycaret/pull/1949))
* Pinned `pyyaml<6.0.0` to fix issues with Google Colab

**Release: PyCaret 2.3.6 | Release Date: January 11th, 2022 (NEW FEATURES, BUG FIXES)**

### **PyCaret 2.3.5**&#x20;

**Release Date: November 19th, 2021 (NEW FEATURES, BUG FIXES)**

* Fixed an issue where `Fix_multicollinearity` would fail if the target was a float ([https://github.com/pycaret/pycaret/pull/1640](https://github.com/pycaret/pycaret/pull/1640))
* MLFlow runs are now nested - thanks to @jfagn ([https://github.com/pycaret/pycaret/pull/1660](https://github.com/pycaret/pycaret/pull/1660))
* Fixed a typo in REG102 tutorial - thanks to @bobo-jamson ([https://github.com/pycaret/pycaret/pull/1684](https://github.com/pycaret/pycaret/pull/1684))
* Fixed `interpret_model` not always respecting `save_path` ([https://github.com/pycaret/pycaret/pull/1707](https://github.com/pycaret/pycaret/pull/1707))
* Fixed certain plots not being logged by MLFlow ([https://github.com/pycaret/pycaret/pull/1769](https://github.com/pycaret/pycaret/pull/1769))
* Added dummy models to set a baseline in `compare_models` - thanks to @reza1615 ([https://github.com/pycaret/pycaret/pull/1739](https://github.com/pycaret/pycaret/pull/1739))
* Improved error message if a column specified in `ignore_features` doesn't exist in the dataset - thanks to @reza1615 ([https://github.com/pycaret/pycaret/pull/1793](https://github.com/pycaret/pycaret/pull/1793))
* Added an ability to set a custom probability threshold for binary classification through the `probability_threshold` argument in various methods ([https://github.com/pycaret/pycaret/pull/1858](https://github.com/pycaret/pycaret/pull/1858))
* Separated internal CV from validation CV for `stack_models` and `calibrate_models` ([https://github.com/pycaret/pycaret/pull/1849](https://github.com/pycaret/pycaret/pull/1849), [https://github.com/pycaret/pycaret/pull/1858](https://github.com/pycaret/pycaret/pull/1858))
* A `RuntimeError` will now be raised if an incorrect version of `scikit-learn` is installed ([https://github.com/pycaret/pycaret/pull/1870](https://github.com/pycaret/pycaret/pull/1870))
* Improved readme, documentation and repository structure
* Unpinned `numba` ([https://github.com/pycaret/pycaret/pull/173](https://github.com/pycaret/pycaret/pull/173))

### **PyCaret 2.3.4**

#### **Release Date: September 23rd, 2021 (NEW FEATURES, BUG FIXES)**

* Added `get_leaderboard` function for classification and regression modules
* It is now possible to specify the plot save path with the save argument of `plot_model` and `interpret_model` - thanks to @bhanuteja2001 ([https://github.com/pycaret/pycaret/pull/1537](https://github.com/pycaret/pycaret/pull/1537))
* Fixed `interpret_model` affecting `plot_model` behavior - thanks to @naujgf ([https://github.com/pycaret/pycaret/pull/1600](https://github.com/pycaret/pycaret/pull/1600))
* Fixed issues with conda builds - thanks to @melonhead901 ([https://github.com/pycaret/pycaret/pull/1479](https://github.com/pycaret/pycaret/pull/1479))
* Documentation improvements - thanks to @caron14 and @harsh204016 ([https://github.com/pycaret/pycaret/pull/1499](https://github.com/pycaret/pycaret/pull/1499), [https://github.com/pycaret/pycaret/pull/1502](https://github.com/pycaret/pycaret/pull/1502))
* Fixed `blend_models` and `stack_models` throwing an exception when using custom estimators ([https://github.com/pycaret/pycaret/pull/1500](https://github.com/pycaret/pycaret/pull/1500))
* Fixed a "Target Missing" issue with **** "Remove Multicolinearity" option ([https://github.com/pycaret/pycaret/pull/1508)](https://github.com/pycaret/pycaret/pull/1508\))
* `errors="ignore"` parameter for `compare_models` now correctly ignores errors during full fit ([https://github.com/pycaret/pycaret/pull/1510](https://github.com/pycaret/pycaret/pull/1510))
* Fixed certain data types being incorrectly encoded as int64 during setup ([https://github.com/pycaret/pycaret/pull/1515](https://github.com/pycaret/pycaret/pull/1515))
* Pinned `numba<0.54` ([https://github.com/pycaret/pycaret/pull/1530](https://github.com/pycaret/pycaret/pull/1530))

### **PyCaret 2.3.3**

#### **Release Date: July 24th, 2021 (NEW FEATURES, BUG FIXES)**

* Fixed issues with `[full]` install by pinning `interpret<=0.2.4`
* Added support for S3 folder path in `deploy_model()` with AWS
* Enabled experimental Optuna `TPESampler` options to improve convergence (in `tune_model()`)

### **PyCaret 2.3.2**

#### **Release Date: July 7th, 2021 (NEW FEATURES, BUG FIXES)**

* Implemented PDP, MSA and PFI plots in `interpret_model` - thanks to @IncubatorShokuhou ([https://github.com/pycaret/pycaret/pull/1415](https://github.com/pycaret/pycaret/pull/1415))
* Implemented Kolmogorov-Smirnov (KS) plot in `plot_model` under `pycaret.classification` module
* Fixed a typo "RVF" to "RBF" - thanks to @baturayo ([https://github.com/pycaret/pycaret/pull/1220](https://github.com/pycaret/pycaret/pull/1220))
* Readme & license updates and improvements
* Fixed `remove_multicollinearity` considering categorical features
* Fixed keyword issues with PyCaret's cuML wrappers
* Improved performance of iterative imputation
* Fixed `gain` and `lift` plots taking wrong arguments, creating misleading plots
* `interpret_model` on LightGBM will now show a beeswarm plot
* Multiple improvements to exception handling and documentation in `pycaret.persistence` ([https://github.com/pycaret/pycaret/pull/1324](https://github.com/pycaret/pycaret/pull/1324))
* `remove_perfect_collinearity` option will now be show in the `setup()` summary - thanks to @mjkanji ([https://github.com/pycaret/pycaret/pull/1342](https://github.com/pycaret/pycaret/pull/1342))
* Fixed `IterativeImputer` setting wrong float precision
* Fixed custom grids in `tune_model` raising an exception when composed of lists
* Improved documentation in `pycaret.clustering` - thanks to @susmitpy ([https://github.com/pycaret/pycaret/pull/1372](https://github.com/pycaret/pycaret/pull/1372))
* Added support for LightGBM CUDA version - thanks to @IncubatorShokuhou ([https://github.com/pycaret/pycaret/pull/1396](https://github.com/pycaret/pycaret/pull/1396))
* Exposed `address` in `get_data` for alternative data sources - thanks to @IncubatorShokuhou ([https://github.com/pycaret/pycaret/pull/1416](https://github.com/pycaret/pycaret/pull/1416))

### **PyCaret 2.3.1**

#### **Release Date: April 28, 2021 (SEVERAL BUGS FIXED)**

* Fixed an exception with missing variables (display\_container etc.) during load\_config()
* Fixed exceptions when using Ridge and RF estimators with cuML (GPU mode)
* Fixed PyCaret's cuML wrappers not being pickleable
* Added an extra check to get\_all\_object\_vars\_and\_properties internal method, fixing exceptions with certain estimators
* save\_model() now supports kwargs, which will be passed to joblib.dump()
* Fixed an issue with load\_model() from AWS (duplicate .pkl extension) - thanks to markgrujic ([https://github.com/pycaret/pycaret/pull/1128](https://github.com/pycaret/pycaret/pull/1128))
* Fixed a typo in documentation - thanks to koorukuroo ([https://github.com/pycaret/pycaret/pull/1149](https://github.com/pycaret/pycaret/pull/1149))
* Optimized Fix\_multicollinearity transformer, drastically reducing the size of saved pipeline
* interpret\_model() now supports data passed as an argument - thanks to jbechtel ([https://github.com/pycaret/pycaret/pull/1184](https://github.com/pycaret/pycaret/pull/1184))
* Removed `infer_signature` from MLflow logging when `log_experiment=True`.
* Fixed a rare issue where binary\_multiclass\_score\_func was not pickleable
* Fixed edge case exceptions in feature selection
* Fixed an exception with `finalize_model` when using GroupKFold CV
* Pinned `mlxtend>=0.17.0`, `imbalanced-learn==0.7.0`, and `gensim<4.0.0`

### PyCaret 2.3.0&#x20;

#### Release Date: February 21, 2021

* **Modules Impacted:** `pycaret.classification` `pycaret.regression` `pycaret.clustering` `pycaret.anomaly` `pycaret.arules`

#### Summary of Changes

* Added new interactive residual plots in `pycaret.regression` module. You can now generate interactive residual plots by using `residuals_interactive` in the `plot_model` function.
* Added plot rendering support for streamlit applications. A new parameter `display_format` is added in the `plot_model` function. To render plot in streamlit app, set this to `streamlit`.
* Revamped Boruta feature selection algorithm. (give it a try!).
* `tune_model` in `pycaret.classification` and `pycaret.regression` is now compatible with custom models.
* Added low\_memory and max\_len support to association rules module ([https://github.com/pycaret/pycaret/pull/1008](https://github.com/pycaret/pycaret/pull/1008)).
* Increased robustness of DataFrame checks ([https://github.com/pycaret/pycaret/pull/1005](https://github.com/pycaret/pycaret/pull/1005)).
* Improved loading of models from AWS ([https://github.com/pycaret/pycaret/pull/1005](https://github.com/pycaret/pycaret/pull/1005)).
* Catboost and XGBoost are now optional dependencies. They are not automatically installed with default slim installation. To install optional dependencies use `pip install pycaret[full]`.
* Added `raw_score` argument in the `predict_model` function for `pycaret.classification` module. When set to True, scores for each class will be returned separately.
* PyCaret now returns base scikit-learn objects, whenever possible.
* When `handle_unknown_categorical` is set to False in the `setup` function, an exception will be raised during prediction if the data contains unknown levels in categorical features.
* `predict_model` for multiclass classification now returns labels as an integer.
* Fixed an edge case where an IndexError would be raised in `pycaret.clustering` and `pycaret.anomaly`.
* Fixed text formatting for certain plots in `pycaret.classification` and `pycaret.regression`.
* If a `logs.log` file cannot be created when `setup` is initialized, no exception will be raised now (support for more configurable logging to come in future).
* User added metrics will not raise exceptions now and instead return 0.0.
* Compatibility with tune-sklearn>=0.2.0.
* Fixed an edge case for dropping NaNs in target column.
* Fixed stacked models not being tuned correctly.
* Fixed an exception with KFold when fold\_shuffle=False.

### **PyCaret 2.2.3**

#### **Release Date: December 22, 2020 (SEVERAL BUGS FIX | CRITICAL COMPATIBILITY FIX)**

* Fixed exceptions with the `predict_model` function when data columns had non-string characters.
* Fixed a rare exception with the `remove_multicollinearity` parameter in the `setup` function\`.
* Improved performance and robustness of conversion of date features to categoricals.
* Fixed an exception with the `models` function when the `type` parameter was passed.
* The data frame displayed after setup can now be accessed with the `pull` function.
* Fixed an exception with save\_config
* Fixed a rare case where the target column would be treated as an ID column and thus dropped.
* SHAP plots can now be saved (pass save parameter as True)
* **| CRITICAL |** Compatibility broke for catboost, pyod (other impacts unknown as of now) with sklearn=0.24 (released on Dec 22, 2020). A temporary fix is requiring 0.23.2 specifically in the `requirements.txt`.

### **PyCaret 2.2.2**

#### **Release Date: November 25, 2020 (SEVERAL BUGS FIX)**

* Fixed an issue with the `optimize_threshold` function the `pycaret.classification` module. It now returns a float instead of an array.
* Fixed issue with the `predict_model` function. It now uses original data frame to append the predictions. As such any extra columns given at the time of inference are not removed when returning the predictions. Instead they are internally ignored at the time of predictions.
* Fixed edge case exceptions for the `create_model` function in `pycaret.clustering`.
* Fixed exceptions when column names are not string.
* Fixed exceptions in `pycaret.regression` when `transform_target` is True in the `setup` function.
* Fixed an exception in the `models` function if the `type` parameter is specified.

### **PyCaret 2.2.1**

#### **Release Date: November 09, 2020 (SEVERAL BUGS FIX)**

Post-release `2.2`, the following issues have been fixed:

* Fixed `plot_model = 'tree'` exceptions.
* Fixed issue with `predict_model` causing errors with non-contiguous indices.
* Fixed issue with `remove_outliers` parameter in the `setup` function. It was introducing extra columns in training data. The issue has been fixed now.
* Fixed issue with `plot_model` in `pycaret.clustering` causing errors with non-contiguous indices.
* Fixed an exception when the model was saved or logged when `imputation_type` is set to 'iterative' in the `setup` function.
* `compare_models` now prints intermediate output when `html=False`.
* Metrics in `pycaret.classification` for binary classification are now calculated with `average='binary'`. Before they were a weighted average of positive and negative class, now they are just calculated for positive class. For multiclass classification `average='weighted'`.
* `optimize_threshold` now returns optimized probability threshold value as numpy object.
* Fixed issue with certain exceptions in `compare_models`.
* Added `profile_kwargs` argument in the `setup` function to pass keyword arguments to Pandas Profiler.
* `plot_model`, `interpret_model`, and `evaluate_model` now accepts a new parameter `use_train_data` which when set to True, generates plot on train data instead of test data.

### PyCaret 2.2**.0**

#### Release Date: October 28, 2020

#### Summary of Changes

* **Modules Impacted:** `pycaret.classification` `pycaret.regression` `pycaret.clustering` `pycaret.anomaly`
* **Separate Train and Test Set:** New parameter `test_data` has been added in the `setup` function of `pycaret.classification` and `pycaret.regression`. When a DataFrame is passed into the `test_data`, it is used as a holdout set and the `train_size` parameter is ignored. `test_data` must be labeled and the shape of `test_data` must match with the shape of `data`.
* **Disable Default Preprocessing:** A new parameter `preprocess` has been added into the `setup` function. When `preprocess` is set to `False`, no transformations are applied except for `train_test_split` and custom transformations passed in the `custom_pipeline` param. Data must be ready for modeling (no missing values, no dates, categorical data encoding) when preprocess is set to False.
* **Custom Metrics:** New functions `get_metric`, `add_metric` and `remove_metric` is now added in `pycaret.classification`, `pycaret.regression`, and `pycaret.clustering`, that can be used to add / remove metrics used in model evaluation.
* **Custom Transformations:** A new parameter `custom_pipeline` has been added into the `setup` function. It takes a tuple of `(str, transformer)` or a list of tuples. When passed, it will append the custom transformers in the preprocessing pipeline and are applied on each CV fold separately and on the final fit. All the custom transformations are applied after `train_test_split` and before pycaret's internal transformations.
* **GPU enabled Training:** To use GPU for training `use_gpu` parameter in the `setup` function can be set to `True` or `force`. When set to True, it will use GPU with algorithms that support it and fall back on CPU for remaining. When set to `force` it will only use GPU-enabled algorithms and raise exceptions if they are unavailable for use. The following algorithms are supported on GPU:
  * Extreme Gradient Boosting `pycaret.classification` `pycaret.regression`
  * LightGBM `pycaret.classification` `pycaret.regression`
  * CatBoost `pycaret.classification` `pycaret.regression`
  * Random Forest `pycaret.classification` `pycaret.regression`
  * K-Nearest Neighbors `pycaret.classification` `pycaret.regression`
  * Support Vector Machine `pycaret.classification` `pycaret.regression`
  * Logistic Regression `pycaret.classification`
  * Ridge Classifier `pycaret.classification`
  * Linear Regression `pycaret.regression`
  * Lasso Regression `pycaret.regression`
  * Ridge Regression `pycaret.regression`
  * Elastic Net (Regression) `pycaret.regression`
  * K-Means `pycaret.clustering`
  * Density-Based Spatial Clustering `pycaret.clustering`
*   **Hyperparameter Tuning:** New methods for hyperparameter tuning has been added in the `tune_model` function for `pycaret.classification` and `pycaret.regression`. New parameter `search_library` and `search_algorithm` in the `tune_model` function is added. `search_library` can be `scikit-learn`, `scikit-optimize`, `tune-sklearn`, and `optuna`. The `search_algorithm` param can take the following values based on its `search_library`:

    * scikit-learn: `random` `grid`
    * scikit-optimize: `bayesian`
    * tune-sklearn: `random` `grid` `bayesian` `hyperopt` `bohb`
    * optuna: `random` `tpe`

    Except for `scikit-learn`, all the other search libraries are not hard dependencies of pycaret and must be installed separately.
* **Early Stopping:** Early stopping now supported for hyperparameter tuning. A new parameter `early_stopping` is added in the `tune_model` function for `pycaret.classification` and `pycaret.regression`. It is ignored when `search_library` is `scikit-learn`, or if the estimator doesn't have a 'partial\_fit' attribute. It can be either an object accepted by the search library or one of the following:
  * `asha` for Asynchronous Successive Halving Algorithm
  * `hyperband` for Hyperband
  * `median` for median stopping rule
  * When `False` or `None`, early stopping will not be used.
* **Iterative Imputation:** Iterative imputation type for numeric and categorical missing values is now implemented. New parameters `imputation_type`, `iterative_imptutation_iters`, `categorical_iterative_imputer`, and `numeric_iterative_imputer` added in the `setup` function. Read the blog post for more details: [https://www.linkedin.com/pulse/iterative-imputation-pycaret-22-antoni-baum/?trackingId=Shg1zF%2F%2FR5BE7XFpzfTHkA%3D%3D](https://www.linkedin.com/pulse/iterative-imputation-pycaret-22-antoni-baum/?trackingId=Shg1zF%2F%2FR5BE7XFpzfTHkA%3D%3D)
* **New Plots:** Following new plots have been added:
  * lift `pycaret.classification`
  * gain `pycaret.classification`
  * tree `pycaret.classification` `pycaret.regression`
  * feature\_all `pycaret.classification` `pycaret.regression`
* **CatBoost Compatibility:** `CatBoostClassifier` and `CatBoostRegressor` is now compatible with `plot_model`. It requires `catboost>=0.23.2`.
* **Log Plots in MLFlow Server:** You can now log any plot in the `MLFlow` tracking server that is available in the `plot_model` function. To log specific plots, pass a list containing plot IDs in the `log_plots` parameter. Check the documentation of the `plot_model` to see all available plots.
* **Data Split Stratification:** A new parameter `data_split_stratify` is added in the `setup` function of `pycaret.classification` and `pycaret.regression`. It controls stratification during `train_test_split`. When set to True, will stratify by target column. To stratify on any other columns, pass a list of column names.
* **Fold Strategy:** A new parameter `fold_strategy` is added in the `setup` function for `pycaret.classification` and `pycaret.regression`. By default, it is 'stratifiedkfold' for `pycaret.classification` and 'kfold' for `pycaret.regression`. Possible values are:
  * `kfold` for KFold CV;
  * `stratifiedkfold` for Stratified KFold CV;
  * `groupkfold` for Group KFold CV;
  * `timeseries` for TimeSeriesSplit CV; or
  * a custom CV generator object compatible with scikit-learn.
* **Global Fold Parameter:** A new parameter `fold` has been added in the `setup` function for `pycaret.classification` and `pycaret.regression`. It controls the number of folds to be used in cross validation. This is a global setting that can be over-written at function level by using `fold` parameter within each function. Ignored when `fold_strategy` is a custom object.
* **Fold Groups:** Optional Group labels when `fold_strategy` is `groupkfold`. It takes an array with shape (n\_samples, ) where n\_samples is the number of rows in the training dataset. When a string is passed, it is interpreted as the column name in the dataset containing the group label.
* **Transformation Pipeline:** All transformations are now applied after `train_test_split`.
* **Data Type Handling:** All data types handling internally has been changed from `int64` and `float64` to `int32` and `float32` respectively in order to improve memory usage and performance, as well as for better compatibility with GPU-based algorithms.
* **AutoML Behavior Change:** `automl` function in `pycaret.classification` and `pycaret.regression` is no more re-fitting the model on the entire dataset. As such, if the model needs to be fitted on the entire dataset including the holdout set, `finalize_model` must be explicitly used.
* **Default Tuning Grid:** Default hyperparameter tuning grid for `RandomForest`, `XGBoost`, `CatBoost`, and `LightGBM` has been amended to remove extreme values for `max_depth` and other training intense parameters to speed up the tuning process.
* **Random Forest Default Values:** Default value of `n_estimators` for `RandomForestClassifier` and `RandomForestRegressor` has been changed from `10` to `100` to make it consistent with the default behavior of `scikit-learn`.
* **AUC for Multiclass Classification:** AUC for Multiclass target is now available in the metric evaluation.
* **Google Colab Display:** All output printed on screen (information grid, score grids) is now format compatible with Google Colab resulting in semantic improvements.
* **Sampling Parameter Removed:** `sampling` parameter is now removed from the `setup` function of `pycaret.classification` and `pycaret.regression`.
* **Type Hinting:** In order to make both the usage and development easier, type hints have been added to all updated pycaret functions, in accordance with best practices. Users can leverage those by using an IDE with support for type hints.
* **Documentation:** All Modules documentation on the website is now retired. Updated documentation is available here: [https://pycaret.readthedocs.io/en/latest/](https://pycaret.readthedocs.io/en/latest/)

#### Function Level Changes

#### New Functions Introduced in PyCaret 2.2

* **get\_metrics:** Returns table of available metrics used for CV. `pycaret.classification` `pycaret.regression` `pycaret.clustering`
* **add\_metric:** Adds a custom metric for model evaluation. `pycaret.classification` `pycaret.regression` `pycaret.clustering`
* **remove\_metric:** Remove custom metrics. `pycaret.classification` `pycaret.regression` `pycaret.clustering`
* **save\_config:** save all global variables to a pickle file, allowing to later resume without rerunning the `setup` function. `pycaret.classification` `pycaret.regression` `pycaret.clustering` `pycaret.anomaly`
* **load\_config:** Load global variables from pickle file into Python environment. `pycaret.classification` `pycaret.regression` `pycaret.clustering` `pycaret.anomaly`

#### setup

`pycaret.classification` `pycaret.regression` `pycaret.clustering` `pycaret.anomaly`

Following new parameters have been added:

* **test\_data: pandas.DataFrame, default = None** If not None, test\_data is used as a hold-out set, and the `train_size` parameter is ignored. test\_data must be labeled and the shape of data and test\_data must match.
* **preprocess: bool, default = True** When set to False, no transformations are applied except for `train_test_split` and custom transformations passed in `custom_pipeline` param. Data must be ready for modeling (no missing values, no dates, categorical data encoding) when `preprocess` is set to False.
* **imputation\_type: str, default = 'simple'** The type of imputation to use. Can be either 'simple' or 'iterative'.
* **iterative\_imputation\_iters: int, default = 5** The number of iterations. Ignored when `imputation_type` is not 'iterative'.
* **categorical\_iterative\_imputer: str, default = 'lightgbm'** Estimator for iterative imputation of missing values in categorical features. Ignored when `imputation_type` is not 'iterative'.
* **numeric\_iterative\_imputer: str, default = 'lightgbm'** Estimator for iterative imputation of missing values in numeric features. Ignored when `imputation_type` is set to 'simple'.
* **data\_split\_stratify: bool or list, default = False** Controls stratification during 'train\_test\_split'. When set to True, will stratify by target column. To stratify on any other columns, pass a list of column names. Ignored when `data_split_shuffle` is False.
* **fold\_strategy: str or sklearn CV generator object, default = 'stratifiedkfold' / 'kfold'** Choice of cross validation strategy. Possible values are:
  * 'kfold'
  * 'stratifiedkfold'
  * 'groupkfold'
  * 'timeseries'
  * a custom CV generator object compatible with scikit-learn.
* **fold: int, default = 10** The number of folds to be used in cross-validation. Must be at least 2. This is a global setting that can be over-written at the function level by using the `fold` parameter. Ignored when `fold_strategy` is a custom object.
* **fold\_shuffle: bool, default = False** Controls the shuffle parameter of CV. Only applicable when `fold_strategy` is 'kfold' or 'stratifiedkfold'. Ignored when `fold_strategy` is a custom object.
* **fold\_groups: str or array-like, with shape (n\_samples,), default = None** Optional group labels when 'GroupKFold' is used for the cross-validation. It takes an array with shape (n\_samples, ) where n\_samples is the number of rows in the training dataset. When a string is passed, it is interpreted as the column name in the dataset containing group labels.
* **use\_gpu: str or bool, default = False** When set to 'force', will try to use GPU with all algorithms that support it, and raise exceptions if they are unavailable. When set to True, will use GPU with algorithms that support it, and fall back to CPU if they are unavailable. When False, all algorithms are trained using CPU only.
* _custom\_pipeline: transformer or list of transformers or tuple, default = None_\* When passed, will append the custom transformers in the preprocessing pipeline and are applied on each CV fold separately and on the final fit. All the custom transformations are applied after 'train\_test\_split' and before pycaret's internal transformations.

#### compare\_models

`pycaret.classification` `pycaret.regression`

Following new parameters have been added:

* **cross\_validation: bool = True** When set to False, metrics are evaluated on holdout set. `fold` param is ignored when cross\_validation is set to False.
* **errors: str = "ignore"** When set to 'ignore', will skip the model with exceptions and continue. If 'raise', will stop the function when exceptions are raised.
* **fit\_kwargs: Optional\[dict] = None** Dictionary of arguments passed to the fit method of the model.
* **groups: Optional\[Union\[str, Any]] = None** Optional group labels when 'GroupKFold' is used for the cross-validation. It takes an array with shape (n\_samples, ) where n\_samples is the number of rows in the training dataset. When a string is passed, it is interpreted as the column name in the dataset containing group labels.

#### create\_model

`pycaret.classification` `pycaret.regression`

Following new parameters have been added:

* **cross\_validation: bool = True** When set to False, metrics are evaluated on holdout set. `fold` param is ignored when cross\_validation is set to False.
* **fit\_kwargs: Optional\[dict] = None** Dictionary of arguments passed to the fit method of the model.
* **groups: Optional\[Union\[str, Any]] = None** Optional group labels when 'GroupKFold' is used for the cross-validation. It takes an array with shape (n\_samples, ) where n\_samples is the number of rows in the training dataset. When a string is passed, it is interpreted as the column name in the dataset containing group labels.

Following parameters have been removed:

* **ensemble** - Deprecated - use `ensemble_model` function directly.
* **method** - Deprecated - use `ensemble_model` function directly.
* **system** - Moved to private API.

#### tune\_model

`pycaret.classification` `pycaret.regression`

Following new parameters have been added:

*   **search\_library: str, default = 'scikit-learn'** The search library used for tuning hyperparameters. Possible values:

    'scikit-learn' - default, requires no further installation [https://github.com/scikit-learn/scikit-learn](https://github.com/scikit-learn/scikit-learn)

    'scikit-optimize' - `pip install scikit-optimize` [https://scikit-optimize.github.io/stable/](https://scikit-optimize.github.io/stable/)

    'tune-sklearn' - `pip install tune-sklearn ray[tune]`[ https://github.com/ray-project/tune-sklearn](https://github.com/ray-project/tune-sklearn)

    'optuna' - `pip install optuna` [https://optuna.org/](https://optuna.org)
*   **search\_algorithm: str, default = None** The search algorithm depends on the `search_library` parameter. Some search algorithms require additional libraries to be installed. When None, will use the search library-specific default algorithm.

    `scikit-learn` possible values: - random (default) - grid

    `scikit-optimize` possible values: - bayesian (default)

    `tune-sklearn` possible values: - random (default) - grid - bayesian `pip install scikit-optimize` - hyperopt `pip install hyperopt` - bohb `pip install hpbandster ConfigSpace`

    `optuna` possible values: - tpe (default) - random
* **early\_stopping: bool or str or object, default = False** Use early stopping to stop fitting to a hyperparameter configuration if it performs poorly. Ignored when `search_library` is scikit-learn, or if the estimator does not have 'partial\_fit' attribute. If False or None, early stopping will not be used. Can be either an object accepted by the search library or one of the following:
  * 'asha' for Asynchronous Successive Halving Algorithm
  * 'hyperband' for Hyperband
  * 'median' for Median Stopping Rule
  * If False or None, early stopping will not be used.
* **early\_stopping\_max\_iters: int, default = 10** The maximum number of epochs to run for each sampled configuration. Ignored if `early_stopping` is False or None.
* **fit\_kwargs: Optional\[dict] = None** Dictionary of arguments passed to the fit method of the model.
* **groups: Optional\[Union\[str, Any]] = None** Optional group labels when 'GroupKFold' is used for the cross-validation. It takes an array with shape (n\_samples, ) where n\_samples is the number of rows in the training dataset. When a string is passed, it is interpreted as the column name in the dataset containing group labels.
* **return\_tuner: bool, default = False** When set to True, will return a tuple of (model, tuner\_object).
* **tuner\_verbose: bool or in, default = True** If True or above 0, will print messages from the tuner. Higher values print more messages. Ignored when `verbose` param is False.

#### ensemble\_model

`pycaret.classification` `pycaret.regression`

Following new parameters have been added:

* **fit\_kwargs: Optional\[dict] = None** Dictionary of arguments passed to the fit method of the model.
* **groups: Optional\[Union\[str, Any]] = None** Optional group labels when 'GroupKFold' is used for the cross-validation. It takes an array with shape (n\_samples, ) where n\_samples is the number of rows in the training dataset. When a string is passed, it is interpreted as the column name in the dataset containing group labels.

#### blend\_models

`pycaret.classification` `pycaret.regression`

Following new parameters have been added:

* **fit\_kwargs: Optional\[dict] = None** Dictionary of arguments passed to the fit method of the model.
* **groups: Optional\[Union\[str, Any]] = None** Optional group labels when 'GroupKFold' is used for the cross-validation. It takes an array with shape (n\_samples, ) where n\_samples is the number of rows in the training dataset. When a string is passed, it is interpreted as the column name in the dataset containing group labels.
* **weights: list, default = None** Sequence of weights (float or int) to weight the occurrences of predicted class labels (hard voting) or class probabilities before averaging (soft voting). Uses uniform weights when None.
* The default value for the `method` parameter has been changed from `hard` to `auto`.

#### stack\_models

`pycaret.classification` `pycaret.regression`

Following new parameters have been added:

* **fit\_kwargs: Optional\[dict] = None** Dictionary of arguments passed to the fit method of the model.
* **groups: Optional\[Union\[str, Any]] = None** Optional group labels when 'GroupKFold' is used for the cross-validation. It takes an array with shape (n\_samples, ) where n\_samples is the number of rows in the training dataset. When a string is passed, it is interpreted as the column name in the dataset containing group labels.

#### calibrate\_model

`pycaret.classification`

Following new parameters have been added:

* **fit\_kwargs: Optional\[dict] = None** Dictionary of arguments passed to the fit method of the model.
* **groups: Optional\[Union\[str, Any]] = None** Optional group labels when 'GroupKFold' is used for the cross-validation. It takes an array with shape (n\_samples, ) where n\_samples is the number of rows in the training dataset. When a string is passed, it is interpreted as the column name in the dataset containing group labels.

#### plot\_model

`pycaret.classification` `pycaret.regression`

Following new parameters have been added:

* **fold: int or scikit-learn compatible CV generator, default = None** Controls cross-validation. If None, the CV generator in the `fold_strategy` parameter of the `setup` function is used. When an integer is passed, it is interpreted as the 'n\_splits' parameter of the CV generator in the `setup` function.
* **fit\_kwargs: Optional\[dict] = None** Dictionary of arguments passed to the fit method of the model.
* **groups: Optional\[Union\[str, Any]] = None** Optional group labels when 'GroupKFold' is used for the cross-validation. It takes an array with shape (n\_samples, ) where n\_samples is the number of rows in the training dataset. When a string is passed, it is interpreted as the column name in the dataset containing group labels.

#### evaluate\_model

`pycaret.classification` `pycaret.regression`

Following new parameters have been added:

* **fold: int or scikit-learn compatible CV generator, default = None** Controls cross-validation. If None, the CV generator in the `fold_strategy` parameter of the `setup` function is used. When an integer is passed, it is interpreted as the 'n\_splits' parameter of the CV generator in the `setup` function.
* **fit\_kwargs: Optional\[dict] = None** Dictionary of arguments passed to the fit method of the model.
* **groups: Optional\[Union\[str, Any]] = None** Optional group labels when 'GroupKFold' is used for the cross-validation. It takes an array with shape (n\_samples, ) where n\_samples is the number of rows in the training dataset. When a string is passed, it is interpreted as the column name in the dataset containing group labels.

#### finalize\_model

`pycaret.classification` `pycaret.regression`

Following new parameters have been added:

* **fit\_kwargs: Optional\[dict] = None** Dictionary of arguments passed to the fit method of the model.
* **groups: Optional\[Union\[str, Any]] = None** Optional group labels when 'GroupKFold' is used for the cross-validation. It takes an array with shape (n\_samples, ) where n\_samples is the number of rows in the training dataset. When a string is passed, it is interpreted as the column name in the dataset containing group labels.
* **model\_only: bool, default = True** When set to False, only the model object is re-trained and all the transformations in Pipeline are ignored.

#### models

`pycaret.classification` `pycaret.regression` `pycaret.clustering` `pycaret.anomaly`

Following new parameters have been added:

* **internal: bool, default = False** When True, will return extra columns and rows used internally.
* **raise\_errors: bool, default = True** When False, will suppress all exceptions, ignoring models that couldn't be created.

### **PyCaret 2.1.2**

#### **Release Date: August 31, 2020 (BUG FIX)**

* Post-release `2.1` a bug has been reported preventing `predict_model` function to work in `regression` module in a new notebook session, when `transform_target` was set to `False` during model training. This issue has been fixed in PyCaret release `2.1.2`. To learn more about the issue: [https://github.com/pycaret/pycaret/issues/525](https://github.com/pycaret/pycaret/issues/525)

### **PyCaret 2.1.1**

#### **Release Date: August 30, 2020 (BUG FIX)**

* Post-release `2.1` a bug has been identified in MLFlow back-end. The error is only caused when `log_experiment` in the `setup` function is set to True and is applicable to all the modules. The cause of the error has been identified and an issue is opened with `MLFlow`. The error is caused by `infer_signature` function in `mlflow.sklearn.log_model` and is only raised when there are missing values in the dataset. This issue has been fixed in PyCaret release `2.1.1` by skipping the signature in cases where `MLFlow` raises exception

### **PyCaret 2.1**

#### **Release Date: August 28, 2020**

#### Summary of Changes

* **Model Deployment** Model deployment support for `gcp` and `azure` has been added in `deploy_model` function for all modules. See `documentation` for details.
* **Compare Models Budget Time** new parameter `budget_time` added in `compare_models` function. To set the upper limit on `compare_models` training time, `budget_time` parameter can be used.
* **Feature Selection** New feature selection method `boruta` has been added for feature selection. By default, `feature_selection_method` parameter in the `setup` function is set to `classic` but can be set to `boruta` for feature selection using boruta algorithm. This change is applicable for `pycaret.classification` and `pycaret.regression`.
* **Numeric Imputation** New method `zero` has been added in the `numeric_imputation` in the `setup` function. When method is set to `zero`, missing values are replaced with constant 0. Default behavior of `numeric_imputation` is unchanged.
* **Plot Model** New parameter `scale` has been added in `plot_model` for all modules to enable high quality images for research publications.
* **User Defined Loss Function** You can now pass `custom_scorer` for optimizing user defined loss function in `tune_model` for `pycaret.classification` and `pycaret.regression`. You must use `make_scorer` from `sklearn` to create custom loss function that can be passed into `custom_scorer` for the `tune_model` function.
* **Change in Pipeline Behavior** When using `save_model` the `model` object is appended into `Pipeline`, as such the behavior of `Pipeline` and `predict_model` is now changed. Instead of saving a `list`, `save_model` now saves `Pipeline` object where trained model is on last position. The user functionality on front-end for `predict_model` remains same.
* **Compare Models** parameter `blacklist` and `whitelist` is now renamed to `exclude` and `include` with no change in functionality.
* **Predict Model Labels** The `Label` column returned by `predict_model` function in `pycaret.classification` now returns the original label instead of encoded value. This change is made to make output from `predict_model` more human-readable. A new parameter `encoded_labels` is added, which is `False` by default. When set to `True`, it will return encoded labels.
* **Model Logging** Model persistence in the backend when `log_experiment` is set to `True` is now changed. Instead of using internal `save_model` functionality, it now adopts to `mlflow.sklearn.save_model` to allow the use of Model Registry and `MLFlow` native deployment functionalities.
* **CatBoost Compatibility** `CatBoostClassifier` is now compatible with `blend_models` in `pycaret.classification`. As such `blend_models` without any `estimator_list` will now result in blending total of `15` estimators including `CatBoostClassifier`.
* **Stack Models** `stack_models` in `pycaret.classification` and `pycaret.regression` now adopts to `StackingClassifier()` and `StackingRegressor` from `sklearn`. As such the `stack_models` function now returns `sklearn` object instead of custom `list` in previous versions.
* **Create Stacknet** `create_stacknet` in `pycaret.classification` and `pycaret.regression` is now removed.
* **Tune Model** `tune_model` in `pycaret.classification` and `pycaret.regression` now inherits params from the input `estimator`. As such if you have trained `xgboost`, `lightgbm` or `catboost` on gpu will not inherits training method from `estimator`.
* **Interpret Model** `**kwargs` argument now added in `interpret_model`.
* **Pandas Categorical Type** All modules are now compatible with `pandas.Categorical` object. Internally they are converted into object and are treated as the same way as `object` or `bool` is treated.
* **use\_gpu** A new parameter added in the `setup` function for `pycaret.classification` and `pycaret.regression`. In `2.1` it was added to prepare for the backend work required to make this change in future releases. As such using `use_gpu` param in `2.1` has no impact.
* **Unit Tests** Unit testing enhanced. Continious improvement in progress [https://github.com/pycaret/pycaret/tree/master/pycaret/tests](https://github.com/pycaret/pycaret/tree/master/pycaret/tests)
* **Automated Documentation Added** Automated documentation now added. Documentation on Website will only update for `major` releases 0.X. For all minor monthly releases, documentation will be available on: https://pycaret.readthedocs.io/en/latest/
* **Introduction of GitHub Actions** CI/CD build testing is now moved from `travis-ci` to `github-actions`. `pycaret-nightly` is now being published every 24 hours automatically.
* **Tutorials** All tutorials are now updated using `pycaret==2.0`. [https://github.com/pycaret/pycaret/tree/master/tutorials](https://github.com/pycaret/pycaret/tree/master/tutorials)
* **Resources** New resources added under `/pycaret/resources/` [https://github.com/pycaret/pycaret/tree/master/resources](https://github.com/pycaret/pycaret/tree/master/resources)
* **Example Notebook** Many example notebooks added under `/pycaret/examples/` [https://github.com/pycaret/pycaret/tree/master/examples](https://github.com/pycaret/pycaret/tree/master/examples)

### **PyCaret 2.0**

#### **Release Date: July 31, 2020**

#### Summary of Changes

* **Experiment Logging** MLFlow logging backend added. New parameters `log_experiment` `experiment_name` `log_profile` `log_data` added in `setup`. Available in `pycaret.classification` `pycaret.regression` `pycaret.clustering` `pycaret.anomaly` `pycaret.nlp`\

* **Save / Load Experiment** `save_experiment` and `load_experiment` function from `pycaret.classification` `pycaret.regression` `pycaret.clustering` `pycaret.anomaly` `pycaret.nlp` is removed in PyCaret 2.0\

* **System Logging** System log files now generated when `setup` is executed. `logs.log` file is saved in current working directory. Function `get_system_logs` can be used to access log file in notebook.\

* **Command Line Support** When using PyCaret 2.0 outside of Notebook, `html` parameter in `setup` must be set to False.\

* **Imbalance Dataset** `fix_imbalance` and `fix_imbalance_method` parameter added in `setup` for `pycaret.classification`. When set to True, SMOTE is applied by default to create synthetic datapoints for minority class. To change the method pass any class from `imblearn` that supports `fit_resample` method in `fix_imbalance_method` parameter.\

* **Save Plot** `save` parameter added in `plot_model`. When set to True, it saves the plot as `png` or `html` in current working directory.\

* **kwargs** `kwargs**` added in `create_model` for `pycaret.classification` `pycaret.regression` `pycaret.clustering` `pycaret.anomaly`\

* **choose\_better** `choose_better` and `optimize` parameter added in `tune_model` `ensemble_model` `blend_models` `stack_models` `create_stacknet` in `pycaret.classification` and `pycaret.regression`. Read the details below to learn more about thi added in `create_model` for `pycaret.classification` `pycaret.regression` `pycaret.clustering` `pycaret.anomaly`\

* **Training Time** `TT (Sec)` added in `compare_models` function for `pycaret.classification` and `pycaret.regression`\

* **New Metric: MCC** `MCC` metric added in score grid for `pycaret.classification`\

* **NEW FUNCTION: automl()** New function `automl` added in `pycaret.classification` `pycaret.regression`\

* **NEW FUNCTION: pull()** New function `pull` added in `pycaret.classification` `pycaret.regression`\

* **NEW FUNCTION: models()** New function `models` added in `pycaret.classification` `pycaret.regression` `pycaret.clustering` `pycaret.anomaly` `pycaret.nlp`\

* **NEW FUNCTION: get\_logs()** New function `get_logs` added in `pycaret.classification` `pycaret.regression` `pycaret.clustering` `pycaret.anomaly` `pycaret.nlp`\

* **NEW FUNCTION: get\_config()** New function `get_config` added in `pycaret.classification` `pycaret.regression` `pycaret.clustering` `pycaret.anomaly` `pycaret.nlp`\

* **NEW FUNCTION: set\_config()** New function `set_config` added in `pycaret.classification` `pycaret.regression` `pycaret.clustering` `pycaret.anomaly` `pycaret.nlp`\

* **NEW FUNCTION: get\_system\_logs** New function `get_logs` added in `pycaret.classification` `pycaret.regression` `pycaret.clustering` `pycaret.anomaly` `pycaret.nlp`\

* **CHANGE IN BEHAVIOR: compare\_models** `compare_models` now returns top\_n models defined by `n_select` parameter, by default set to 1.\

* **CHANGE IN BEHAVIOR: tune\_model** `tune_model` function in `pycaret.classification` and `pycaret.regression` now requires trained model object to be passed as `estimator` instead of string abbreviation / ID.\

* **REMOVED DEPENDENCIES** `awscli` and `shap` removed from requirements.txt. To use `interpret_model` function in `pycaret.classification` `pycaret.regression` and `deploy_model` function in `pycaret.classification` `pycaret.regression` `pycaret.clustering` `pycaret.anomaly`, these libraries will have to be installed separately.

#### setup

**`pycaret.classification`  `pycaret.regression`  `pycaret.clustering`  `pycaret.anomaly`  `pycaret.nlp`**\


* **`remove_perfect_collinearity`** parameter added in `setup()`. Default set to False.\
  When set to True, perfect collinearity (features with correlation = 1) is removed from the dataset, When two features are 100% correlated, one of it is randomly dropped from the dataset.\

* **`fix_imbalance`** parameter added in `setup()`. Default set to False.\
  When dataset has unequal distribution of target class it can be fixed using fix\_imbalance parameter. When set to True, SMOTE (Synthetic Minority Over-sampling Technique) is applied by default to create synthetic datapoints for minority class.\

* **`fix_imbalance_method`** parameter added in `setup()`. Default set to None.\
  When fix\_imbalance is set to True and fix\_imbalance\_method is None, 'smote' is applied by default to oversample minority class during cross validation. This parameter accepts any module from 'imblearn' that supports 'fit\_resample' method.\

* **`data_split_shuffle`** parameter added in `setup()`. Default set to True.\
  If set to False, prevents shuffling of rows when splitting data.\

* **`folds_shuffle`** parameter added in `setup()`. Default set to False.\
  If set to False, prevents shuffling of rows when using cross validation.\

* **`n_jobs`** parameter added in `setup()`. Default set to -1.\
  The number of jobs to run in parallel (for functions that supports parallel processing) -1 means using all processors. To run all functions on single processor set n\_jobs to None\

* **`html`** parameter added in `setup()`. Default set to True.\
  If set to False, prevents runtime display of monitor. This must be set to False when using environment that doesnt support HTML.\

* **`log_experiment`** parameter added in `setup()`. Default set to False.\
  When set to True, all metrics and parameters are logged on MLFlow server.\

* **`experiment_name`** parameter added in `setup()`. Default set to None.\
  Name of experiment for logging. When set to None, 'clf' is by default used as alias for the experiment name.\

* **`log_plots`** parameter added in `setup()`. Default set to False.\
  When set to True, specific plots are logged in MLflow as a png file.\

* **`log_profile`** parameter added in `setup()`. Default set to False.\
  When set to True, data profile is also logged on MLflow as a html file.\

* **`log_data`** parameter added in `setup()`. Default set to False.\
  When set to True, train and test dataset are logged as csv.\

* **`verbose`** parameter added in `setup()`. Default set to True.\
  Information grid is not printed when verbose is set to False.

#### compare\_models

**`pycaret.classification`  `pycaret.regression`**

* **`whitelist`** parameter added in `compare_models`. Default set to None.\
  In order to run only certain models for the comparison, the model ID's can be passed as a list of strings in whitelist param.\

* **`n_select`** parameter added in `compare_models`. Default set to 1.\
  Number of top\_n models to return. use negative argument for bottom selection. For example, n\_select = -3 means bottom 3 models.\

* **`verbose`** parameter added in `compare_models`. Default set to True.\
  Score grid is not printed when verbose is set to False.

#### create\_model

**`pycaret.classification`  `pycaret.regression`  `pycaret.clustering`  `pycaret.anomaly`**

* **`cross_validation`** parameter added in `create_model`. Default set to True.\
  When cross\_validation set to False fold parameter is ignored and model is trained on entire training dataset. No metric evaluation is returned. Only applicable in `pycaret.classification` and `pycaret.regression`\

* **`system`** parameter added in `create_model`. Default set to True.\
  Must remain True all times. Only to be changed by internal functions**.**\

* **`ground_truth`** parameter added in `create_model`. Default set to None.\
  When ground\_truth is provided, Homogeneity Score, Rand Index, and Completeness Score is evaluated and printer along with other metrics. This is only available in **`pycaret.clustering`**\

* **`kwargs`** parameter added in `create_model`.\
  Additional keyword arguments to pass to the estimator.

#### tune\_model

**`pycaret.classification`  `pycaret.regression`  `pycaret.clustering`  `pycaret.anomaly`  `pycaret.nlp`**

* **`custom_grid`** parameter added in `tune_model`. Default set to None.\
  To use custom hyperparameters for tuning pass a dictionary with parameter name and values to be iterated. When set to None it uses pre-defined tuning grid. For `pycaret.clustering` `pycaret.anomaly` `pycaret.nlp`, custom\_grid param must be a list of values to iterate over.\

* **`choose_better`** parameter added in `tune_model`. Default set to False.\
  When set to set to True, base estimator is returned when the performance doesn't improve by tune\_model. This gurantees the returned object would perform atleast equivalent to base estimator created using create\_model or model returned by compare\_models.

#### ensemble\_model

**`pycaret.classification`  `pycaret.regression`**

* **`choose_better`** parameter added in `ensemble_model`. Default set to False.\
  When set to set to True, base estimator is returned when the performance doesn't improve by tune\_model. This gurantees the returned object would perform atleast equivalent to base estimator created using create\_model or model returned by compare\_models.\

* **`optimize`** parameter added in `ensemble_model`. Default set to **`Accuracy`** for `pycaret.classification` and **`R2`** for `pycaret.regression`.\
  Only used when choose\_better is set to True. optimize parameter is used to compare emsembled model with base estimator. Values accepted in optimize parameter for `pycaret.classification` are 'Accuracy', 'AUC', 'Recall', 'Precision', 'F1', 'Kappa', 'MCC' and for `pycaret.regression` are 'MAE', 'MSE', 'RMSE' 'R2', 'RMSLE' and 'MAPE'.

#### blend\_models

**`pycaret.classification`  `pycaret.regression`**

* **`choose_better`** parameter added in `blend_models`. Default set to False.\
  When set to set to True, base estimator is returned when the performance doesn't improve by tune\_model. This gurantees the returned object would perform atleast equivalent to base estimator created using create\_model or model returned by compare\_models.\

* **`optimize`** parameter added in `blend_models`. Default set to **`Accuracy`** for `pycaret.classification` and **`R2`** for `pycaret.regression`.\
  Only used when choose\_better is set to True. optimize parameter is used to compare emsembled model with base estimator. Values accepted in optimize parameter for `pycaret.classification` are 'Accuracy', 'AUC', 'Recall', 'Precision', 'F1', 'Kappa', 'MCC' and for `pycaret.regression` are 'MAE', 'MSE', 'RMSE' 'R2', 'RMSLE' and 'MAPE'.

#### stack\_models

**`pycaret.classification`  `pycaret.regression`**

* **`choose_better`** parameter added in `stack_models`. Default set to False.\
  When set to set to True, base estimator is returned when the performance doesn't improve by tune\_model. This gurantees the returned object would perform atleast equivalent to base estimator created using create\_model or model returned by compare\_models.\

* **`optimize`** parameter added in `stack_models`. Default set to **`Accuracy`** for `pycaret.classification` and **`R2`** for `pycaret.regression`.\
  Only used when choose\_better is set to True. optimize parameter is used to compare emsembled model with base estimator. Values accepted in optimize parameter for `pycaret.classification` are 'Accuracy', 'AUC', 'Recall', 'Precision', 'F1', 'Kappa', 'MCC' and for `pycaret.regression` are 'MAE', 'MSE', 'RMSE' 'R2', 'RMSLE' and 'MAPE'.

#### create\_stacknet

**`pycaret.classification`  `pycaret.regression`**

* **`choose_better`** parameter added in `create_stacknet`. Default set to False.\
  When set to set to True, base estimator is returned when the performance doesn't improve by tune\_model. This gurantees the returned object would perform atleast equivalent to base estimator created using create\_model or model returned by compare\_models.\

* **`optimize`** parameter added in `create_stacknet`. Default set to **`Accuracy`** for `pycaret.classification` and **`R2`** for `pycaret.regression`.\
  Only used when choose\_better is set to True. optimize parameter is used to compare emsembled model with base estimator. Values accepted in optimize parameter for `pycaret.classification` are 'Accuracy', 'AUC', 'Recall', 'Precision', 'F1', 'Kappa', 'MCC' and for `pycaret.regression` are 'MAE', 'MSE', 'RMSE' 'R2', 'RMSLE' and 'MAPE'.

#### predict\_model

**`pycaret.classification`  `pycaret.regression`**

* **`verbose`** parameter added in `predict_model`. Default set to True.\
  Holdout score grid is not printed when verbose is set to False.

#### plot\_model

**`pycaret.classification`  `pycaret.regression`  `pycaret.clustering`  `pycaret.anomaly`  `pycaret.nlp`**\


* **`save`** parameter added in `plot_model`. Default set to False.\
  When set to True, Plot is saved as a 'png' file in current working directory.\

* **`verbose`** parameter added in `plot_model`. Default set to True.\
  Progress bar not shown when verbose set to False.\

* **`system`** parameter added in `plot_model`. Default set to True.\
  Must remain True all times. Only to be changed by internal functions.

#### NEW FUNCTION: automl

**`pycaret.classification`  `pycaret.regression`**

* This function returns the best model out of all models created in current active environment based on metric defined in optimize parameter.

#### Parameters:

* **`optimize`** string, default = 'Accuracy' for `pycaret.classification` and 'R2' for `pycaret.regression`\
  Other values you can pass in optimize param are 'AUC', 'Recall', 'Precision', 'F1', 'Kappa', and 'MCC' for `pycaret.classification` and 'MAE', 'MSE', 'RMSE', 'R2', 'RMSLE', and 'MAPE' for `pycaret.regression`\

* **`use_holdout`** bool, default = False\
  When set to True, metrics are evaluated on holdout set instead of CV.

#### NEW FUNCTION: pull

**`pycaret.classification`  `pycaret.regression`**

* This function returns the last printed score grid as pandas dataframe.

#### NEW FUNCTION: models

**`pycaret.classification`  `pycaret.regression`  `pycaret.clustering`  `pycaret.anomaly`  `pycaret.nlp`**

* This function Returns the table of models available in model library.

#### Parameters:

* **`type`** string, default = None\
  linear : filters and only return linear models\
  tree : filters and only return tree based models\
  ensemble : filters and only return ensemble models

`type` parameter only available in `pycaret.classification` and `pycaret.regression`

#### NEW FUNCTION: get\_logs

**`pycaret.classification`  `pycaret.regression`  `pycaret.clustering`  `pycaret.anomaly`  `pycaret.nlp`**

* This function returns a table with experiment logs consisting run details, parameter, metrics and tags.

#### Parameters:

* **`experiment_name`** string, default = None\
  When set to None current active run is used.\

* **`save`** bool, default = False\
  When set to True, csv file is saved in current directory.

#### NEW FUNCTION: get\_config

**`pycaret.classification`  `pycaret.regression`  `pycaret.clustering`  `pycaret.anomaly`  `pycaret.nlp`**

* This function is used to access global environment variables. Check docstring for the list of global var accessible.

#### NEW FUNCTION: set\_config

**`pycaret.classification`  `pycaret.regression`  `pycaret.clustering`  `pycaret.anomaly`  `pycaret.nlp`**

* This function is used to reset global environment variables. Check docstring for the list of global var accessible.

#### NEW FUNCTION: get\_system\_logs

**`pycaret.classification`  `pycaret.regression`  `pycaret.clustering`  `pycaret.anomaly`  `pycaret.nlp`**

* This function is reads and print 'logs.log' file from current active directory. logs.log is generated from `setup` is initialized in any module.
