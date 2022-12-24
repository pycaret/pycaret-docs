---
description: Frequently Asked Questions!
---

# ❓ FAQs

{% hint style="info" %}
**PyCaret 3.0-rc is now available**. `pip install --pre pycaret` to try it. Check out this example [Notebook](https://colab.research.google.com/drive/1\_H0sHYhzKGZDmgzrQLosuZAR3nOaL6CN?usp=sharing).
{% endhint %}

<details>

<summary>Why PyCaret?</summary>

The short answer is it's an open-source, low-code machine learning library built on top of your favorite libraries and frameworks like _scikit-learn, xgboost, lightgbm, etc._ Machine Learning experments take a lot of iterations and the primary goal of PyCaret is to give you the ability to iterate with lightning speed. In comparison with the other awesome open-source machine learning libraries, PyCaret is an alternate low-code library that can be used to replace hundreds of lines of code with a few lines only. Give it a try!

</details>

<details>

<summary>Does PyCaret work with all OS and Python versions?</summary>

PyCaret is tested and supported on the following 64-bit systems:

* Python 3.6 – 3.8&#x20;
* Python 3.9 for Ubuntu only
* Ubuntu 16.04 or later
* Windows 7 or later

PyCaret also works on Mac OS but we do not guarantee the performance as the releases are not tested for Mac. To learn more about our testing workflows, [click here](https://github.com/pycaret/pycaret/blob/master/.github/workflows/test.yml).

</details>

<details>

<summary>Can  I use PyCaret on Google Colab or Kaggle Notebooks?</summary>

Absolutely. Just do `pip install pycaret`

Since base installations on these platforms are not in our control, time-to-time, you may have issues in installing PyCaret due to some dependency conflicts. Those issues with the temporary solutions are reported [here](../get-started/installation.md#common-installation-issues).

</details>

<details>

<summary>Does PyCaret support model training on GPU?</summary>

Yes. We have integrated PyCaret with the amazing [RAPIDS.AI](https://rapids.ai/) project. To use GPU instead of CPU, just pass `use_gpu=True` in the `setup` function.&#x20;

**This will use CPU for model training:**

```
from pycaret.classification import *
s = setup(data, target = 'target_name')
```

**This will use GPU for model training:**

```
from pycaret.classification import *
s = setup(data, target = 'target_name', use_gpu = True)
```

There is no change in the use of the API, however, in some cases, additional libraries have to be installed as they are not installed with the default version or the full version of PyCaret. You can learn more about this [here](../get-started/installation.md#gpu).

</details>

<details>

<summary>Can I use PyCaret for distributed training on cluster like Spark, Dask, Ray, etc.?</summary>

Yes. All the functions of PyCaret are just normal python function and all these frameworks like Spark, Dask, Ray provides you an option to distribute any arbitrary code on a cluster of machines. In future releases, we aim to integrate these distributed frameworks within PyCaret but for now, if you are interested in doing that, [This article](https://towardsdatascience.com/scaling-pycaret-with-spark-or-dask-through-fugue-60bdc3ce133f) by the [Fugue project team](https://github.com/fugue-project/fugue) shows how you can distribute PyCaret code on Spark or Dask without any significant changes to the code.

</details>

<details>

<summary>How can I contribute to PyCaret?</summary>

Thank you for choosing to contribute to PyCaret. There are a ton of great open-source projects out there, so we appreciate your interest in contributing to PyCaret. Please check out our [Contribution Guidelines](https://github.com/pycaret/pycaret/blob/master/CONTRIBUTING.md).

</details>

<details>

<summary>How can I support PyCaret If I can't code?</summary>

Absolutely. There are many ways you can support us. You can join our documentation team and help us build and maintain this amazing documentation that is used by thousands of members every day. [Learn more](../#support-us) about other ways you can support us.

</details>

<details>

<summary>Does PyCaret support Deep Learning or Reinforcement Learning?</summary>

Not yet. In the future, maybe.

</details>

<details>

<summary>Can I integrate PyCaret with BI tools like Power BI, Tableau, Qlik, etc.?</summary>

Yes, any tool that supports the Python environment. You can use PyCaret within Power BI, Tableau, SQL, Alteryx, KNIME. If you would like to learn more, read these [official tutorials](official-blog/#pycaret-and-bi-integrations).

</details>

<details>

<summary>Does PyCaret gurantee end-to-end experiment reproducibility?</summary>

Absolutely. Without a guarantee for reproducibility, any framework is pretty much useless. In any ML workflow, there are many aspects that cause randomization such as `train_test_split`. Sometimes the randomization is also built in the algorithm inherently. Some examples are Random Forest, Extra Trees, Gradient Boosting Machines. To ensure that you can reproduce your end-to-end experiment at a later time, you must pass `session_id` parameter in the `setup`.

**Example:**

```
from pycaret.classification import *
s = setup(data, target = 'target_name', session_id = 123)
```

It doesn't matter what number you pass to `session_id` as long as you can remember it. `session_id` parameter in PyCaret is equivalent to `random_state` in scikit-learn models. The benefit here is we take the `session_id` from the `setup` and perpetuate to all the functions that uses `random_state` so that you nothing to worry about.

</details>

<details>

<summary>Can I run PyCaret on command line or any other environment than Notebook?</summary>

Absolutely. PyCaret is designed and developed to work in a Notebook environment, that doesn't mean you can't use it outside of Notebook in other IDE's such as Visual Code, PyCharm, or Spyder. When you are using PyCaret outside of the Notebook environment, you must pass `html=False` and `silent=True` in the `setup` function. Since PyCaret uses IPython for some callbacks functionality, without passing these two parameters explicitly, your code will fail when you are outside of the Notebook environment.&#x20;

**NOTE:** The name of these parameters may change in the future to something like `mode='notebook'.`

</details>

<details>

<summary>How can I bypass the user confirmation for data types when I run the setup function?</summary>

Whenever you run `setup` in any module of PyCaret, it generates a dialogue box to confirm data types where users are expected to press enter to continue. This is a helpful feature when you are using PyCaret during active experimentation in Notebook but when you are using PyCaret in the command line or as a Python script, you must bypass the confirmation dialogue box. You can do that by passing `silent=True` in the `setup` function.

**Example:**

```
from pycaret.classification import *
s = setup(data, target = 'target_name', silent = True)
```

</details>

<details>

<summary>How can I change verbosity in PyCaret?</summary>

Most functions in PyCaret has `verbose` parameter. Simply set `verbose=False` in the function.&#x20;

**Example:**

```
lr = create_model('lr', verbose = False)
```

</details>

<details>

<summary>Sometimes logs get printed on my screen as the code is running. Can I silent the logger?</summary>

We have noticed in some situations that the logger of PyCaret can conflict with other libraries in the environment causing an abnormal behavior resulting in logs being printed on the screen (Notebook or CLI) as the code is running. While in the next major release (3.0), we are planning to make the logger more configurable, allowing you to totally disable it if you want. In the meantime, there is a way around using environment variables. Run the following code on the top of your Notebook:

```
import os
os.environ["PYCARET_CUSTOM_LOGGING_LEVEL"] = "CRITICAL"
```

**NOTE:** This command will set an environment variable that is used by PyCaret's logger. Setting it to `CRITICAL` means that only critical messages will be logged and there aren't many critical messages in PyCaret.&#x20;

</details>

<details>

<summary>I am having issues in installating PyCaret, what can I do?</summary>

The first place is to check out [Common Installation Issues](../get-started/installation.md#common-installation-issues) and then [Issues](https://github.com/pycaret/pycaret/issues) on our GitHub.&#x20;

</details>

<details>

<summary>Can I add my own custom models in PyCaret?</summary>

Absolutely. PyCaret's vision is to give you full control of your ML pipeline. To add custom models, there is only one rule. They must be compatible with standard `sklearn` API. To learn how to do it, you can read the following tutorials by Fahad Akbar:

* [Custom Estimator with PyCaret - Part I](https://towardsdatascience.com/custome-estimator-with-pycaret-part-1-by-fahad-akbar-839513315965)
* [Custom Estimator with PyCaret - Part II](https://towardsdatascience.com/custom-estimator-with-pycaret-part-2-by-fahad-akbar-aee4dbdacbf)

</details>

<details>

<summary>Can I add custom metrics for cross-validation in PyCaret?</summary>

Absolutely. PyCaret aim's to balance the abstraction with flexibility and so far we are doing a pretty good job. You can use PyCaret's `add_metric` and `remove_metric` functions to add or remove metrics used for cross-validation.&#x20;

</details>

<details>

<summary>Can I just use PyCaret for data preprocessing?</summary>

Yes if you would like. You can run the `setup` function which handles all the data preprocessing and after that you can access the transformed train set and test set using the `get_config` function.&#x20;

**Example:**

```
from pycaret.classification import *
s = setup(data, target = 'target_name')

X_train, y_train = get_config('X_train'), get_config('y_train')
X_test, y_test = get_config('X_test'), get_config('y_test')
```

</details>

<details>

<summary>Can I export models from PyCaret and work on them outside of PyCaret?</summary>

Absolutely. You can use the `save_model` function of PyCaret to export the entire Pipeline as a `pkl` file. [Learn more](../get-started/functions/#save-model) about this function.

</details>

<details>

<summary>Why is there no object-oriented API for PyCaret? Will there ever be?</summary>

The first release (1.0) of PyCaret had many several critical design decisions which quickly became the common practice in the community. Having a solo functional API was one of the choices. However, in subsequent releases, we realize the use case and need for a more conventional OOP API which is now on its way. The default API of PyCaret will continue to be the functional API as a very large user base depends on it. However, in the next major release (3.0) we will have a separate OOP API for users who are interested in using it.

**Functional API Example (current)**

```
from pycaret.classification import *
s = setup(data, target = 'target_name')
best_model = compare_models()
```

**Object Oriented API Example (Future state)**

```
from pycaret.classification import ClassificationExperiment
exp = ClassificationExperiment()
exp.setup(data, target = 'target_name')
best_model = exp.compare_models()
```

</details>

<details>

<summary>Can I deploy ML pipelines on cloud using PyCaret?</summary>

Absolutely. PyCaret is an end-to-end library with a lot of deployment functionalities. There are many official tutorials on deployment on different cloud platforms such as Azure, AWS, and GCP. You can check out these [tutorials here](official-blog/#pycaret-add-ml-deployment).

</details>

<details>

<summary>Can I use models trained using PyCaret for inference without installing PyCaret?</summary>

Not right now but with our next major release (3.0) our goal is to allow you to use plain `sklearn` for inference runtime. At the moment, there are few custom functionalities of PyCaret in the Pipeline that forces you to install `pycaret` during inference but we are committed to either removing those custom functionalities in the future release or push those to the base libraries like scikit-learn. Our goal and vision is to become a mega-abstractor framework for training and ML development. We do not want to reinvent the wheel. We do not want you to carry the huge overhead of PyCaret's framework during inference.

</details>

<details>

<summary>I ran the setup function and it is stucked, what should I do? </summary>

If your setup function is stuck, the first thing you should check is if you are in an environment that allows for a confirmatory dialogue box and if not you must pass `silent=True` in the setup. Secondly, if you are using Visual Code the dialogue box appears on the top of the screen as opposed to inline as you might have seen with Jupyter Notebook. Finally, sometimes it may really take a very long time especially if your dataset has categorical features with many levels (1000+ levels). In that case, you should try to combine the levels and make the features less granular before passing into PyCaret. If all this doesn't resolve your issue and you are very certain that this is some kind of bug or you can improve the code, please feel free to open a new [Issue ](https://github.com/pycaret/pycaret/issues)on our GitHub.

</details>

<details>

<summary>Can I install and run PyCaret on Apply M1 Macbook?</summary>

It's not straighforward due to some issues in the underlying dependencies of PyCaret. However, if you have tried everything and still can't find a solution, this [article](https://pareekshithkatti.medium.com/setting-up-python-for-data-science-on-m1-mac-ced8a0d05911) by Pareekshith Katti may help you.

</details>

<details>

<summary>Do I need a powerful computer to use PyCaret?</summary>

No, as long as your data can fit in the memory, you can use PyCaret. No super computer is needed.

</details>

<details>

<summary>Why is my pull request not getting any attention?</summary>

The review process may take some time. You should not be discouraged by delays in review on your pull request. We have many features that are requested by the community and only limited time from our maintainers to review and approve these pull requests. Since every feature comes at a cost of lifetime maintenance, we care a lot about getting things right the first time.&#x20;

</details>

<details>

<summary>Is PyCaret comparable to scikit-learn and ML libraries and framework?</summary>

Well, PyCaret is built on top of common ML libraries and frameworks such as scikit-learn, LightGBM, XGBoost, etc. The benefit of using PyCaret is you don't have to write a lot of code. The underlying models and evaluation framework is the same as what you are used to. When we first created PyCaret we did a small comparison of performing a given set of tasks using PyCaret vs. without using PyCaret and here are the results:

![](<../.gitbook/assets/image (202).png>)

</details>
