---
description: >-
  PyCaret 2.3.6 is Here! Learn What’s New? From EDA to Deployment to AI
  Fairness — By far the biggest release of PyCaret
---

# PyCaret 2.3.6 is Here! Learn What’s New?

### 🚀 Introduction <a href="#261e" id="261e"></a>

PyCaret is an open-source, low-code machine learning library in Python that automates machine learning workflows. It is an end-to-end machine learning and model management tool that speeds up the experiment cycle exponentially and makes you more productive.

By far PyCaret 2.3.6 is the biggest release in terms of the new features and functionalities. This article demonstrates the use of new functionalities added in the recent release of [PyCaret 2.3.6](https://pycaret.gitbook.io/docs/get-started/release-notes#pycaret-2.3.6).

### 💻 Installation <a href="#90a4" id="90a4"></a>

Installation is easy and will only take a few minutes. PyCaret’s default installation from pip only installs hard dependencies as listed in the [requirements.txt](https://github.com/pycaret/pycaret/blob/master/requirements.txt) file.

```
pip install pycaret
```

To install the full version:

```
pip install pycaret[full]
```

### 📈 Dashboard <a href="#5b8f" id="5b8f"></a>

This function will generate the interactive dashboard for a trained model. The dashboard is implemented using the [ExplainerDashboard](http://explainerdashboard.readthedocs.io).

```
# load dataset
from pycaret.datasets import get_data
data = get_data('iris')

# init setupfrom pycaret.classification import *
s = setup(data, target = 'species', session_id = 123)

# train model
lr = create_model('lr')

# generate dashboard
dashboard(lr)
```

![](https://cdn-images-1.medium.com/max/800/1\*MlXSTs8BmiICexLfcajJKA.png)

**Video Demo:**

{% embed url="https://www.youtube.com/watch?v=FZ5-GtdYez0" %}

### 📊 Exploratory Data Analysis (EDA) <a href="#3223" id="3223"></a>

This function will generate automated EDA using the [AutoViz](https://github.com/AutoViML/AutoViz) integration.

```
# load dataset
from pycaret.datasets import get_data
data = get_data('iris')

# init setup
from pycaret.classification import *
s = setup(data, target = 'species', session_id = 123)

# generate EDA
eda()
```

![](https://cdn-images-1.medium.com/max/800/1\*lByuyZL-pR2eZ0rPc1qsxA.png)

**Video Demo:**

{% embed url="https://www.youtube.com/watch?v=Pm5VOuYqU4Q" %}

### 🚊 Convert Model <a href="#2e61" id="2e61"></a>

This function will transpile trained machine learning models into native inference scripts in different programming languages (Python, C, Java, Go, JavaScript, Visual Basic, C#, PowerShell, R, PHP, Dart, Haskell, Ruby, F#). This functionality is very useful if you want to deploy models into environments where you can’t install your normal Python stack to support model inference.

```
# load dataset
from pycaret.datasets import get_data
data = get_data('iris')

# init setup
from pycaret.classification import *
s = setup(data, target = 'species', session_id = 123)

# train model
lr = create_model('lr')

# convert model
lr_java = convert_model(lr, language = 'java')
print(lr_java)
```

![](<../../.gitbook/assets/image (520).png>)

**Video Demo:**

{% embed url="https://www.youtube.com/watch?v=xwQgfNC7808" %}

### ☑️ Check Fairness <a href="#95da" id="95da"></a>

There are many approaches to conceptualizing fairness. This new function follows the approach known as [group fairness](https://github.com/fairlearn/fairlearn), which asks: Which groups of individuals are at risk for experiencing harm. This function provides fairness-related metrics between different groups (also called subpopulations).

```
# load dataset
from pycaret.datasets import get_data
data = get_data('income')

# init setup
from pycaret.classification import *
s = setup(data, target = 'income >50K', session_id = 123)

# train model
lr = create_model('lr')

# check fairness
check_fairness(lr, sensitive_features = ['race'])
```

![](<../../.gitbook/assets/image (452).png>)

**Video Demo:**

{% embed url="https://youtu.be/mjhDKuLRpM0" %}

### 📩 Create Web API <a href="#ea0e" id="ea0e"></a>

This function will create a POST API for the ML pipeline for inference using [FastAPI](https://github.com/tiangolo/fastapi) framework. It only creates the API and doesn’t run it automatically.

```
# load dataset
from pycaret.datasets import get_data
data = get_data('iris')

# init setup
from pycaret.classification import *
s = setup(data, target = 'species', session_id = 123)

# train model
lr = create_model('lr')

# create API
create_api(lr, 'my_first_api')

# Run the API
!python my_first_api.py
```

![](<../../.gitbook/assets/image (162).png>)

![](<../../.gitbook/assets/image (170) (1).png>)

#### **Video Demo:** <a href="#4653" id="4653"></a>

{% embed url="https://www.youtube.com/watch?t=1s&v=88M9c5Hc-k0" %}

### 🚢 Create Docker <a href="#4653" id="4653"></a>

This function will create a `Dockerfile`and `requirements`file for your API end-point.

```
# load dataset
from pycaret.datasets import get_data
data = get_data('iris')

# init setup
from pycaret.classification import *
s = setup(data, target = 'species', session_id = 123)

# train model
lr = create_model('lr')

# create API
create_api(lr, 'my_first_api')

# create Docker
create_docker('my_first_api')
```

![](<../../.gitbook/assets/image (514).png>)

**Video Demo:**

{% embed url="https://www.youtube.com/watch?v=xMgwEJ57uxs" %}

### 💻 Create Web Application <a href="#5897" id="5897"></a>

This function creates a basic [Gradio](https://github.com/gradio-app/gradio) web app for inference. It will later be expanded for other app types such as Streamlit.

```
# load dataset
from pycaret.datasets import get_data
data = get_data('iris')

# init setup
from pycaret.classification import *
s = setup(data, target = 'species', session_id = 123)

# train model
lr = create_model('lr')
```

![](<../../.gitbook/assets/image (54).png>)

**Video Demo:**

{% embed url="https://www.youtube.com/watch?v=4JyYhbW6eCA" %}

### 🎰 Monitor Drift of ML Models <a href="#b01d" id="b01d"></a>

A new parameter called `drift_report` is added to the `predict_model` function that generates the drift report using [Evidently AI](https://github.com/evidentlyai/evidently?) framework. At the moment this functionality is in experimental mode and will only work on test data. Later on, it will be expanded for production use.

```
# load dataset
from pycaret.datasets import get_data
data = get_data('iris')

# init setup
from pycaret.classification import *
s = setup(data, target = 'species', session_id = 123)

# train model
lr = create_model('lr')

# generate report
preds = predict_model(lr, drift_report = True)
```

![](<../../.gitbook/assets/image (84).png>)

![](<../../.gitbook/assets/image (259).png>)

**Video Demo:**

{% embed url="https://www.youtube.com/watch?v=C9TNq1bndRI" %}

### 🔨 Plot Model is now more configurable <a href="#ac70" id="ac70"></a>

`plot_model` function is PyCaret is now more configurable. For example, previously if you wanted to see percentages in Confusion Matrix instead of absolute numbers, it wasn’t possible, or if you want to change the color map of visuals, it wasn’t possible. Now it is possible with the new parameter `plot_kwargs` in the `plot_model` function. See example:

```
# load dataset
from pycaret.datasets import get_data
data = get_data('iris')

# init setup
from pycaret.classification import *
s = setup(data, target = 'species', session_id = 123)

# train model
lr = create_model('lr')

# plot model (without plot kwargs)
plot_model(lr, plot = 'confusion_matrix') 

# plot model (with plot kwargs)
plot_model(lr, plot = 'confusion_matrix', plot_kwargs = {'percent' : True})
```

![](<../../.gitbook/assets/image (405).png>)

### 🏆Optimize Threshold <a href="#bc52" id="bc52"></a>

This is not a new function but it was completely revamped in 2.3.6. This function is to optimize the probability threshold for binary classification problems. Previously you had to pass cost function as `true_positive` , `false_positive` , `true_negative` , `false_negative` in this function and now it automatically picks up all the metrics including the custom ones from your active experiment run.

```
# load dataset
from pycaret.datasets import get_data
data = get_data('blood')

# init setup
from pycaret.classification import *
s = setup(data, target = 'Class', session_id = 123)

# train model
lr = create_model('lr')

# optimize threshold
optimize_threshold(lr)
```

![](<../../.gitbook/assets/image (16).png>)

### 📚 New Documentation <a href="#c5b7" id="c5b7"></a>

The biggest and hardest of all is the completely new documentation. This is a single source of truth for everything related to PyCaret, from official tutorials to release notes and from API ref to community contributions. Take a video tour:

{% embed url="https://youtu.be/NpJiD5H0dJc" %}

Finally, if you want to take the tour of all new functionalities added in 2.3.6, watch this 10 minutes video:

{% embed url="https://www.youtube.com/watch?t=4s&v=Qr6Hu2t2gwY" %}

To learn about all the other changes, bug fixes, and minor updates in PyCaret 2.3.6, check out the detailed [release notes](https://github.com/pycaret/pycaret/releases/tag/2.3.6).

Thank you for reading.

### :link: Important Links <a href="#b749" id="b749"></a>

* 📚 [Official Docs:](https://pycaret.gitbook.io) The bible of PyCaret. Everything is here.
* 🌐 [Official Web:](https://www.pycaret.org) Check out our official website
* 😺 [GitHub](https://www.github.com/pycaret/pycaret) Check out our Git
* ⭐ [Tutorials](https://pycaret.gitbook.io/docs/get-started/tutorials) New to PyCaret? Check out our official notebooks!
* 📋 [Example Notebooks](https://github.com/pycaret/pycaret/tree/master/examples) created by the community.
* 📙 [Blog](https://pycaret.gitbook.io/docs/learn-pycaret/official-blog) Tutorials and articles by contributors.
* ❓ [FAQs](https://pycaret.gitbook.io/docs/learn-pycaret/faqs) Check out frequently asked questions.
* 📺 [Video Tutorials](https://pycaret.gitbook.io/docs/learn-pycaret/videos) Our video tutorial from various events.
* 📢 [Discussions](https://github.com/pycaret/pycaret/discussions) Have questions? Engage with community and contributors.
* 🛠️ [Changelog](https://pycaret.gitbook.io/docs/get-started/release-notes) Changes and version history.
* 🙌 [User Group](https://www.meetup.com/pycaret-user-group/) Join our Meetup user group.
