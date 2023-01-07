---
description: An open-source, low-code machine learning library in Python
coverY: 0
---

# Welcome to PyCaret

{% hint style="info" %}
**PyCaret 3.0-rc is now available**. `pip install --pre pycaret` to try it. Check out this example [Notebook](https://colab.research.google.com/drive/1\_H0sHYhzKGZDmgzrQLosuZAR3nOaL6CN?usp=sharing).
{% endhint %}

PyCaret is an open-source, low-code machine learning library in Python that automates machine learning workflows. It is an end-to-end machine learning and model management tool that exponentially speeds up the experiment cycle and makes you more productive.

Compared with the other open-source machine learning libraries, PyCaret is an alternate low-code library that can be used to replace hundreds of lines of code with a few lines only. This makes experiments exponentially fast and efficient. PyCaret is essentially a Python wrapper around several machine learning libraries and frameworks, such as scikit-learn, XGBoost, LightGBM, CatBoost, spaCy, Optuna, Hyperopt, Ray, and a few more.

The design and simplicity of PyCaret are inspired by the emerging role of citizen data scientists, a term first used by Gartner. Citizen Data Scientists are power users who can perform both simple and moderately sophisticated analytical tasks that would previously have required more technical expertise.

## Quick Links

| ⭐ [**Tutorials**](get-started/tutorials.md)****                             | Checkout our official notebooks!        |
| --------------------------------------------------------------------------- | --------------------------------------- |
| 📋 [**Examples**](learn-pycaret/examples.md)****                            | Example notebooks.                      |
| 📙 [**Blog**](learn-pycaret/official-blog/)****                             | Tutorials and articles by contributors. |
| 📚 [**API Reference**](https://pycaret.readthedocs.io/en/latest/index.html) | The detailed API docs of PyCaret        |
| 📺 [**Video Tutorials**](learn-pycaret/videos.md)****                       | Our video tutorial from various events. |
| 📢 [**Discussions**](https://github.com/pycaret/pycaret/discussions)        | Engage with community and contributors. |
| 🛠️ [**Changelog**](get-started/release-notes.md)****                       | Changes and version history.            |
| :question:[**FAQs**](learn-pycaret/faqs.md)                                 | Frequently Asked Questions              |
| 🌳 [**Roadmap**](https://github.com/pycaret/pycaret/issues/1756)            | PyCaret's development plan.             |
| :m: [**Meetup**](https://www.meetup.com/pycaret-user-group/)****            | Join our Meetup user group.             |

## Features

PyCaret is an **open-source**, **low-code** machine learning library in Python that aims to reduce the hypothesis to insights cycle time in an ML experiment.  It enables data scientists to perform end-to-end experiments quickly and efficiently. In comparison with the other open-source machine learning libraries, PyCaret is an alternate low-code library that can be used to perform complex machine learning tasks with only a few lines of code. PyCaret is **simple and** **easy to use**.&#x20;

#### PyCaret for Citizen Data Scientists

The design and simplicity of PyCaret is inspired by the emerging role of **citizen data scientists**, a term first used by _Gartner_. Citizen Data Scientists are ‘power users’ who can perform both simple and moderately sophisticated analytical tasks that would previously have required more expertise. Seasoned data scientists are often difficult to find and expensive to hire but citizen data scientists can be an effective way to mitigate this gap and address data science challenges in the business setting.

#### PyCaret deployment capabilities

PyCaret is a deployment ready library in Python which means all the steps performed in an ML experiment can be reproduced using a pipeline that is reproducible and guaranteed for production.  A pipeline can be saved in a binary file format that is transferable across environments.

#### PyCaret is seamlessly integrated with BI

PyCaret and its Machine Learning capabilities are seamlessly integrated with environments supporting Python such as Microsoft Power BI, Tableau, Alteryx, and KNIME to name a few. This gives immense power to users of these BI platforms who can now integrate PyCaret into their existing workflows and add a layer of Machine Learning with ease.

![](<.gitbook/assets/image (506).png>)

#### PyCaret is ideal for:

* Experienced Data Scientists who want to increase productivity.
* Citizen Data Scientists who prefer a low code machine learning solution.
* Data Science Professionals who want to build rapid prototypes.
* Data Science and Machine Learning students and enthusiasts.

## PyCaret at a glance

### Classification

![](<.gitbook/assets/image (86).png>)

### Regression

![](<.gitbook/assets/image (216).png>)

### Clustering

![](<.gitbook/assets/image (98).png>)

### Anomaly Detection

![](<.gitbook/assets/image (243).png>)

### Time Series

PyCaret **new time series module** is now available in the `3.0-rc`. Staying true to the simplicity of PyCaret, it is consistent with the existing API and fully loaded with functionalities. Statistical testing, model training and selection (30+ algorithms), model analysis, automated hyperparameter tuning, experiment logging, deployment on cloud, and more. All of this with only a few lines of code. If you would like to give it a try, check out our official [quick start](https://nbviewer.org/github/pycaret/pycaret/blob/master/examples/time\_series/forecasting/time\_series\_101.ipynb) notebook.

* 📚 [Time Series Docs](https://pycaret.readthedocs.io/en/latest/api/time\_series.html)
* ❓ [Time Series FAQs](https://github.com/pycaret/pycaret/discussions/categories/faqs?discussions\_q=category%3AFAQs+label%3Atime\_series)
* 🚀 [Features and Roadmap](https://github.com/pycaret/pycaret/issues/1648)

To use PyCaret's time series module you can install `PyCaret 3.0-rc`.

```
pip install --pre pycaret
```

## Core Team

The following people are currently core contributors to PyCaret's development and maintenance:

{% embed url="https://github.com/moezali1" %}

{% embed url="https://github.com/Yard1" %}

{% embed url="https://github.com/ngupta23" %}

{% embed url="https://github.com/tvdboom" %}

{% embed url="https://github.com/wkuopt" %}

{% embed url="https://github.com/mfahadakbar" %}

{% embed url="https://github.com/PabloJMoreno" %}

{% embed url="https://github.com/reza1615" %}

{% hint style="info" %}
**Contribute to PyCaret!** If you would like to contribute to this project, please see our [Contribution Guidelines](https://github.com/pycaret/pycaret/blob/master/CONTRIBUTING.md).
{% endhint %}

## Documentation

This documentation is being developed and maintained by:

* [Moez Ali](https://www.linkedin.com/in/profile-moez/)
* [Pablo Moreno](https://www.linkedin.com/in/pablo-moreno-363188150/)

{% hint style="info" %}
**Contribute!** If you would like to join our documentation team and help us build and maintain this awesome documentation for the community, join the **#docs-revamp** channel on our [Slack](https://join.slack.com/t/pycaret/shared\_invite/zt-row9phbm-BoJdEVPYnGf7\_NxNBP307w).
{% endhint %}

## Contributors

![](https://contributors-img.web.app/image?repo=pycaret/pycaret)

## Get Help

* The fastest way to get help is through [our Slack group](https://join.slack.com/t/pycaret/shared\_invite/zt-row9phbm-BoJdEVPYnGf7\_NxNBP307w).&#x20;
* You can also see our [Issue log](./#pycaret-at-a-glance) or [Discussions](https://github.com/pycaret/pycaret/discussions) for answers to questions asked in the past by other members or raise a new question if it's not asked before.
* You can also look out for answers to common questions on [Stack Overflow](https://stackoverflow.com/questions/tagged/pycaret).
* We also have a pretty active [LinkedIn](https://www.linkedin.com/company/pycaret/) page.
* Check our [Frequently Asked Questions](learn-pycaret/faqs.md) (FAQs) page.
* Newsletter coming soon.

## Citation

```
@Manual {PyCaret, 
    title = {PyCaret: An open source, low-code machine learning library in Python}, 
    author = {Moez Ali}, 
    year = {2020}, 
    month = {April}, 
    note = {PyCaret version 1.0}, 
    url = {https://www.pycaret.org
    }
```

## Support us

#### :star:Star **PyCaret** on GitHub  <a href="#star-fastapi-in-github" id="star-fastapi-in-github"></a>

Give us a star on our [GitHub repository](https://github.com/pycaret/pycaret) (click the star button on the top right corner)

#### :link: Follow us on LinkedIn

We have a pretty active [LinkedIn page](https://www.linkedin.com/company/36103360/admin/). Follow us for PyCaret updates and learning content.

#### :person\_tipping\_hand: Help others with issues on GitHub

You can see existing [open issues](https://github.com/pycaret/pycaret/issues) and [discussions](https://github.com/pycaret/pycaret/discussions) and help other members of the community by answering their questions.

#### :tv: Subscribe to our YouTube Channel

Subscribe to our [YouTube](https://www.youtube.com/channel/UCxA1YTYJ9BEeo50lxyI\_B3g) for learning content related to PyCaret.

#### :speaking\_head: Tweet about PyCaret

Help us spread the word. We love to hear success stories and use cases.

#### :writing\_hand: Blog on PyCaret

Like to write? You can write a medium blog and share it with us. Check out this [blog](https://moez-62905.medium.com/) by Author.

#### :m: Join our Meetup <a href="#blog-on-pycaret" id="blog-on-pycaret"></a>

Want to stay in touch and learn what's the latest and greatest in the community. Join our [Meetup](https://www.meetup.com/pycaret-user-group/).

## Sponsors

Placeholder

## Let's get started

All good? Let's explore PyCaret, starting with the [**Installation**](get-started/installation.md#dockerfile).
