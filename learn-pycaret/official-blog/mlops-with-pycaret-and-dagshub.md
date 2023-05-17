# DagsHub's Integration with PyCaret
With the latest integration between PyCaret and DagsHub, you can log your experiments and artifacts to DagsHub remote servers without making any changes to your code. This includes versioning raw and processed data with DVC and DDA, as well as logging experiment metrics, parameters, and trained models with MLflow. This integration allows you to continue using the familiar MLflow interface while also enabling you to collaborate with others, compare the results of different runs, and make data-driven decisions with ease.

<center><b>To use the integration install pycaret==3.0.0.rc7 and above</b></center>

## What is PyCaret?

PyCaret is an open-source, low-code machine learning library in Python that simplifies the process of training and deploying machine learning models. It offers a wide range of functions and features that make it easy to go from preparing your data to deploying your model within seconds.

<figure>
  <img src="https://dagshub.com/blog/content/images/2023/01/Untitled--29-.png" alt="Alt Text">
  <figcaption>PyCaret Pilers, from the office PyCaret website</figcaption>
</figure>

One of the main advantages of PyCaret is its end-to-end pipeline, which allows you to handle all aspects of your machine learning project within a single, integrated framework. This includes tasks such as data visualization, feature selection, model selection, and model training and deployment. In addition, PyCaret is designed to be low-code, meaning that you can accomplish a lot with just a few lines of code. This makes it accessible to users who may not have extensive coding experience.

## What does the integration between DagsHub and PyCaret include?

PyCaret provides an out-of-the-box integration with MLflow, enabling users to log important metrics, data, and plots on their local machines. This helps to organize the research phase and manage the projects as we move to production. However, it lacks the ability to collaborate with teammates and share results without moving to 3rd party platform (e.g., sending screenshots on Slack). Also, when logging data there is no easy way to see how different processing methods affected the data to make qualitative decisions.

**This is where DagsHub comes into play.**

DagsHub provides a remote MLflow server for each repository, enabling users to log experiments with MLflow and view and manage the results and trained models from the built-in UI. The DagsHUb repository also includes a fully configured object storage to store data, models, and any large file. Those files are diffable, enabling users to see the changes between different versions of their data and models, helping them to understand the impact of those changes on their results.

<figure>
  <img src="https://dagshub.com/blog/content/images/size/w1600/2023/01/Untitled--30-.png" alt="Alt Text">
  <figcaption>Data Diff on DagsHub</figcaption>
</figure>

With the new integration between PyCaret and DagsHub, you can now log experiments to your remote MLflow server hosted on DagsHub, diff experiments and share them with your friends and colleagues. On top of that, you can version your raw and processed data using DVC, push it to DagsHub to view, diff, and share them with others. All these are encapsulated under the new DagsHub Logger that is integrated into PyCaret. This means you need to change ONE line in your code and get all of these (and more) candies, without breaking any sweets.

<figure>
  <img src="https://dagshub.com/blog/content/images/size/w1600/2023/01/Untitled--31-.png" alt="Alt Text">
  <figcaption>Experiment Tracking with DagsHub</figcaption>
</figure>

## How to use DagsHub Logger with PyCaret?

To use the DagsHub Logger with PyCaret, set the `log_experiment` parameter to `dagshub` when initializing your PyCaret experiment. For example:

```python
from pycaret.datasets import get_data
from pycaret.regression import *

data = get_data('diamond')

s = setup(data,
		  target = 'Price',
          transform_target=True,
          log_experiment="dagshub",
          experiment_name='predict_price',
          log_data=True)
```
If the DagsHub Logger is not already authenticated on your local machine, the terminal will prompt you to enter the `repo_owner/repo_name` and provide an authentication link. The repository and remote MLflow server will then be automatically initialized in the background.

## How to use DagsHub Logger programmatically?
To avoid the authentication process, you can set two environment variables which will enable you to run your script programmatically.

```bash
os.environ["DAGSHUB_USER_TOKEN"] = "<enter-your-DagsHub-token>"
os.environ['MLFLOW_TRACKING_URI'] = "<enter-your-MLflow-remote-DagsHub>"
```

The first environment variable will set up you’re DagsHub Token for our Client, which will be used for authentication and write access to the repo and remote. The second is you’re MLflow Tracking URI, hosted on DagsHub. We will use it to set up the remote configuration for the DagsHub Logger. Here is an example of such code:

```python
import os
os.environ["DAGSHUB_USER_TOKEN"] = "<ENTER DAGSHUB TOKEN>"
os.environ['MLFLOW_TRACKING_URI'] = "https://dagshub.com/nirbarazida/pycaret-test.mlflow"

s = setup(data,
		  target = 'Price',
          transform_target=True,
          log_experiment="dagshub",
          experiment_name='predict_price',
          log_data=True)
```

## Conclusion
The new integration between PyCaret and DagsHub makes it easy for you to log experiments, version data, and collaborate with others on machine learning projects. Give the DagsHub Logger a try and see how it can enhance your machine learning workflow with PyCaret. Let us know how it goes on our community [Discord](https://discord.com/invite/skXZZjJd2w) and if you have any enhancements requests - we’d love to enrich this integration and add more capabilities!