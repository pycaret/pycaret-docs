# Predicting Crashes in Gold Prices Using PyCaret

#### [Gold Prediction](https://towardsdatascience.com/tagged/gold-price-prediction)

### Predicting Crashes in Gold Prices Using Machine Learning

#### by Mohammad Riazuddin

#### Part — III of Gold Prediction Series. Step by step guide to predict a crash in Gold prices using Classification with PyCaret

![Gold Price Movement](https://cdn-images-1.medium.com/max/3666/0\*etQrkJsMtNUP9EWE.png)

### Approach

In the previous two parts of the [Gold Prediction](https://towardsdatascience.com/machine-learning-to-predict-gold-price-returns-4bdb0506b132) series, we discussed how to import data from free yahoofinancials API and build a regression model to predict return from Gold over two horizons. i.e 14-Day and 22-Day.

In this part, we will try to predict if there would be a ‘sharp fall’ or ‘crash’ in Gold Prices over next 22-Day period. We would be using a classification technique for this experiment. We will also learn how to use the trained model to make a prediction on new data every day. The steps for this exercise would be:

1. \*\*Import and shape the data \*\*— This would be similar to what was explained in Part-I here. You can also download the final dataset from my git [repo](https://github.com/Riazone/Gold-Return-Prediction/blob/master/Training%20Data.csv).
2. **Define **_**‘sharp fall’**_ in Gold Prices. Sharp is not an absolute measure. We will try to define _‘sharp fall’_ objectively
3. \*\*Create Label \*\*— Based on the definition of _‘sharp fall’_, we would create labels on historical data
4. Train models to predict the \*‘sharp fall’ \*and use trained model to make prediction on new data

### Preparing Data

This part is exactly the same as we did in Part I. The [notebook](https://github.com/Riazone/Gold-Return-Prediction/blob/master/Classification/Gold%20Prediction%20Experiment%20%20Classification-%20PyCaret.ipynb) contains the entire code chunk for data import and manipulation or you can directly start by loading the dataset which can be downloaded from the link [here](https://github.com/Riazone/Gold-Return-Prediction/blob/master/Training%20Data.csv).

### Defining ‘Sharp Fall’

Any classification problem needs labels. Here we need to create labels by defining and quantifying _‘sharp fall’._

To define ‘sharp’, I am defining a threshold such that the probability of returns being lower than the threshold for any window (22 Days here and 14 Days here) is 15% (basically left tail of normal distribution with p=0.15). For this, I will need to assume that the distribution of returns is normal. Looking at the distribution of returns, this is a very reasonable assumption to make.

![](https://cdn-images-1.medium.com/max/2000/1\*hjdmU6aYxcPQfqQ0QncN5w.png)

![Histogram of Returns for both 14 and 22 Day window. Very close to Normal Distribution](https://cdn-images-1.medium.com/max/2000/1\*LiI4fq4eFHkKzvdXamWfxA.png)

To get to the threshold return level for both the windows (14-Day and 22-Day), I would first define the p-value of the left tail of the distribution, which would be 15% in this case. Using this p-value, we get the z-value of the -1.0364 from a standard normal distribution. The following code would do it for us.

```
import scipy.stats as st
#Select Threshold p (left tail probability)
p= 0.15
#Get z-Value
z = st.norm.ppf(p)
print(z)
```

Now, based on the above z-value and Mean and SD of returns for each window, we will get threshold return levels. The forward returns for 14 Day and 22 Day period are in columns “Gold-T+14” and “Gold-T+22” in the ‘data’.

```
#Calculating Threshold (t) for each Y
t_14 = round((z*np.std(data[“Gold-T+14”]))+np.mean(data[“Gold-T+14”]),5)
t_22 = round((z*np.std(data[“Gold-T+22”]))+np.mean(data[“Gold-T+22”]),5)

print(“t_14=”,t_14)
print(“t_22=”,t_22)

t_14= -0.03733
t_22= -0.04636
```

So the threshold return levels are -0.0373 or -3.73% for 14-Day window and -0.0463 or -4.63% for 22-Day window. What this implies is, there is only 15% probability of 14-Day returns being lower than -3.73% and 22-Day return being lower than -4.63%. This is a concept similar to what is used in the calculation of Value At Risk (VAR).

### Creating Labels

We will use the above threshold levels to create labels. Any returns in the two windows lower than the respective threshold will be labeled as 1, else as 0.

```
#Creating Labels
data[‘Y-14’] = (data[‘Gold-T+14’]< t_14)*1
data[‘Y-22’]= (data[‘Gold-T+22’]< t_22)*1
print(“Y-14”, sum(data[‘Y-14’]))
print(“Y-22”, sum(data[‘Y-22’]))

Y-14 338
Y-22 356
```

Out of total 2,379 instances, there have been 338 instances where 14-Day returns have been lower than the threshold of -3.73% and 356 instances for 22-Day when returns have been lower than threshold of -4.63%.

Once we have these labels, we do not actually need the returns columns and hence we delete the actual returns columns.

```
data = data.drop([’Gold-T+14',’Gold-T+22'],axis=1)
data.head()
```

### Modelling with PyCaret

#### 22-Day Window

We will start with the 22-Day window here. I will be using PyCaret’s Classification module here for experimentation.

We import the module above form PyCaret and then delete the label for 14-Day since we are working with 22-Day window here. Just like in Regression, to begin the classification exercise, we need to run _setup()_ command to point out the data and target column. Remember all the basic pre-processing is taken care of by PyCaret in the background.

```
from pycaret.classification import *

data_22 = data.drop([’Y-14'],axis=1)
s22 = setup(data=data_22, target=’Y-22', session_id=11, silent=True);
```

To evaluate the set of all the models, we will run the \*compare\_models() \*command with _turbo_ set to _**False**_ because I want to evaluate all the models currently available in the library.

```
compare_models(turbo=False)
```

![Compare Models Output](https://cdn-images-1.medium.com/max/2832/1\*Wmvq7DeuVtmcMdkpz4B94w.png)

Before proceeding with the selection of models, we need to understand which metric is most valuable to us. Choosing a metric in a classification experiment depends on the business problem. There is always a trade-off between Precision and Recall. This means we have to choose and favor a balance between True Positives and False Negatives.

Here, the final model will be used to create a flag for the investor/analyst warning him about the possibility of an impending crash. The investor will then make the decision to hedge his position against the possible fall. Therefore, it is very important that the model is able the predict all/most of the drastic falls. In other words, we want to choose a model with a better ability to have True Positives (better Recall), even if it comes with the cost of some False Positives (lower Precision). In other words, we do not want the model to miss the possibility of \*‘sharp fall’. \*We can afford to have some False Positive because if the model predicts that there will be a sharp fall, and the investor hedges his position, but the fall does not occur, the investor will lose opportunity cost of remaining invested or at most hedge cost (say if he buys out of money Put Options). This cost will be lower than the cost of false negative where the model predicts no _‘Sharp Fall’_, but a massive fall does happen. We need to however keep a tab on the trade-offs in \*\*Precision \*\*and **AUC**.

We will go ahead and create four models namely MLP Classifier (_mlp)_, Extra-Tree Classifier (_et)_, Cat Boost Classifier (_catb_) and Light Gradient Boosting Machine (_lgbm_) with best \*\*Recall \*\*and reasonable **AUC/Precision.**

![](https://cdn-images-1.medium.com/max/2000/1\*OUz2ZUmbZLTYwgFn0m3Hvw.png)

![](https://cdn-images-1.medium.com/max/2000/1\*AFDkbDD6Mxv7YDz1G\_5rbg.png)

![](https://cdn-images-1.medium.com/max/2000/1\*kJUEp\_A989tlTGHiaFsz2g.png)

![](https://cdn-images-1.medium.com/max/2000/1\*LYnj\_G6F7yx4x5JmYOdcxA.png)

Based on the results we have, MLP Classifier appears to be the best choice with the highest recall and very decent _**AUC of 94.7%**_.

**Hyper-Parameter Tuning**

Once we have the top four models which we would want to pursue further, we need to find the best hyperparameters for the models. PyCaret has a very convenient function of \*\*\*tune\_model() \*\*\*which loops through pre-defined hyper-parameter grids to find the best parameters for our model through 10 fold cross-validation. PyCaret uses the standard _**Randomized-Grid**_ search to iterate through the parameters. The number of iterations (_n\_iter_) can be specified to a high number based on compute capacity and time constrain. Within the _**tune\_function()**_, PyCaret also allows us to specify the metric we want to optimize. The default is Accuracy, but we can choose other metrics as well. Like we would choose **Recall** here since it is the metric we would want to increase/optimize.

![CatBoost Tuned Output](https://cdn-images-1.medium.com/max/2000/1\*D8\_Opw0chItS8Z1HB8OCVA.png)

The code above tunes the Cat-Boost classifier to optimize \*\*‘Recall’ \*\*by iterating 50 times over the defined grid and displays the 6 metrics for each fold. We see that **Mean Recall** has improved from _**58.2%**_ in base Cat-Boost to _**62.6%**_ here. This is a massive jump, not so common at the tuning stage. However, it still remains lower than _**66.6%**_ of base _**‘mlp**_’ that we created earlier. For the other three models, we did not see improvement in performance by tuning (example in the [notebook](https://github.com/Riazone/Gold-Return-Prediction/blob/master/Classification/Gold%20Prediction%20Experiment%20%20Classification-%20PyCaret.ipynb)). The reason being the random nature of iterating through the parameters. There however would exist a parameter that would equal or exceed the performance of the base model, but for that, we need to increase _n\_iter_ even more, which means more compute time.

So currently our top-4 models are:

![](https://cdn-images-1.medium.com/max/2000/1\*ok58GJ9EY2YGkhRrhBdpVg.png)

#### **Evaluate Models**

Before moving ahead, let us evaluate the model performance. We will exploit _**evaluate\_model()**_ function of PyCaret to evaluate and highlight important aspects of the winning models.

**Confusion Matrix**

![](https://cdn-images-1.medium.com/max/2000/1\*W7dfGY1PnVEepgq9Y4LzgA.png)

**Feature Importance**

Since _mlp_ and _catb\_tuned_ do not provide feature importance, we will use \*lgbm \*to see which features are most important in our predictions:

![](https://cdn-images-1.medium.com/max/2000/1\*k9IZNrOlsfgiT4jm27HXOA.png)

We can see that Return of Gold in the past 180 days is the most important factor here. This is also intuitive because if the Gold prices have increased a lot in the past, their chances of falling/correcting are higher and vice-versa. The next 3 features are returns from Silver over 250, 60 and 180 days. Again, Silver and Gold are two most widely traded and correlated precious metals, hence the relationship is very intuitive.

#### **Ensemble Models**

After having tuned the hyper-parameters of the model, we can try Ensembling methods to improve performance. Two ensembling methods we can try is [‘Bagging’ and ‘Boosting’](https://towardsdatascience.com/ensemble-methods-bagging-boosting-and-stacking-c9214a10a205) . Models that do not provide probability estimates cannot be used for Boosting. Hence, we can use Boosting with ‘lgbm’ and ‘et’ only. For others, we tried bagging to see if there were any improvements in performance or not. Below are the code snapshots and the 10-fold results.

![](https://cdn-images-1.medium.com/max/2000/1\*lknUUSTHIjRxQ8s0\_BRohQ.png)

![Ensembling Results](https://cdn-images-1.medium.com/max/2000/1\*aGIg3u79djyiWCE9poQ-BA.png)

As we can see above, the results did not improve for the two models. For other models as well, there was deterioration in performance (check [notebook](https://github.com/Riazone/Gold-Return-Prediction/blob/master/Classification/Gold%20Prediction%20Experiment%20%20Classification-%20PyCaret.ipynb)). Hence our winning model remains the same.

**Blending Models**

Blending models is basically building a voting classifier on top of estimators. For models that provide prediction probability, we can use soft voting (using their probabilities), while for others, we would use hard-voting. The _**blend\_model()**_ function defaults to using hard voting, which can be changed manually. I built two blends to see if there was some additional performance that can be extracted.

![](https://cdn-images-1.medium.com/max/2000/1\*UwqjMhfu0JtQJKfCOV1VYw.png)

![](https://cdn-images-1.medium.com/max/2000/1\*dokZi-hH-1jL2ifHmyAjow.png)

Though none of the models could dethrone _**‘mlp’**_ from top position, it is very interesting to see the second blend, _**‘blend2’**_, which is a soft combination of _**‘lgbm’**_ and _**‘et’**_. The performance on **Recall** and **AUC** of **62.25%** and **97.43%** is higher than both _**‘lgbm’**_ and _**‘et’**_ individually. This exhibits the benefit of blending models. Now our winner models would be reduced to 3.

![](https://cdn-images-1.medium.com/max/2000/1\*owcSPrycYFxoW9KrfrJ96g.png)

[\*\*Creating StackNet](https://www.coursera.org/lecture/competitive-data-science/stacknet-s8RLi)\*\*

[Stacking models](https://www.geeksforgeeks.org/stacking-in-machine-learning/) is a method where we allow predictions of one models (or set of models) in one layer to be used as a feature for subsequent layers and finally, the meta-model is allowed to train on the predictions of previous layers and the original features (if _restack=True_) to make the final prediction. PyCaret has a very easy implementation of this where we can build a stack with one layer and one meta-model using _**stack\_model()**_ or multiple layers and one meta-model using _**create\_stacknet()**_. I have used a different combination to build different stacks and evaluate performance.

![Results of Stack-1](https://cdn-images-1.medium.com/max/2000/1\*d6PR6Hj2ZYrBFAifh-KSvw.png)

In the first stack, _**‘stack1’**_, I used the under-performing models, _**catb\_tuned**_ and _**blend2**_ in the first layer to pass on their predictions to the leader model _**mlp**_ which would help it to make predictions and the prediction of _**mlp**_ are used by meta-model ,which is \*\*Logistic-Regression (LR) \*\*here by default, to make the final prediction. Since _**LR**_ did not do very well with complete data (see compare models results) I used _restack=False_, which means only the predictions from previous models gets passed to subsequent estimators, not the original features. What we see here is nothing short of magic. The **Recall** jumps from **66.63%** of our best model _**‘mlp’**_ to massive **77.8%** and so does **AUC** and **Precision**. _**‘stack1’**_ is definitely far superior to all the models we built earlier. Obviously I had to try other configurations as well to see if better performance can be achieved. And I did find a better configuration:

![Results of Stack 3](https://cdn-images-1.medium.com/max/2000/1\*n5ZY0N\_c4kBaSVpxn78siw.png)

The stack above, _**‘stack3**_, had most resounding success with **Recall** averaging **80.7%**, i.e **14%** higher than our winner _**‘mlp’**_ model without sacrificing on **Accuracy** (_95%, an improvement of 3%_), **AUC** (_97.34%, an improvement of 2.5%_) or **Precision** (_86.28%, improvement of massive 7.7%_)

We can use this model to predict on the Test-Data, _(30% of the total observations)_ that we separated at the setup stage. We can do that by using _predict\_model()_

![](https://cdn-images-1.medium.com/max/2000/1\*lJGUuf\_sa1xsaaCB58QG0w.png)

We can see that the performance of the model on test data is even better with \*\*Recall \*\*of **86.9%**

Since we have _**‘stack3’**_ as the leading model, we would fit the model on the entire data (including the Test data) and save the model for prediction on new data.

```
classifier_22 = finalize_model(stack3)

save_model(classifier_22,”22D Classifier”)
```

The code above fits the model on entire data and using the _**save\_model()**_ function, I have saved the trained model and the pre-processing pipeline in a _pkl_ file named [“_**22D Classifier**_”](https://github.com/Riazone/Gold-Return-Prediction/blob/master/Classification/22D%20Classifier.pkl) in my active directory. This saved model can and will be called on to predict on new data.

#### Predicting on New Data

For making prediction we will need to import the raw prices data just the way we did it the beginning of the exercise to extract the features and then load the model to make the predictions. The only difference would be that we would be importing data till the last trading day to make a prediction on the most recent data, just like we would do in real life. The notebook titled _**‘**_[_**Gold Prediction New Data — Classification’**_](https://github.com/Riazone/Gold-Return-Prediction/blob/master/Classification/Gold%20Prediction%20New%20Data%20-%20Classification.ipynb) in the repo exhibits the data import, preparation and prediction codes.

We will skip the data import and preparation part (check notebook for details) and see the prediction process.

```
from pycaret.classification import *
***#Loading the stored model***
classifier_22 = load_model(“22D Classifier”);
***#Making Predictions***
prediction = predict_model(classifier_22,data=prediction_data)
```

Using _**predict\_model()**_ we can apply the loaded model on new set of data to generate the prediction (1 or 0) and the score (the probability attached to the prediction). The _‘prediction’_ dataframe will also contain all the features that we extracted.

![Prediction Tail](https://cdn-images-1.medium.com/max/2000/1\*MW9Qm\_DdhnQFonTPD-meeg.png)

Looking at the Label and Score column of the prediction, the model did not predict a significant fall on any of the days. For example, given the historical returns as of 29th April, the model predicts that a significant fall in Gold Prices over the next 22-Day period is _**not likely**_, hence **Label = 0**, with a probability of a meagre 9.19%.

### Conclusion

So here I have walked through steps to create a classifier to predict a significant fall in prices over the next 22-Day period in Gold Prices. The notebook contains codes for 14- Day model as well. You can try to create labels and try to predict similar fall over different windows of time in a similar fashion. Till now, we have created a [regression](https://towardsdatascience.com/machine-learning-to-predict-gold-price-returns-4bdb0506b132) and a classification model. In future, we will try to use the prediction from classification models as features in the regression problem and see if it improves the performance of regression.

### Important Links

[\*\*\*Git-hub Repository](https://github.com/Riazone/Gold-Return-Prediction/tree/master/Classification)\*\*\*

[\*\*\*Part-I and II — Regression](https://towardsdatascience.com/machine-learning-to-predict-gold-price-returns-4bdb0506b132)\*\*\*

[\*\*\*PyCaret](https://pycaret.org)\*\*\*

[\*\*\*My LinkedIn profile](https://www.linkedin.com/in/riazuddin-mohammad/)\*\*\*
