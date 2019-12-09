# Basketball Winning Team Prediction
Authors: Xingjia Wang, Stella Huang, Charles Bi

## Introduction

This project sought to predict a basketball team’s winning/losing status given a sizable amount of features. These features included statistics on the home team’s scores and visiting team’s scores, player performance in different positions, minutes played, and specific moves made during the game.

## Methodology

We employed several classification methods: logistic regression, linear discriminant analysis (LDA), extreme gradient boosting (XGBoost), random forest, ridge and lasso regression. In addition to using these methods, we further used feature selection methods to help us select the most significant variables for predicting the outcomes.

## Feature selection

First, we eliminated duplicated variables, where the values were the exact same but had different names, using Pearson’s correlation test. An example of a repeated variable pair is HT.TA.dreb and VT.TS.dreb, where the home team’s team allowed defense rebounds matches the visiting team’s defensive rebounds. We further refined these variables by looking at variables with a correlation = 1 and eliminating one of the variables in such a pair. This yielded us with 156 variables.
With a list of non-repeated features, our group members each used different feature selection methods and compiled our results to see the features with the highest frequency. These methods, implemented in R or Python, included: correlation test, LASSO, ridge, best subset with the forward and backward selection, stability selection, randomized logistic regression, Boruta, VIF, and random forest. We ran the models mentioned above with the most selected features. The training set was further split into a training and testing set so that we could validate the model’s accuracy after constructing it. We tested different subsets, using their frequencies as a cutoff, of variables and selected the subset with the highest accuracy on the test set.

## Time series ridge model

The dataset, train.csv, contains game statistics from years 2001 to 2012, while test.csv contained game statistics from 2013 to 2014. We found out that features selected from the above procedure did not include the date, while we had to predict future outcomes. In order to take the effect of time into account, we employed a time-based resampling method to perform cross validation for the model.

We first performed a train-test split by randomly sampling 20-25% of train.csv to use as a test set for model evaluation. Then we sorted all observations in the train set by date and chose the first 1800 observations to be the initial model-training set. For each model-training iteration, instead of randomly resampling our validation set, we chose our validation set to be the immediate 600 observations sequential to the model-training set. Then we added the validation set to the current model-training set to perform the next round of the train-validation process. This procedure is done by customizing trainControl() from the caret package. We also performed a hyperparameter tuning on lambda (from 0 to 1000 by 20) and selected the best-fit model to predict the test set. The prediction accuracy on the test set ranges from 65-68%.

## Conclusion & Discussion

With the selected features, the ridge time series model’s prediction achieved an average 67% accuracy in cross-validation. Even though the model has an overall strong predictive power, there is one remaining concern. Looking retrospectively, while the model was able to predict winning games with an accuracy as high as 80%, it doesn’t capture the losses well, meaning that the type II error is high. The private scoreboard shows that the multi-layer perceptron classifier (MLP), LDA, and LASSO yielded the highest accuracy. For the top 6 models, we used the most common variables from the feature selection methods. We believe that using our approach of selecting variables from multiple feature selection methods was correct, yet our reasoning for selecting final submissions was not. To validate the models, later on, we split the train data into a train and test set, then selected the models that yielded a relatively high test accuracy. Even though these models had a relatively high accuracy on the training set and public leaderboard, there is a possibility that the models overfitted the data.

In conclusion, the final ridge regression model could explain the variance of the response variable up to 67.354% with 26 variables combined. Despite its limitations, the model sheds an interesting light into which factors have the most predictive powers towards the response variable, when plus-minus of small forwards and guards can determine a game outcome. The results can provide basketball managers and coaches with handy information on how to optimize the team’s performance.
