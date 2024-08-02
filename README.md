

<h1 align="center"> Immo-Eliza-Estimator </h1> <br>
<p align="center">
  <a href="https://immo-eliza-estimator.onrender.com/">
    <img alt="Immo-Eliza-Estimator" title="Immo-Eliza-Estimator" src="images/real_estate_agent.png" width="100">
  </a>
</p>

<p align="center">
  A web application for individuals and real estate agents.
</p>



<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
## Table of Contents

- [Introduction](#introduction)
- [Description](#description)
- [Modeling](#modeling)
- [Deployment](#deployement)
- [Technology](#technology)
- [Acknowledgments](#acknowledgments)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## Introduction

Price predictors are one of the main types of tools that can be designed with artificial intelligence and machine learning. As part of our training at BeCode, we had the opportunity to design such a model and deploy it as a web application.

Here we wish to share the results of this enriching and fruitful experience.


## Description

The project was organised in four phases:

1. Web scraping;
2. Data visualization;
3. Modeling (Machine learning);
4. Deployment.

The first phase consisted of scraping the website of the famous company [**immoweb**](https://www.immoweb.be/en) to create a relevant dataset of real estate properties in Belgium. The data collected were stored in a structured manner as a json file.

In the second phase of the project, the json file was conveted into a pandas DataFrame. The latter was then cleaned to perform an exploratory data analysis and to make insightful visualizations. This work allowed us to have a better understanding of the database and the field of real estate. So we were equipped for modeling.

The challenge of the third phase was to study different machine learning models, choose the best one for the situation, and trained it.

The aim of the fourth and final phase was to design and deploy a basic web application using the previously trained model.

In the following, we will detail the work carried out in the last two phases of the project.


## Modeling

The modeling phase started with a raw DataFrame consisting of 118714 rows and 32 columns. Before the modeling itself, we have the following preprocessing operations.

1. **Data cleaning** (data validation, addressing missing data).

2. **Features engineering**:

    - *Lasso regression* for feature selection;

    - *ColumnTransformer*, *RobustScaler*, *OneHotEncoder* and *OrdinalEncoder* for feature transformation.


After the preprocessing step, we tried a model with multilinear regression, as well as other linear models such as Ridge and Lasso regressions. But, the *R2-score* did not exceed 0.5 and the *Mean Absolute Error* was above 150,000 euros. We then turned to scikit-learn **ensemble methods** and to **Extreme Gradient Boosting Regressor (XGBoost)**.

We used *cross-validation* without *hyperparameter tuning* to evaluate and compare the models. With this in mind, the diagram below provides an overview of the distribution of the R2-score of the following four models:

1. Gradient Boosting Regressor (GBRegressor);

2. Histogram Gradient Boosting Regressor (HGBRegressor); 

3. Extreme Gradient Boosting Regressor (XGBRegressor);

4. Random Forest Regressor.


<p align="center">
  <img src = "images/regressors_comparison.png" width=600>
</p>



## Deployment



## Technology

The main tools we used in this project are the following.

<a href="https://www.python.org" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" alt="python" width="40" height="40"/> </a>

<a href="https://pandas.pydata.org/" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/2ae2a900d2f041da66e950e4d48052658d850630/icons/pandas/pandas-original.svg" alt="pandas" width="40" height="40"/></a>
  
  <a href="https://scikit-learn.org/" target="_blank" rel="noreferrer"> <img src="https://upload.wikimedia.org/wikipedia/commons/0/05/Scikit_learn_logo_small.svg" alt="scikit_learn" width="40" height="40"/>
  
  </a> <a href="https://seaborn.pydata.org/" target="_blank" rel="noreferrer"> <img src="https://seaborn.pydata.org/_images/logo-mark-lightbg.svg" alt="seaborn" width="40" height="40"/> </a> </p>


## Acknowledgments

All my thanks to my coaches at BeCode, Antoine and Denis, for the flawless coaching. Thank you also to my fellow trainees for their support and for the many sessions of mutual enrichment.