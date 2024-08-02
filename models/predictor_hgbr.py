import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import pickle

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, PassiveAggressiveRegressor
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, median_absolute_error
from sklearn.model_selection import cross_val_score, KFold, RandomizedSearchCV, GridSearchCV

from time import time


start = time()

with open("features_target_wo.pkl", "rb") as f:
    df, pre, X, y = pickle.load(f)


# Cross validation and hyper-parameters search.


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=17)


kf = KFold(n_splits=5, shuffle=True, random_state=42)

param_grid = {"loss": ["absolute_error", "squared_error"],
              "learning_rate": np.arange(0.1, 1, 10),
              "max_iter": np.arange(100, 11000, 10),
              "max_leaf_nodes": np.arange(31, 110, 10)
              }

regressor = HistGradientBoostingRegressor()

regressor_cv = RandomizedSearchCV(regressor, param_grid, cv=kf, n_iter=6)

regressor_cv.fit(X_train, y_train)

cv_results = cross_val_score(regressor_cv, X_train, y_train, cv=kf)



print(f"Best score: {regressor_cv.best_score_}")
print(f"Best parameters: {regressor_cv.best_params_}")
print(f"95% Confindence interval of the score: {np.quantile(cv_results, [0.025, 0.975])}")
print("Training in progress ...")


# Model training with the best parameters.


hgb_regressor = HistGradientBoostingRegressor(**regressor_cv.best_params_, random_state=42)

hgb_regressor.fit(X_train, y_train)

y_pred = hgb_regressor.predict(X_test)


model_metrics = {"class": "Histogram Gradient Boosting Regressor",
                 "Best parameters": regressor_cv.best_params_,
                 "95% Confindence interval of the score": np.quantile(cv_results, [0.025, 0.975]),
                 "R2-score (train)": hgb_regressor.score(X_train, y_train),
                 "R2-score (test)": hgb_regressor.score(X_test, y_test),
                 "Mean Absolute Error": mean_absolute_error(y_test, y_pred),
                 "Root Mean Squared Error": root_mean_squared_error(y_test, y_pred),
                 "Median Absolute Error": median_absolute_error(y_test, y_pred)
                 }

print(f"R2-score (train): {hgb_regressor.score(X_train, y_train)}")
print(f"R2-score (test): {hgb_regressor.score(X_test, y_test)}")
print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred)}")
print(f"Root Mean Squared Error: {root_mean_squared_error(y_test, y_pred)}")
print(f"Median Absolute Error: {median_absolute_error(y_test, y_pred)}")



with open("predictor_hgbr.pkl", "wb") as f:
    pickle.dump((hgb_regressor, model_metrics), f, protocol=5)


end = time()

print("Done. Predictor with 'HistGradientBoostingRegressor' trained and stored!")
print(f"It took {round(end - start, 2)} seconds.")