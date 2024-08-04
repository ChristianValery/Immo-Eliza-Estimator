import pandas as pd
import numpy as np

import pickle

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, PassiveAggressiveRegressor
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, median_absolute_error
from sklearn.model_selection import cross_val_score, KFold, RandomizedSearchCV, GridSearchCV

from time import time


start = time()


with open("models/features_target_wo.pkl", "rb") as f:
    df, pre, X, y = pickle.load(f)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=17)


# Cross validation and hyper-parameter search.


kf = KFold(n_splits=5, shuffle=True, random_state=42)

param_grid = {"booster": ["gbtree"],
              "objective": ["reg:absoluteerror", "reg:squarederror"],
              "eval_metric": ["mae"],
              "n_estimators": [2000],
              "learning_rate": [0.03, 0.05, 0.2],
              "max_depth": [6]
              }

regressor = XGBRegressor()

regressor_cv = RandomizedSearchCV(regressor, param_grid, cv=kf, n_iter=2)

regressor_cv.fit(X_train, y_train)

cv_results = cross_val_score(regressor_cv, X_train, y_train, cv=kf)


print(f"Best score: {regressor_cv.best_score_}")
print(f"Best parameters: {regressor_cv.best_params_}")
print(f"95% Confindence interval of the score: {np.quantile(cv_results, [0.025, 0.975])}")
print("Training in progress ...")


# Model training with the best parameters.

xgb_regressor = XGBRegressor(**regressor_cv.best_params_, random_state=42)

xgb_regressor.fit(X_train, y_train)

y_pred = xgb_regressor.predict(X_test)


model_metrics = {"class": "Extreme Gradient Boosting (XGBoost)",
                 "Best parameters": regressor_cv.best_params_,
                 "95% Confindence interval of the score": np.quantile(cv_results, [0.025, 0.975]),
                 "R2-score (train)": xgb_regressor.score(X_train, y_train),
                 "R2-score (test)": xgb_regressor.score(X_test, y_test),
                 "Mean Absolute Error": mean_absolute_error(y_test, y_pred),
                 "Root Mean Squared Error": root_mean_squared_error(y_test, y_pred),
                 "Median Absolute Error": median_absolute_error(y_test, y_pred)
                 }


print(f"R2-score (train): {xgb_regressor.score(X_train, y_train)}")
print(f"R2-score (test): {xgb_regressor.score(X_test, y_test)}")
print(f"MAE: {mean_absolute_error(y_test, y_pred)}")
print(f"RMSE: {root_mean_squared_error(y_test, y_pred)}")
print(f"Median Absolute Error: {median_absolute_error(y_test, y_pred)}")

with open("models/predictor_xgbr.pkl", "wb") as f:
    pickle.dump((xgb_regressor, model_metrics), f, protocol = 5)


end = time()

print("Done. Predictor with 'XGBRegressor' trained and stored!")
print(f"It took {round(end - start, 2)} seconds.")