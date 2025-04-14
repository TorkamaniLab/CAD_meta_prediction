#!/usr/bin/env python3

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, CatBoostRegressor

# classifiers = [("xgb", "classifier"), ("lgb", "classifier"), ("cat", "classifier")]
# regressors = [("xgb", "regressor"), ("lgb", "regressor"), ("cat", "regressor")]

ESTIMATORS_DICT = {
    # classifer
    ("xgb", "classifier"): xgb.XGBClassifier(),
    ("lgb", "classifier"): lgb.LGBMClassifier(),
    ("cat", "classifier"): CatBoostClassifier(),
    # regressor
    ("xgb", "regressor"): xgb.XGBRegressor(),
    ("lgb", "regressor"): lgb.LGBMRegressor(),
    ("cat", "regressor"): CatBoostRegressor(),
}

PARAMS_DICT = {
    # classifier
    ("xgb", "classifier"): {
        "objective": ["binary:logistic"],
        "eval_metric": ["auc"],
        "gamma": [0, 8],
        "learning_rate": [0.001, 0.3],
        "max_bin": [2, 20],
        "max_depth": [3, 15],
        "min_child_weight": [1, 12],
        # https://machinelearningmastery.com/tune-number-size-decision-trees-xgboost-python/
        # https://stackoverflow.com/questions/65983344/how-to-choose-the-values-of-n-estimators-and-seed-in-xgbregressor
        "n_estimators": [100, 400],
        "n_jobs": [-1, -1],
        "random_state": [19920722, 19920722],
        "reg_alpha": [0, 5],
        "reg_lambda": [0, 5],
        "scale_pos_weight": [0.05, 20],  ## 95/5 = 19, 5/95 = 0.05
    },
    ("lgb", "classifier"): {
        "boosting_type": ["gbdt"],
        "num_leaves": [20, 80],
        "learning_rate": [0.001, 0.3],
        "n_estimators": [100, 400],
        "reg_alpha": [0, 5],
        "reg_lambda": [0, 5],
        "random_state": [19920722, 19920722],
        "n_jobs": [-1, -1],
        "verbose": [-1, -1],
    },
    ("cat", "classifier"): {
        "eval_metric": ["Logloss"],
        "random_seed": [19920722, 19920722],
        "depth": [4, 12],
        "loss_function": ["Logloss"],
        "learning_rate": [0.001, 0.3],
        "verbose": [False, False],
    },
    # regressor
    ("xgb", "regressor"): {
        "objective": ["reg:squarederror"],
        "gamma": [0, 8],
        "learning_rate": [0.001, 0.3],
        "max_bin": [2, 20],
        "max_depth": [3, 15],
        "min_child_weight": [1, 12],
        "n_estimators": [100, 400],
        "n_jobs": [-1, -1],
        "random_state": [19920722, 19920722],
        "reg_alpha": [0, 5],
        "reg_lambda": [0, 5],
        "scale_pos_weight": [0.05, 20],
    },
    ("lgb", "regressor"): {
        "boosting_type": ["gbdt"],
        "num_leaves": [20, 80],
        "learning_rate": [0.001, 0.3],
        "n_estimators": [100, 400],
        "reg_alpha": [0, 5],
        "reg_lambda": [0, 5],
        "random_state": [19920722, 19920722],
        "n_jobs": [-1, -1],
        "verbose": [-1, -1],
    },
    ("cat", "regressor"): {
        "eval_metric": ["RMSE"],
        "random_seed": [19920722, 19920722],
        "depth": [4, 12],
        "loss_function": ["RMSE"],
        "learning_rate": [0.001, 0.3],
        "verbose": [False, False],
    },
}
