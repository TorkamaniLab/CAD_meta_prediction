import xgboost as xgb
from catboost import CatBoostClassifier, CatBoostRegressor
import lightgbm as lgb
# from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
# from sklearn.linear_model import LogisticRegression, LinearRegression


global classifiers, regressors, estimators_dict, params_dict
classifiers = [("xgb", "classifier"), ("lgb", "classifier"), ("cat", "classifier"), ("rf", "classifier"), ("lm", "classifier")]
regressors = [("xgb", "regressor"), ("lgb", "regressor"), ("cat", "regressor"), ("rf", "regressor"), ("lm", "regressor")]


global estimators_dict
estimators_dict = {
    # classifer
    ("xgb", "classifier"): xgb.XGBClassifier(),
    ("lgb", "classifier"): lgb.LGBMClassifier(),
    ("cat", "classifier"): CatBoostClassifier(), 
    # ("rf", "classifier"): RandomForestClassifier(),
    # ("lm", "classifier"): LogisticRegression(),
    
    #regressor
    ("xgb", "regressor"): xgb.XGBRegressor(),
    ("lgb", "regressor"): lgb.LGBMRegressor(),
    ("cat", "regressor"): CatBoostRegressor(),
    # ("rf", "regressor"): RandomForestRegressor(),
    # ("lm", "regressor"): LinearRegression(),
}


global params_dict
params_dict = {
    ("xgb", "classifier"): {
            "objective": ["binary:logistic"],
            "eval_metric": ["auc"],
            "gamma": [1e-8, 10],
            "learning_rate": [0.001, 0.3],
            "max_bin": [2, 20],
            "max_depth": [4, 50],
            "min_child_weight": [0, 15],
            # https://machinelearningmastery.com/tune-number-size-decision-trees-xgboost-python/
            # https://stackoverflow.com/questions/65983344/how-to-choose-the-values-of-n-estimators-and-seed-in-xgbregressor
            "n_estimators": [50, 500], 
            "n_jobs": [-1, -1],
            "random_state": [19920722, 19920722],
            "reg_alpha": [0, 10],
            "reg_lambda": [0, 10],
            "scale_pos_weight": [0.5, 20],  ## 5/95; 95/5
        }, 
    
    ("lgb", "classifier"): {
            "boosting_type": ["gbdt"],
            "num_leaves": [10, 100],
            "learning_rate": [0.001, 0.3],
            "n_estimators": [50, 500],
            "reg_alpha": [0, 10],
            "reg_lambda": [0, 10],
            "random_state": [19920722, 19920722],
            "n_jobs": [-1, -1],
        },
    
    ("cat", "classifier"): {
            "eval_metric": ["Logloss"],
            "random_seed": [19920722, 19920722],
            "depth": [4, 10],
            "loss_function": ["Logloss"],
            "learning_rate": [0.001, 0.3],
            'verbose': [False, False],
        },
    
    ("xgb", "regressor"): {
            "objective": ["reg:squarederror"],
            "gamma": [1e-8, 10],
            "learning_rate": [0.001, 0.3],
            "max_bin": [2, 20],
            "max_depth": [4, 50], 
            "min_child_weight": [0, 15],
            "n_estimators": [50, 500], 
            "n_jobs": [-1, -1],
            "random_state": [19920722, 19920722],
            "reg_alpha": [0, 10],
            "reg_lambda": [0, 10],
            "scale_pos_weight": [0.5, 20],  ## 5/95; 95/5
        }, 
    
    ("lgb", "regressor"): {
            "boosting_type": ["gbdt"],
            "num_leaves": [10, 500], 
            "learning_rate": [0.001, 0.3],
            "n_estimators": [50, 500], 
            "reg_alpha": [0, 10],
            "reg_lambda": [0, 10],
            "random_state": [19920722, 19920722],
            "n_jobs": [-1, -1],
        },
    
    ("cat", "regressor"): {
            "eval_metric": ["RMSE"],
            "random_seed": [19920722, 19920722],
            "depth": [4, 10],
            "loss_function": ["RMSE"],
            "learning_rate": [0.001, 0.3],
            'verbose': [False, False],
    },
}

