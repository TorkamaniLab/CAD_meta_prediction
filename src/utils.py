#!/usr/bin/env python3

import random
import time
import warnings
from pathlib import Path

import category_encoders as cen
import numpy as np
import pandas as pd
from joblib import dump
from lohrasb.best_estimator import BaseModel
from optuna.pruners import HyperbandPruner
from optuna.samplers._tpe.sampler import TPESampler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import classification_report, confusion_matrix, f1_score, r2_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from zoish.feature_selectors.shap_selectors import ShapFeatureSelector

from src.hyperparams import ESTIMATORS_DICT, PARAMS_DICT

random.seed(19920722)
np.random.seed(19920722)
warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")


class ColumnSelector(BaseEstimator, TransformerMixin):
    # https://towardsdatascience.com/customizing-sklearn-pipelines-transformermixin-a54341d8d624
    def __init__(self, selected_cols):
        if not isinstance(selected_cols, list):
            self.selected_cols = [selected_cols]
        else:
            self.selected_cols = selected_cols

    def fit(self, X: pd.DataFrame, y: pd.Series):
        # there is nothing to fit
        return self

    def transform(self, X: pd.DataFrame):
        X = X.copy()
        return X[self.selected_cols]


def _get_acc_metric(y, y_pred, trainer, classifier_metric="F1"):
    if trainer[1] == "classifier":
        try:
            if classifier_metric == "AUC":
                auc = roc_auc_score(y, y_pred)
                print(f"AUC score : {auc}")
                out = auc
            else:
                pred_labels = np.rint(y_pred)
                f1 = f1_score(y, pred_labels, average="macro")
                print(f"F1 score : {f1}")
                print(f"Classification report : ")
                print(classification_report(y, pred_labels))
                print(f"Confusion matrix : ")
                print(confusion_matrix(y, pred_labels))
                out = f1
        except Exception as e:
            print(f"Exception during metric calculation: {e}")
            out = "None"
    else:
        r2 = r2_score(y, y_pred)
        print(f"r2 score : {r2}")
        out = r2
    return out


def _set_opt_params(trainer):
    # set classification / regression
    if trainer[1] == "classifier":
        measure_of_accuracy = f"f1_score(y_true, y_pred, average='macro')"
        print(f"measure_of_accuracy = {measure_of_accuracy}")
        with_stratified = True
    else:
        measure_of_accuracy = f"r2_score(y_true, y_pred)"
        with_stratified = False
    return measure_of_accuracy, with_stratified


def _make_lohrasb_obj(y_train, trainer, n_trials, measure_of_accuracy, study_name, with_stratified):
    estimator = ESTIMATORS_DICT[trainer]
    estimator_params = PARAMS_DICT[trainer]

    stratify = y_train[y_train.columns.to_list()[0]] if with_stratified else None

    kwargs = {
        "fit_optuna_kwargs": {"sample_weight": None},
        "main_optuna_kwargs": {
            "estimator": estimator,
            "estimator_params": estimator_params,
            "refit": True,
            "measure_of_accuracy": measure_of_accuracy,
        },
        "train_test_split_kwargs": {"stratify": stratify, "test_size": 0.10},
        "study_search_kwargs": {
            "storage": None,
            "sampler": TPESampler(seed=19920722),
            "pruner": HyperbandPruner(),
            "study_name": study_name,
            "direction": "maximize",
            "load_if_exists": False,
        },
        "optimize_kwargs": {
            "n_trials": n_trials,
            "timeout": 3600,
            "n_jobs": 1,
            "catch": (),
            "callbacks": None,
            "gc_after_trial": True,
            "show_progress_bar": False,
        },
    }

    lohrasb_opt = BaseModel().optimize_by_optuna(kwargs=kwargs)

    return lohrasb_opt


def _execute_zoish_shap_selector(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    trainer: tuple[str, str],
    n_trials: int,
    n_features: int,
    measure_of_accuracy: str,
    with_stratified: bool,
    study_name: str,
    feature_list: list | None = None,
    shap_n_jobs: int = -1,
    approximate: bool = False,
):
    # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html
    # https://scikit-learn.org/stable/modules/model_evaluation.html
    if trainer[1] == "classifier":
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=19920722)
        cv_scoring = "f1_macro"
    else:
        cv = KFold(n_splits=5, shuffle=True, random_state=19920722)
        cv_scoring = "r2"

    # Initiate from all columns if no assigned
    if feature_list is None:
        feature_list = X_train.columns.tolist()

    # Form sklearn pipeline
    # int_cols = X_train.select_dtypes(include=['int']).columns.tolist()
    # float_cols = X_train.select_dtypes(include=['float']).columns.tolist()
    cat_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
    transformers = []
    if len(cat_cols) > 0:
        transformers.append(("CatBoostEncoder", cen.CatBoostEncoder(random_state=19920722)))
    lohrasb_opt = _make_lohrasb_obj(y_train, trainer, n_trials, measure_of_accuracy, study_name, with_stratified)
    pre_shap_pip = Pipeline(
        [("ALLColumn", ColumnSelector(feature_list))]
        + transformers
        + [(f"LohrasbOptimizer_{trainer[0]}_{trainer[1]}", lohrasb_opt)]
    )

    print(f"\npre-SHAP init:")
    print(f"- X_train.shape:{X_train.shape}")
    print(f"- y_train.shape{y_train.shape}")
    start_time = time.time()
    pre_shap_pip.fit(X_train, y_train)
    pre_SHAP_best_estimator = pre_shap_pip[-1].best_estimator
    end_time = time.time()
    print(f"pre-SHAP done - {end_time - start_time} seconds")

    # Use the best estimator for SHAP feature selection
    shap_feature_selector = ShapFeatureSelector(
        model=pre_SHAP_best_estimator,
        num_features=n_features,
        scoring=cv_scoring,
        direction="maximum",
        cv=cv,
        n_iter=5,
        algorithm="auto",
        random_state=19920722,
        use_faster_algorithm=True,
        shap_fast_tree_explainer_kwargs={
            # "model_output": "raw", # default
            "algorithm": "v2",
            "feature_perturbation": "interventional",
            "n_jobs": shap_n_jobs,
            # "memory_tolerance": -1, # default
            "random_state": 19920722,
            "approximate": approximate,
            "shortcut": True,
        },
    )

    print(f"\nSHAP init:")
    print(f"- X_train.shape:{X_train.shape}")
    print(f"- y_train.shape{y_train.shape}")
    start_time = time.time()
    # Feed the scaled input for SHAP explainer
    scaled_X_train = pre_shap_pip[:-1].transform(X_train)
    shap_feature_selector.fit(scaled_X_train, y_train)
    end_time = time.time()
    print(f"SHAP done - {end_time - start_time} seconds")

    # Output feature selection results
    shap_df = pd.DataFrame(
        {
            "column_name": shap_feature_selector.feature_names,
            "abs_mean_shap": np.abs(shap_feature_selector.shap_values).mean(axis=0),
        }
    ).sort_values(by="abs_mean_shap", ascending=False)

    return shap_df


def _make_zoish_lohrasb_pipeline(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    trainer: tuple[str, str],
    n_features: int,
    n_trials: int,
    n_approx=100,
):
    all_features = X_train.columns.tolist()

    # get int/float/cat colnames
    # int_cols = X_train.select_dtypes(include=['int']).columns.tolist()
    # float_cols = X_train.select_dtypes(include=['float']).columns.tolist()
    cat_cols = X_train.select_dtypes(include=["object"]).columns.tolist()

    # get transformer
    transformers = []
    if len(cat_cols) > 0:
        transformers.append(("CatBoostEncoder", cen.CatBoostEncoder(random_state=19920722)))

    measure_of_accuracy, with_stratified = _set_opt_params(trainer)

    feature_selectors = []
    # shap approximate cannot work with catboost
    if trainer[0] == "cat":
        if X_train.shape[1] <= n_approx and n_features >= X_train.shape[1]:
            print("Skip feature selection for Catboost - Select more features than provided.")
            shap_df = pd.DataFrame({"column_name": X_train.columns, "abs_mean_shap": [0] * len(X_train.columns)})
        else:
            shap_df = _execute_zoish_shap_selector(
                X_train,
                y_train,
                trainer,
                n_trials,
                n_features,
                measure_of_accuracy,
                with_stratified,
                "catboost_feature_selection",
                shap_n_jobs=1,
            )  # detailed shap values for all features (saved for catboost)
            feature_selectors.append(("ColumnSelector", ColumnSelector(list(shap_df["column_name"].head(n_features)))))

    # approx would introduce variability for final feauture selection; 200 will trigger additivity issue without approx
    # this would keep the consistency for feature selection and avoid additivity issue
    elif X_train.shape[1] > n_approx and n_features > n_approx:
        shap_df = _execute_zoish_shap_selector(
            X_train,
            y_train,
            trainer,
            n_trials,
            n_features,
            measure_of_accuracy,
            with_stratified,
            "approx_feature_selection",
            shap_n_jobs=-1,
            approximate=True,
        )  # approx shap values for top (approx+) N features (saved but not reuse)
        feature_selectors.append(("ColumnSelector", ColumnSelector(list(shap_df["column_name"].head(n_features)))))

    # approx first and further narrow down toward target number
    elif X_train.shape[1] > n_approx and n_features <= n_approx:
        approx_shap_df = _execute_zoish_shap_selector(
            X_train,
            y_train,
            trainer,
            n_trials,
            n_approx,
            measure_of_accuracy,
            with_stratified,
            "approx_feature_selection",
            shap_n_jobs=-1,
            approximate=True,
        )  # approx shap values for all features
        shap_df = _execute_zoish_shap_selector(
            X_train[list(approx_shap_df["column_name"].head(n_approx))],
            y_train,
            trainer,
            n_trials,
            n_features,
            measure_of_accuracy,
            with_stratified,
            "feature_selection",
            shap_n_jobs=1,
        )  # detailed shap values for top (approx) N features (saved for reuse)
        feature_selectors.append(("ColumnSelector", ColumnSelector(list(shap_df["column_name"].head(n_features)))))

    # if the init feature number low enough, then skip approx
    # this would keep the consistency for feature selection (additivity comes)
    elif X_train.shape[1] <= n_approx and n_features < X_train.shape[1]:
        shap_df = _execute_zoish_shap_selector(
            X_train,
            y_train,
            trainer,
            n_trials,
            n_features,
            measure_of_accuracy,
            with_stratified,
            "feature_selection",
            shap_n_jobs=1,
        )  # detailed shap values for (approx-) all features (saved for reuse)
        feature_selectors.append(("ColumnSelector", ColumnSelector(list(shap_df["column_name"].head(n_features)))))

    # if the init feature number less than selected number, then no feature selection is needed. (Ex. age_sex_only)
    # this would avoid segmental fault due to estimators can't converge throughout optuna trials
    # elif X_train.shape[1] <= n_approx and n_features >= X_train.shape[1]:
    else:
        print(
            f"Skip feature selection - Select more features than provided. {X_train.shape[1]} {n_approx} {n_features}"
        )
        shap_df = pd.DataFrame(
            {  # age+sex don't need further shap feature ranking
                "column_name": X_train.columns,
                "abs_mean_shap": [0] * len(X_train.columns),
            }
        )

    lohrasb_opt = _make_lohrasb_obj(
        y_train, trainer, n_trials, measure_of_accuracy, f"LohrasbOptimizer_{trainer[0]}_{trainer[1]}", with_stratified
    )

    # make pipeline
    pip = Pipeline(
        [("ALLColumn", ColumnSelector(all_features))]
        + transformers
        + feature_selectors
        + [(f"LohrasbOptimizer_{trainer[0]}_{trainer[1]}", lohrasb_opt)]
    )

    return pip, shap_df


def _fit_pipeline(
    *,
    X_train,
    y_train,
    trainer,
    fname,
    n_features=0,
    n_trials=0,
    n_approx=200,
):
    joblib_fp = f"final_pipeline__{fname}.joblib"

    if n_features > n_approx:
        shap_path = f"shap.{fname}__preselect.tsv"
    else:
        if trainer[0] == "cat":
            shap_path = f"shap__{fname}__preselect.tsv".replace(f"top_{n_features}", f"top_N")
        else:
            shap_path = f"shap__{fname}__preselect.tsv".replace(f"top_{n_features}", f"top_{n_approx}")

    pip, shap_df = _make_zoish_lohrasb_pipeline(
        X_train,
        y_train,
        trainer,
        n_features,
        n_trials,
        n_approx=n_approx,
    )
    pip.fit(X_train, y_train)
    shap_df.to_csv(shap_path, sep="\t", index=False)

    dump(pip, joblib_fp)
    print(f"######################################")
    print(f"- {joblib_fp} saved.")
    print(f"######################################")

    return pip


def _get_X_y_idx(
    *,
    df: pd.DataFrame,
    id_col: str,
    y_label: str,
    trainer: tuple[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    idx = df[id_col]
    df = df.reindex(sorted(df.columns), axis=1)
    if trainer[1] == "classifier":
        y = df.loc[:, df.columns == y_label].astype(int)
    else:
        y = df.loc[:, df.columns == y_label].astype(float)
    X = df.loc[:, df.columns != y_label]
    print(f"input data shape: {X.shape}")
    print(f"input outcome shape: {y.shape}")
    return X, y, idx


def _check_df(
    df: pd.DataFrame,
    y_label: str,
    trainer: tuple[str, str],
) -> None:
    print(f" - preview {y_label}")
    if trainer[1] == "classifier":
        print(df[y_label].value_counts())
    else:
        print(df[y_label].describe())


def _set_train_test(
    trainer: tuple[str, str],
    X: pd.DataFrame,
    y: pd.DataFrame,
    idx: pd.Series,
    test_size: float = 0.20,
    random_state: int = 19920722,
):
    if trainer[1] == "classifier":
        X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
            X, y, idx, test_size=test_size, random_state=random_state, stratify=y
        )
    else:
        X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
            X, y, idx, test_size=test_size, random_state=random_state
        )
    return X_train, X_test, y_train, y_test, idx_train, idx_test


def _eval_pred(
    *,
    pip: Pipeline,
    trainer: tuple[str, str],
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[np.ndarray, float]:
    y_pred = pip.predict_proba(X)[:, 1] if trainer[1] == "classifier" else pip.predict(X)
    acc = _get_acc_metric(y, y_pred, trainer)
    return y_pred, acc


def _save_preds(
    pred_fp: str,
    trainer: tuple[str, str],
    fname: str,
    idx,
    y_pred,
    acc_lbl: str,
    acc: float,
):
    my_pred = Path(pred_fp)
    if my_pred.is_file():
        print(f"found {pred_fp}: found existed predicts!!!")
    else:
        acc_metric = "AUC" if trainer[1] == "classifier" else "R2"

    pred_df = pd.DataFrame({f"eid": idx, f"{fname}({acc_lbl}-{acc_metric}={acc:.8f})": y_pred})
    pred_df.to_pickle(pred_fp)
    print(f"######################################")
    print(f"- {pred_fp} saved.")
    print(f"######################################")


def execute_model(
    *,
    y_label: str,
    input_pickle_fp: str,
    id_col: str,
    n_features: int,
    n_trials: int,
    trainer: tuple[str],
):
    fname = f"{trainer[0]}_{trainer[1]}__{y_label}"

    df = pd.read_pickle(input_pickle_fp)
    _check_df(df, y_label, trainer)

    X, y, idx = _get_X_y_idx(df=df, id_col=id_col, y_label=y_label, trainer=trainer)

    X_train, X_test, y_train, y_test, _, idx_test = _set_train_test(trainer, X, y, idx)

    fitted_pipeline = _fit_pipeline(
        X_train=X_train,
        y_train=y_train,
        trainer=trainer,
        fname=fname,
        n_features=n_features,
        n_trials=n_trials,
    )
    y_pred, acc = _eval_pred(pip=fitted_pipeline, trainer=trainer, X=X_test, y=y_test)

    _save_preds(f"final_pred.{fname}.pkl", trainer, fname, idx_test, y_pred, "test", acc)

    return y_pred, acc
