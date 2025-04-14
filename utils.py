#!/usr/bin/env python3

import itertools
import random
import sys
import warnings
from functools import reduce
from pathlib import Path

import catboost as cb
import category_encoders as cen
import fasttreeshap
import lightgbm as lgb
import lohrasb
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import scipy
import shap
import sklearn
import xgboost as xgb
import zoish
from joblib import dump, load
from lohrasb.best_estimator import BaseModel
from optuna.pruners import HyperbandPruner
from optuna.samplers._tpe.sampler import TPESampler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import classification_report, confusion_matrix, f1_score, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight
from zoish.feature_selectors.shap_selectors import ShapFeatureSelector

from hyperparams import ESTIMATORS_DICT, PARAMS_DICT

random.seed(19920722)
warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")
np.random.seed(19920722)

def set_dtype(df, pheno_df, new_numeric_cols):
    for col in df.columns:
        # check if col included in pheno_df
        query = pheno_df.loc[(pheno_df["Name"] == col),]["ValueType"]
        value_type = None if query.empty else query.values[0]

        if col in new_numeric_cols:
            df[col] = df[col].astype(float)
        elif col == "eid":
            df[col] = df[col].astype(int)
        else:
            if value_type is not None:
                if value_type == "Continuous":
                    df[col] = df[col].astype(float)
                elif value_type == "Integer":
                    df[col] = df[col].astype(int)
                else:
                    df[col] = df[col].astype(str)
            else:
                df[col] = df[col].astype(str)
    return df


def merge_data(pre_fp, pgs_fp, dur_dx_fp, admixture_fp, pheno_df):
    pre_df = pd.read_csv(pre_fp).astype({"eid": int})
    print(f"load pre from {pre_fp}:\t{pre_df.shape} | {round(sys.getsizeof(pre_df) / (1024**3), 2)}GB")
    dur_dx_df = pd.read_csv(dur_dx_fp).astype(int)
    print(f"load dur_dx from {dur_dx_fp}:\t{dur_dx_df.shape} | {round(sys.getsizeof(dur_dx_df) / (1024**3), 2)}GB")

    pgs_df = pd.read_pickle(pgs_fp)  # .astype({"eid": int})
    # pgs_df = pgs_df[[i for i in pgs_df.columns if i == "eid" or int(i.split("(")[1].split("|")[0]) >= 10]] #ADJUSTED
    print(f"load pgs from {pgs_fp}:\t{pgs_df.shape} | {round(sys.getsizeof(pgs_df) / (1024**3), 2)}GB")
    admixture_df = pd.read_csv(admixture_fp, sep=" ", header=None)
    admixture_df.columns = ["eid", "EUR", "EAS", "AMR", "SAS", "AFR"]
    admixture_df = admixture_df.astype({"EUR": int, "EAS": int, "AMR": int, "SAS": int, "AFR": int})  # float #ADJUSTED
    admixture_df["eid"] = admixture_df["eid"].str.split("_").str[0].astype(int)
    print(
        f"load admix from {admixture_fp}:\t{admixture_df.shape} | {round(sys.getsizeof(admixture_df) / (1024**3), 2)}GB"
    )

    print(f"")
    print(f"######################################")
    print(f"Merging...")
    print(f"######################################")
    qrisk_cols = pre_df.columns[pre_df.columns.str.contains(pat="QRISK")].tolist()
    ascvd_cols = pre_df.columns[pre_df.columns.str.contains(pat="ascvd")].tolist()
    age_of_dx_cols = pre_df.columns[pre_df.columns.str.contains(pat="^AGE_")].tolist()

    pre_df = set_dtype(pre_df, pheno_df, qrisk_cols + ascvd_cols + age_of_dx_cols)
    pre_df = reduce(
        lambda left, right: pd.merge(left, right, on="eid"),
        [pre_df, pgs_df, dur_dx_df, admixture_df],
    )

    all_cols = pre_df.columns[1:].tolist()  # excluding eid
    age_gender_cols = pre_df.columns[pre_df.columns.str.contains(pat="_f21003$|_f31$")].tolist()
    treatment_cols = pre_df.columns[pre_df.columns.str.contains(pat="_f20003$|medication|Medication")].tolist()
    ancestry_family_history_cols = pre_df.columns[
        pre_df.columns.str.contains(pat="EUR|SAS|EAS|AMR|AFR|ethnic|Family_illness")
    ].tolist()
    ancestry_family_history_cols = pre_df.columns[
        pre_df.columns.str.contains(pat="EUR|SAS|EAS|AMR|AFR|ethnic|Family_illness")
    ].tolist()
    non_ukb_pgs_cols = [col for col in pgs_df.columns.tolist() if "JoinUKB" not in col and "eid" not in col]
    join_ukb_pgs_cols = [col for col in pgs_df.columns.tolist() if "JoinUKB" in col]
    unmodifiable_cols = age_gender_cols + ancestry_family_history_cols + non_ukb_pgs_cols + join_ukb_pgs_cols
    dur_dx_cols = pre_df.columns[pre_df.columns.str.contains(pat="^DUR_")].tolist()

    print(f" - age and gender counts: {len(age_gender_cols)}")
    print(f" - total qrisk counts: {len(qrisk_cols)}")
    print(f" - total ascvd counts: {len(ascvd_cols)}")
    print(f" - total treatment counts: {len(treatment_cols)}")
    print(f" - age of event counts: {len(age_of_dx_cols)}")
    print(f" - ancestry and history counts: {len(ancestry_family_history_cols)}")
    print(f" - non-UKBB-derived PGS counts: {len(non_ukb_pgs_cols)}")
    print(f" - join-UKBB-derived PGS counts: {len(join_ukb_pgs_cols)}")
    print(f" - total duration of diagnosis counts: {len(dur_dx_cols)}")

    print(f"\nTotal feature counts: {len(all_cols)}")
    print(f" -- Total unmodifiable counts: {len(unmodifiable_cols)}")
    print(f" -- Total modifiable counts: {len(all_cols) - len(unmodifiable_cols)}")

    # cols dictionary
    col_dict = {
        "age_gender": age_gender_cols,
        "qrisk": qrisk_cols,
        "ascvd": ascvd_cols,
        "treatment": treatment_cols,
        "age_of_dx": age_of_dx_cols,
        "ancestry_family_history": ancestry_family_history_cols,
        "non_ukb_prs": non_ukb_pgs_cols,
        "join_ukb_prs": join_ukb_pgs_cols,
        "unmodifiable": unmodifiable_cols,
        "duration_of_dx": dur_dx_cols,
    }

    # excluded columns
    base_excluded = (
        [
            "eid",
            "#sampleID",
            "ANCESTRY",
            "home_location_at_assessment_east_coordinate_rounded_f20074",
            "home_location_at_assessment_north_coordinate_rounded_f20075",
            "uk_biobank_assessment_centre_f54",
        ]
        + age_of_dx_cols
        + pre_df.columns[pre_df.isnull().all()].tolist()
    )
    print(f"\nBasic exclusion counts: {len(base_excluded)}")
    print(f"{base_excluded}")

    return pre_df, col_dict, base_excluded


def load_data(step, my_pkg=None, my_test=None, base_n_features=0, my_final_round=None):
    print(f"")
    print(f"######################################")
    print(f"Loading...")
    print(f"######################################")
    main = "/mnt/stsi/stsi1/sfchen/210430_ukbb_pheno_harmonization"
    admixture_dir = "/mnt/stsi/stsi3/Internal/UKBB/imputed_HRC_minimac4/hg19/2_merge"

    pheno_fp = f"{main}/seventh_data_request_basket/supplementary_table.csv"
    pgs_fp = f"{main}/analysis/csv/230610_non_ukb_join_ukb_prs_std.pkl"
    dur_dx_fp = f"{main}/analysis/csv/230404_prevalent_dx_duration.csv"
    admixture_fp = f"{admixture_dir}/ukb_hap_v2.lifted_already_GRCh37.GH.pruned.intersect1KG.5.Q.IDs"
    prevalent_merged_ukbb_fp = f"{main}/analysis/csv/230916_prevalence_merged_ukbb.csv"
    DX_pred_fp = f"{main}/analysis/{my_test}/DX_{my_pkg}_preds.n_feature_{base_n_features}.pkl"
    non_DX_pred_fp = f"{main}/analysis/{my_test}/non_DX_{my_pkg}_preds.n_feature_{base_n_features}.pkl"

    pheno_df = pd.read_csv(pheno_fp)

    pre_df, col_dict, base_excluded = merge_data(prevalent_merged_ukbb_fp, pgs_fp, dur_dx_fp, admixture_fp, pheno_df)

    # get colname of meta features
    if step == "final":
        DX_pred_df = pd.read_pickle(DX_pred_fp)
        print(
            f"load DX_pred from {DX_pred_fp}: {DX_pred_df.shape} | {round(sys.getsizeof(DX_pred_df) / (1024**3), 2)}GB"
        )
        non_DX_pred_df = pd.read_pickle(non_DX_pred_fp)
        print(
            f"load non_DX_pred from {non_DX_pred_fp}: {non_DX_pred_df.shape} | "
            f"{round(sys.getsizeof(non_DX_pred_df) / (1024**3), 2)}GB"
        )
        print(f"predicted diagnosis counts: {len(DX_pred_df.columns[1:].tolist())}")
        print(f"predicted non-diagnosis counts: {len(non_DX_pred_df.columns[1:].tolist())}")
        preds_cols = [col for col in DX_pred_df.columns.tolist() + non_DX_pred_df.columns.tolist() if col != "eid"]
        contemps_preds_cols = [col for col in preds_cols if ".all(" in col]
        no_contemps_preds_cols = [col for col in preds_cols if "no_contemps" in col]
        pre_df = reduce(
            lambda left, right: pd.merge(left, right, on="eid"),
            [pre_df, DX_pred_df, non_DX_pred_df],
        )
    else:
        preds_cols = []
        contemps_preds_cols = []
        no_contemps_preds_cols = []

    col_dict["preds"] = preds_cols
    col_dict["contemps_preds"] = contemps_preds_cols
    col_dict["no_contemps_preds"] = no_contemps_preds_cols

    return pre_df, pheno_df, col_dict, base_excluded


def get_model_design(my_step, my_y_lbl, my_pkg, pre_df, col_dict):
    model_design_fp = (
        "/mnt/stsi/stsi1/sfchen/210430_ukbb_pheno_harmonization/analysis//UKBB_phenotyping_Model_design.tsv"
    )
    model_design_df = pd.read_csv(model_design_fp, sep="\t", doublequote=True, quotechar='"', quoting=3)

    model_design_df = model_design_df.fillna(value={"In_ASCVD": "F", "In_QRISK3": "F"})
    config_df = model_design_df.loc[model_design_df["Type"].isin(["binary", "num", "diagnosis"]),][
        ["Type", "Trait", "In_ASCVD", "In_QRISK3", "Exclusion"]
    ]
    target_trait_config = config_df.loc[config_df["Trait"] == my_y_lbl,].squeeze()
    if target_trait_config.shape[0] == 0:
        sys.exit(f"Invalid target outcome not found in the model designs {my_y_lbl}")
    target_type = target_trait_config["Type"]
    target_ascvd = target_trait_config["In_ASCVD"]
    target_qrisk = target_trait_config["In_QRISK3"]

    if str(target_trait_config["Exclusion"]) == "nan":
        target_exclusion = ""
    else:
        if my_step != "final":
            target_exclusion_str = target_trait_config["Exclusion"]
            target_exclusion_pt = target_exclusion_str.replace('"', "").replace(", ", "|")
            target_exclusion = pre_df.columns[pre_df.columns.str.contains(pat=target_exclusion_pt)].tolist()
        else:
            target_exclusion = []

    if target_type == "binary":
        trainer = (my_pkg, "classifier")
    elif target_type == "num":
        trainer = (my_pkg, "regressor")
    elif target_type == "diagnosis":
        trainer = (my_pkg, "classifier")
    else:
        print(f"{my_y_lbl} not included.")
        sys.exit()

    cus_drop_cols = [target_exclusion]
    cus_drop_cols = cus_drop_cols + [col_dict["ascvd"]] if target_ascvd == "T" else cus_drop_cols
    cus_drop_cols = cus_drop_cols + [col_dict["qrisk"]] if target_qrisk == "T" else cus_drop_cols
    cus_drop_cols = list(itertools.chain(*cus_drop_cols))  # flatten list

    print(f" - trainer: {trainer}")
    print(f" - cus_drop_cols: {cus_drop_cols}")

    return trainer, cus_drop_cols, target_type


def split_ukbb(stratify_fp, ukbb_df, dx, lbls):
    stratify_df = pd.read_csv(stratify_fp).astype({"eid": int})
    sub_ids = stratify_df[stratify_df[dx].isin(lbls)]["eid"]
    sub_df = ukbb_df[(ukbb_df["eid"].isin(sub_ids))]
    return sub_df


def update_y(df, y_label, fp, ex_fp=None):
    updated_df = df.copy()
    updated_df = updated_df.drop(y_label, errors="ignore", axis=1)
    y_df = pd.read_csv(fp, usecols=["eid", y_label]).astype({"eid": int})
    if ex_fp is not None:
        ex_y_df = pd.read_csv(ex_fp, usecols=["eid", y_label]).astype({"eid": int})
        ex_y = ex_y_df[y_label] * (-1)
        y_df[y_label] = y_df[y_label] + ex_y
    updated_df = pd.merge(updated_df, y_df, on="eid")
    return updated_df


def load_config(my_step, my_y_lbl, my_pkg, my_final_round, pre_df, col_dict):
    print(f"")
    print(f"")
    print(f"")
    print(
        f"########################################################################################################################################################"
    )
    print(
        f"########################################################################################################################################################"
    )
    print(f"Load {my_step} config...")
    print(
        f"########################################################################################################################################################"
    )
    print(
        f"########################################################################################################################################################"
    )

    # load model design
    trainer, cus_drop_cols, target_type = get_model_design(my_step, my_y_lbl, my_pkg, pre_df, col_dict)

    # manage categorical outcome
    if "." in my_y_lbl:
        y_cat = my_y_lbl.split(".")[1]
        y_label = my_y_lbl.split(".")[0]
        print(f"Multi-categorical outcomes in {y_label}: set {y_cat} as positive.")
        pre_df[y_label] = np.where(pre_df[y_label] == y_cat, "1", "0")
        print(f"recode multi-categorical outcome: set {y_cat} as 1; the rest as 0.")
    else:
        y_label = my_y_lbl

    print(f"")
    print(
        f"##################################################################################################################"
    )
    print(f"Split dataset...")
    print(
        f"##################################################################################################################"
    )
    # split ukbb into baseline and observed cohorts
    ukbb_stratify_fp = (
        "/mnt/stsi/stsi1/sfchen/210430_ukbb_pheno_harmonization/analysis/csv/230404_ukbb_stratification.csv"
    )
    base_stratify_fp = (
        "/mnt/stsi/stsi1/sfchen/210430_ukbb_pheno_harmonization/analysis/csv/230404_base_stratification.csv"
    )
    baseline_df = split_ukbb(ukbb_stratify_fp, pre_df, "DX_Coronary_artery_disease", ["baseline"])
    observed_df = split_ukbb(ukbb_stratify_fp, pre_df, "DX_Coronary_artery_disease", ["observed"])

    if my_step == "base_fit":
        print(f"base model training   - using baseline cohort.")
        input_df = baseline_df
    elif my_step == "base_predict":
        print(f"base model predicting - using observed cohort.")
        input_df = observed_df
    elif my_step == "final":
        print(f"final train/test      - using observed cohort.")
        input_df = observed_df
    else:
        sys.exit(f"invalid step: {my_step}")

    # update the outcome with additional time point
    prevalent_early_merged_ukbb_fp = (
        "/mnt/stsi/stsi1/sfchen/210430_ukbb_pheno_harmonization/analysis/csv/230916_early_onset_merged_ukbb.csv"
    )
    prevalent_merged_ukbb_fp = (
        "/mnt/stsi/stsi1/sfchen/210430_ukbb_pheno_harmonization/analysis/csv/230916_prevalence_merged_ukbb.csv"
    )
    incident_10yr_merged_ukbb_fp = (
        "/mnt/stsi/stsi1/sfchen/210430_ukbb_pheno_harmonization/analysis/csv/230916_incidence_10yr_merged_ukbb.csv"
    )
    tot_merged_ukbb_fp = (
        "/mnt/stsi/stsi1/sfchen/210430_ukbb_pheno_harmonization/analysis/csv/230916_lifetime_merged_ukbb.csv"
    )

    dfs = {}
    # base
    if my_step in ["base_fit", "base_predict"]:
        dfs["base"] = input_df
        if target_type != "diagnosis":
            training_configs = {
                ("base", "age_gender_only"): list(set(["eid", y_label] + col_dict["age_gender"])),
                ("base", "unmodifiable"): list(set(["eid", y_label] + col_dict["unmodifiable"])),
                ("base", "no_contemps"): [
                    n for n in dfs["base"].columns if n not in col_dict["ascvd"] + col_dict["qrisk"]
                ],
            }

        if target_type == "diagnosis":
            if my_step in "base_fit":  # fit
                tot_y_df = update_y(input_df, y_label, tot_merged_ukbb_fp)

                dfs["earlyonset"] = split_ukbb(
                    base_stratify_fp,
                    tot_y_df,
                    y_label,
                    ["control", "prevonlycontrol", "death", "earlyonset"],
                )
                dfs["lateonset"] = split_ukbb(
                    base_stratify_fp,
                    tot_y_df,
                    y_label,
                    ["control", "prevonlycontrol", "death", "lateonset"],
                )
                dfs["prediagnosis"] = split_ukbb(
                    base_stratify_fp,
                    tot_y_df,
                    y_label,
                    ["control", "prevonlycontrol", "death", "earlyonset", "lateonset"],
                )
                dfs["10yr_risk"] = split_ukbb(base_stratify_fp, tot_y_df, y_label, ["control", "10yr"])
                dfs["20yr_risk"] = split_ukbb(base_stratify_fp, tot_y_df, y_label, ["control", "10yr", "20yr"])

            else:  # transform
                # all will be used to generate/test predictions from pre-trained models
                dfs["earlyonset"] = update_y(input_df, y_label, prevalent_early_merged_ukbb_fp)
                dfs["lateonset"] = update_y(input_df, y_label, prevalent_merged_ukbb_fp)
                dfs["prediagnosis"] = update_y(input_df, y_label, prevalent_merged_ukbb_fp)
                dfs["10yr_risk"] = update_y(input_df, y_label, incident_10yr_merged_ukbb_fp)
                dfs["20yr_risk"] = update_y(input_df, y_label, tot_merged_ukbb_fp)

            exclude_dx_duration = [n for n in col_dict["duration_of_dx"] if y_label in n]

            training_configs = {
                ("earlyonset", "age_gender_only"): list(set(["eid", y_label] + col_dict["age_gender"])),
                ("earlyonset", "unmodifiable"): list(set(["eid", y_label] + col_dict["unmodifiable"])),
                ("lateonset", "age_gender_only"): list(set(["eid", y_label] + col_dict["age_gender"])),
                ("lateonset", "unmodifiable"): list(set(["eid", y_label] + col_dict["unmodifiable"])),
                ("prediagnosis", "age_gender_only"): list(set(["eid", y_label] + col_dict["age_gender"])),
                ("prediagnosis", "unmodifiable"): list(set(["eid", y_label] + col_dict["unmodifiable"])),
                ("10yr_risk", "age_gender_only"): list(set(["eid", y_label] + col_dict["age_gender"])),
                ("10yr_risk", "no_contemps"): [
                    n
                    for n in dfs["10yr_risk"].columns
                    if n not in exclude_dx_duration + col_dict["ascvd"] + col_dict["qrisk"]
                ],
                ("20yr_risk", "age_gender_only"): list(set(["eid", y_label] + col_dict["age_gender"])),
                ("20yr_risk", "no_contemps"): [
                    n
                    for n in dfs["20yr_risk"].columns
                    if n not in exclude_dx_duration + col_dict["ascvd"] + col_dict["qrisk"]
                ],
            }

    # final
    elif my_step == "final":
        # we only predict 10yr CAD risk
        dfs["final"] = update_y(input_df, y_label, incident_10yr_merged_ukbb_fp)
        training_configs = {
            ("final", "age_gender_only"): list(set(["eid", y_label] + col_dict["age_gender"])),
            ("final", "unmodifiable"): list(set(["eid", y_label] + col_dict["unmodifiable"])),
            ("final", "modifiable"): [
                n
                for n in dfs["final"].columns
                if n not in col_dict["ascvd"] + col_dict["qrisk"] + col_dict["preds"] + col_dict["unmodifiable"]
            ],
            ("final", "ascvd_only"): list(set(["eid", y_label] + col_dict["ascvd"])),
            ("final", "qrisk_only"): list(set(["eid", y_label] + col_dict["qrisk"])),
            ("final", "contemps"): list(set(["eid", y_label] + col_dict["ascvd"] + col_dict["qrisk"])),
            ("final", "preds"): list(set(["eid", y_label] + col_dict["preds"])),
            ("final", "no_contemps"): [
                n
                for n in dfs["final"].columns
                if n not in col_dict["ascvd"] + col_dict["qrisk"] + col_dict["contemps_preds"]
            ],
            ("final", "no_preds"): [
                n for n in dfs["final"].columns if n not in col_dict["ascvd"] + col_dict["qrisk"] + col_dict["preds"]
            ],
            ("final", "all"): dfs["final"].columns,
        }

    config_keys = list(training_configs.keys())
    print(f"All training setting: {config_keys}")

    if my_step != "final":
        training_rounds = config_keys
    elif my_step == "final":
        training_rounds = [("final", my_final_round)]
        if training_rounds[0] not in config_keys:
            sys.exit(f"unknown final training_config {training_rounds[0]}")

    return dfs, y_label, cus_drop_cols, trainer, training_rounds, training_configs


def get_df(dfs, y_label, training_round, my_step, trainer, training_configs, debug):
    y_title = y_label + "." + ".".join(training_round)
    df = dfs[training_round[0]][training_configs[training_round]]

    if debug:
        df = df.head(20000)

    df = df.rename(columns={y_label: y_title})
    print(f" - {training_round[0]} dataset: {df.shape}")
    print(f" - preview {y_title}")
    if trainer[1] == "classifier":
        print(df[y_title].value_counts())
    else:
        print(df[y_title].describe())

    return df, y_title


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


def make_zoish_selector(
    X_train,
    y_train,
    estimators_dict,
    params_dict,
    trainer,
    n_trials,
    n_features,
    shap_n_jobs,
    measure_of_accuracy,
    with_stratified,
    fname,
    approximate=False,
):
    feature_selector = (
        ShapFeatureSelector.shap_feature_selector_factory.set_model_params(
            X=X_train,
            y=y_train,
            verbose=0,
            random_state=19920722,
            estimator=estimators_dict[trainer],
            estimator_params=params_dict[trainer],
            fit_params={},  # fit_params,
            method="optuna",  # "optuna", "tunesearch"
            # if n_features=None only the threshold will be considered as a cut-off of features grades.
            # if threshold=None only n_features will be considered to select the top n features.
            # if both of them are set to some values, the threshold has the priority for selecting features.
            n_features=n_features,
            threshold=None,
            list_of_obligatory_features_that_must_be_in_model=[],
            list_of_features_to_drop_before_any_selection=[],
        )
        .set_shap_params(
            model_output="raw",
            feature_perturbation="interventional",
            algorithm="v2",
            shap_n_jobs=shap_n_jobs,
            memory_tolerance=-1,
            feature_names=None,
            approximate=approximate,
            shortcut=False,
        )
        .set_optuna_params(
            measure_of_accuracy=measure_of_accuracy,
            # optuna params
            with_stratified=with_stratified,
            test_size=0.10,
            n_jobs=-1,
            # optuna params
            # optuna study init params
            study=optuna.create_study(
                storage=None,
                sampler=TPESampler(seed=19920722),
                pruner=HyperbandPruner(),
                study_name=None,
                direction="maximize",
                load_if_exists=False,
                directions=None,
            ),
            # optuna optimization params
            study_optimize_objective=None,
            study_optimize_objective_n_trials=n_trials,
            study_optimize_objective_timeout=600,
            study_optimize_n_jobs=1,
            study_optimize_catch=(),
            study_optimize_callbacks=None,
            study_optimize_gc_after_trial=True,
            study_optimize_show_progress_bar=False,
        )
    )
    return feature_selector


def set_opt_params(my_step, trainer):
    # set classification / regression
    if trainer[1] == "classifier":
        if my_step != "final":  # ADJUSTED
            measure_of_accuracy = f"f1_score(y_true, y_pred, average='macro')"
        else:
            measure_of_accuracy = f"f1_score(y_true, y_pred, average='macro')"
        print(f"measure_of_accuracy USED {measure_of_accuracy}")
        with_stratified = True
    else:
        measure_of_accuracy = f"r2_score(y_true, y_pred)"
        with_stratified = False
    return measure_of_accuracy, with_stratified


def make_zoish_lohrasb_pipeline(
    my_step,
    X_train,
    y_train,
    trainer,
    fname,
    n_features,
    n_trials,
    n_approx=100,
    shap_df=None,
):
    all_features = X_train.columns.tolist()

    # get int/float/cat colnames
    # int_cols = X_train.select_dtypes(include=['int']).columns.tolist()
    # float_cols = X_train.select_dtypes(include=['float']).columns.tolist()
    cat_cols = X_train.select_dtypes(include=["object"]).columns.tolist()

    # get transformer
    transformers = []
    if len(cat_cols) > 0:
        transformers.append(
            (
                "CatBoostEncoder",
                cen.CatBoostEncoder(random_state=19920722),
            )
        )

    measure_of_accuracy, with_stratified = set_opt_params(my_step, trainer)

    feature_selectors = []
    if shap_df is None:
        # shap approximate cannot work with catboost
        if trainer[0] == "cat":
            if X_train.shape[1] <= n_approx and n_features >= X_train.shape[1]:
                pass
            else:
                feature_selectors.append(
                    (
                        "ZoishSelector",
                        make_zoish_selector(
                            X_train,
                            y_train,
                            ESTIMATORS_DICT,
                            PARAMS_DICT,
                            trainer,
                            n_trials,
                            n_features,
                            1,
                            measure_of_accuracy,
                            with_stratified,
                            fname,
                        ),
                    )
                )
        # approx would introduce variability for final feauture selection; 200 will trigger additivity issue
        # without approx this would keep the consistency for feature selection and avoid additivity issue
        elif X_train.shape[1] > n_approx and n_features > n_approx:
            feature_selectors.append(
                (
                    "ApproxZoishSelector",
                    make_zoish_selector(
                        X_train,
                        y_train,
                        ESTIMATORS_DICT,
                        PARAMS_DICT,
                        trainer,
                        n_trials,
                        n_features,
                        -1,
                        measure_of_accuracy,
                        with_stratified,
                        fname,
                        approximate=True,
                    ),
                )
            )

        # approx first and further narrow down toward target number
        elif X_train.shape[1] > n_approx and n_features <= n_approx:
            feature_selectors.append(
                (
                    "ApproxZoishSelector",
                    make_zoish_selector(
                        X_train,
                        y_train,
                        ESTIMATORS_DICT,
                        PARAMS_DICT,
                        trainer,
                        n_trials,
                        n_approx,
                        -1,
                        measure_of_accuracy,
                        with_stratified,
                        fname,
                        approximate=True,
                    ),
                )
            )
            feature_selectors.append(
                (
                    "ZoishSelector",
                    make_zoish_selector(
                        X_train,
                        y_train,
                        ESTIMATORS_DICT,
                        PARAMS_DICT,
                        trainer,
                        n_trials,
                        n_features,
                        1,
                        measure_of_accuracy,
                        with_stratified,
                        fname,
                    ),
                )
            )

        # if the init feature number low enough, then skip approx
        # this would keep the consistency for feature selection (additivity comes)
        elif X_train.shape[1] <= n_approx and n_features < X_train.shape[1]:
            feature_selectors.append(
                (
                    "ZoishSelector",
                    make_zoish_selector(
                        X_train,
                        y_train,
                        ESTIMATORS_DICT,
                        PARAMS_DICT,
                        trainer,
                        n_trials,
                        n_features,
                        1,
                        measure_of_accuracy,
                        with_stratified,
                        fname,
                    ),
                )
            )

        # if the init feature number less than selected number, then no feature selection is needed.
        # (Ex. age_gender_only)
        # this would avoid segmental fault due to estimators can't converge throughout optuna trials
        # elif X_train.shape[1] <= n_approx and n_features >= X_train.shape[1]:
        else:
            pass
    else:
        # select top N features from the pre-calcualted SHAP importance df; no redo to avoid variaiblity
        feature_selectors.append(
            (
                "ColumnSelector",
                ColumnSelector(list(shap_df["column_name"].head(n_features))),
            )
        )

    lohrasb_opt = BaseModel().optimize_by_optuna(
        estimator=ESTIMATORS_DICT[trainer],
        estimator_params=PARAMS_DICT[trainer],
        measure_of_accuracy=measure_of_accuracy,
        with_stratified=with_stratified,
        fit_params={},
        test_size=0.10,
        verbose=0,
        n_jobs=1,
        random_state=19920722,
        # optuna params
        # optuna study init params
        study=optuna.create_study(
            storage=None,
            sampler=TPESampler(seed=19920722),
            pruner=HyperbandPruner(),
            study_name=None,
            direction="maximize",
            load_if_exists=False,
            directions=None,
        ),
        # optuna optimization params
        study_optimize_objective=None,
        study_optimize_objective_n_trials=n_trials,
        study_optimize_objective_timeout=600,
        study_optimize_n_jobs=1,
        study_optimize_catch=(),
        study_optimize_callbacks=None,
        study_optimize_gc_after_trial=True,
        study_optimize_show_progress_bar=False,
    )

    # make pipeline
    SHAP_PIP = Pipeline(
        [("ALLColumn", ColumnSelector(all_features))]
        + transformers
        + feature_selectors
        + [(f"LohrasbOptimizer_{trainer[0]}_{trainer[1]}", lohrasb_opt)]
    )

    return SHAP_PIP


def make_mini_pipeline(pre_PIP, re_X_train, re_y_train, trainer, fname):
    cat_cols = re_X_train.select_dtypes(include=["object"]).columns.tolist()

    transformer_list = []
    if len(cat_cols) > 0:
        transformer_list.append(
            (
                "CatBoostEncoder",
                cen.CatBoostEncoder(random_state=19920722),
            )
        )

    if any("Selector" in step for step in list(pre_PIP.named_steps.keys())):
        selector_step = max(loc for loc, val in enumerate(list(pre_PIP.named_steps.keys())) if "Selector" in val)
        selected_cols = sorted(pre_PIP[selector_step].selected_cols)
    else:
        selected_cols = sorted(list(re_X_train.columns))
    print(f"re-training selected cols: {len(selected_cols)}")

    if trainer[0] == "cat":
        estimator = pre_PIP[-1].best_estimator  # to avoid "CatBoostError: You can't change params of fitted model."
    else:
        estimator = ESTIMATORS_DICT[trainer]
        best_params = pre_PIP[-1].best_estimator.get_params()
        estimator.set_params(**best_params)

    # make piepline
    if trainer[0] == "ebm":
        RE_PIP = Pipeline(
            [("ColumnSelector", ColumnSelector(selected_cols))]
            + [(f"BestEstimator_{trainer[0]}_{trainer[1]}", estimator)]
        )
    else:
        RE_PIP = Pipeline(
            [("ColumnSelector", ColumnSelector(selected_cols))]
            + transformer_list
            + [(f"BestEstimator_{trainer[0]}_{trainer[1]}", estimator)]
        )

    return RE_PIP


def importance_review(
    X_test,
    y_title,
    PIP,
    trainer,
    n_features,
    fname,
    review_cohort,
    retrain=False,
    show=False,
    n_approx=50,
):
    if n_features > n_approx:
        pass
    shap_fp = f"retrain_shap.{fname}.{review_cohort}.png" if retrain else f"shap.{fname}.{review_cohort}.png"

    try:
        my_shap = Path(shap_fp)
        if my_shap.is_file():
            print(f"found {shap_fp}: found pre-examined shap!!!")
        else:
            plt.cla()
            plt.clf()
            fig = plt.gcf()

            if len(PIP.steps) == 1:
                scaled_X_test = X_test
                estimator = PIP[0]
            else:
                scaled_X_test = PIP[:-1].transform(X_test)
                estimator = PIP[-1].best_estimator if "best_estimator" in dir(PIP[-1]) else PIP[-1]

            explainer = fasttreeshap.TreeExplainer(estimator, algorithm="v2", n_jobs=-1)
            shap_values = explainer.shap_values(scaled_X_test)

            if show:
                shap.summary_plot(shap_values, scaled_X_test, max_display=n_features, show=False)
                plt.title(f"{y_title}")
                fig.savefig(
                    shap_fp,
                    dpi=300,
                    bbox_inches="tight",
                    transparent="True",
                    pad_inches=0,
                )
            else:
                shap.summary_plot(shap_values, scaled_X_test, max_display=n_features, show=show)
                plt.title(f"{y_title}")
        plt.close()
        plt.cla()
        plt.clf()

    except Exception as e:
        print(f"Model too complicated to review the SHAP... ({e})")


def get_acc_metric(y, y_pred, trainer, classifier_metric="F1"):
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


def eval_pred(PIP, trainer, X, y):
    y_pred = PIP.predict_proba(X)[:, 1] if trainer[1] == "classifier" else PIP.predict(X)
    acc = get_acc_metric(y, y_pred, trainer)
    return y_pred, acc


def save_preds(pred_fp, trainer, fname, idx, y_pred, acc_lbl, acc, classifier_metric="F1"):
    my_pred = Path(pred_fp)
    if my_pred.is_file():
        print(f"found {pred_fp}: found existed predicts!!!")
    else:
        acc_metric = ("F1" if classifier_metric == "F1" else "AUC") if trainer[1] == "classifier" else "R2"

        pred_df = pd.DataFrame({f"eid": idx, f"{fname}({acc_lbl}-{acc_metric}={acc})": y_pred})
        pred_df.to_pickle(pred_fp)
        print(f"######################################")
        print(f"- {pred_fp} saved.")
        print(f"######################################")


def save_joblib(PIP, joblib_fp):
    dump(PIP, joblib_fp)
    print(f"######################################")
    print(f"- {joblib_fp} saved.")
    print(f"######################################")


def get_X_y_idx(df, y_title, drop_cols, trainer):
    idx = df["eid"]
    df = df.reindex(sorted(df.columns), axis=1)
    drop_cols = drop_cols + df.columns[df.isna().any()].tolist()
    if trainer[1] == "classifier":
        y = df.loc[:, df.columns == y_title].astype(int)
    else:
        y = df.loc[:, df.columns == y_title].astype(float)
    df = df.drop(drop_cols, errors="ignore", axis=1)
    X = df.loc[:, df.columns != y_title]
    print(f"training data size: {X.shape}")
    print(f"training outcome size: {y.shape}")
    return X, y, idx


def set_train_test(trainer, X, y, idx, test_size=0.20, random_state=19920722):
    if trainer[1] == "classifier":
        X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
            X, y, idx, test_size=test_size, random_state=random_state, stratify=y
        )
    else:
        X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
            X, y, idx, test_size=test_size, random_state=random_state
        )
    return X_train, X_test, y_train, y_test, idx_train, idx_test


def fit_pipeline(
    my_step,
    joblib_fp,
    X_train,
    y_train,
    trainer,
    fname,
    n_features=0,
    n_trials=0,
    RE_PIP=None,
    with_sample_weight=False,
):
    my_joblib = Path(joblib_fp)

    if my_joblib.is_file():
        print(f"found {joblib_fp}: load pre-trained model!!!")
        PIP = load(joblib_fp)

    elif not my_joblib.is_file():
        n_approx = 200 if my_step == "final" else 100

        if n_features > n_approx:
            my_shap_path = f"shap.{fname}.preselect.tsv"
        else:
            if trainer[0] == "cat":
                my_shap_path = f"shap.{fname}.preselect.tsv".replace(f"top_{n_features}", f"top_N")
            else:
                my_shap_path = f"shap.{fname}.preselect.tsv".replace(f"top_{n_features}", f"top_{n_approx}")
        my_shap = Path(my_shap_path)

        if RE_PIP is None:
            if my_shap.is_file():
                print(f"found {my_shap_path}: load pre-selected shap importance scores!!!")
                shap_df = pd.read_csv(my_shap_path, sep="\t")
                pre_PIP = make_zoish_lohrasb_pipeline(
                    my_step,
                    X_train,
                    y_train,
                    trainer,
                    fname,
                    n_features,
                    n_trials,
                    shap_df=shap_df,
                )
            else:
                pre_PIP = make_zoish_lohrasb_pipeline(
                    my_step,
                    X_train,
                    y_train,
                    trainer,
                    fname,
                    n_features,
                    n_trials,
                    n_approx=n_approx,
                )
            pre_PIP.fit(X_train, y_train)

            if any("ZoishSelector" in step for step in list(pre_PIP.named_steps.keys())):
                pre_PIP[-2].importance_df.to_csv(my_shap_path, sep="\t", index=False)

            PIP = make_mini_pipeline(pre_PIP, X_train, y_train, trainer, fname)
        else:
            PIP = RE_PIP

        # https://stackoverflow.com/questions/47399350/how-does-sample-weight-compare-to-class-weight-in-scikit-learn
        # https://stackoverflow.com/questions/67868420/xgboost-for-multiclassification-and-imbalanced-data
        if with_sample_weight and trainer[1] == "classifier":
            print(">>> apply sample_weight <<<")
            sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
        else:
            sample_weight = None

        # ValueError: Pipeline.fit does not accept the sample_weight parameter.
        # You can pass parameters to specific steps of your pipeline using the stepname__parameter format,
        # e.g. `Pipeline.fit(X, y, logisticregression__sample_weight=sample_weight)`.
        PIP.fit(
            X_train,
            y_train,
            **{f"BestEstimator_{trainer[0]}_{trainer[1]}__sample_weight": sample_weight},
        )

        save_joblib(PIP, joblib_fp)

    else:
        sys.exit(f"Can't find {joblib_fp}")

    return PIP


def train_opt_pipeline(
    my_step,
    df,
    y_title,
    drop_cols,
    trainer,
    fname,
    n_features,
    n_trials,
    PIP=None,
    test_acc=0,
):
    print(f"\n==== train_opt_pipeline ==== {my_step} ====")

    X, y, idx = get_X_y_idx(df, y_title, drop_cols, trainer)

    if my_step == "base_fit":
        X_train, X_test, y_train, y_test, idx_train, idx_test = set_train_test(trainer, X, y, idx)
        PIP = fit_pipeline(
            my_step,
            f"base_pipeline.{fname}.joblib",
            X_train,
            y_train,
            trainer,
            fname,
            n_features,
            n_trials,
        )
        y_pred, acc = eval_pred(PIP, trainer, X_test, y_test)

    elif my_step == "base_predict":
        X_train, X_test, y_train, y_test, idx_train, idx_test = X, X, y, y, idx, idx
        y_pred, acc = eval_pred(PIP, trainer, X_test, y_test)
        print("save base_predict first time.")
        save_preds(
            f"base_pred.{fname}.pkl",
            trainer,
            fname,
            idx_test,
            y_pred,
            "test/validate",
            f"{test_acc}/{acc}",
        )

    elif my_step == "final":  # for now it's always CAD
        X_train, X_test, y_train, y_test, idx_train, idx_test = set_train_test(trainer, X, y, idx)
        PIP = fit_pipeline(
            my_step,
            f"final_pipeline.{fname}.joblib",
            X_train,
            y_train,
            trainer,
            fname,
            n_features,
            n_trials,
        )
        y_pred, acc = eval_pred(PIP, trainer, X_test, y_test)
        print("save final_predict first time.")
        save_preds(f"final_pred.{fname}.pkl", trainer, fname, idx_test, y_pred, "test", acc)

    else:
        sys.exit(f"invalid step: {my_step}")
    return PIP, y_pred, acc


def retrain_best_pipeline(my_step, df, y_title, drop_cols, trainer, fname, PIP, test_acc=0):
    print(f"\n==== retrain_best_pipeline ==== {my_step} ====")

    re_X, re_y, re_idx = get_X_y_idx(df, y_title, drop_cols, trainer)

    if my_step == "base_fit":
        REFIT_PIP = fit_pipeline(
            my_step,
            f"retrain_base_pipeline.{fname}.joblib",
            re_X,
            re_y,
            trainer,
            fname,
            RE_PIP=PIP,
            with_sample_weight=True,
        )
        re_y_pred, re_acc = eval_pred(REFIT_PIP, trainer, re_X, re_y)

    elif my_step == "base_predict":
        REFIT_PIP = PIP
        re_y_pred, re_acc = eval_pred(REFIT_PIP, trainer, re_X, re_y)
        save_preds(
            f"retrain_base_pred.{fname}.pkl",
            trainer,
            fname,
            re_idx,
            re_y_pred,
            "test/validate",
            f"{test_acc}/{re_acc}",
        )

    elif my_step == "final":
        re_X_train, re_X_test, re_y_train, re_y_test, re_idx_train, re_idx_test = set_train_test(
            trainer, re_X, re_y, re_idx
        )
        REFIT_PIP = fit_pipeline(
            my_step,
            f"retrain_final_pipeline.{fname}.joblib",
            re_X_train,
            re_y_train,
            trainer,
            fname,
            RE_PIP=PIP,
            with_sample_weight=True,
        )
        re_y_pred, re_acc = eval_pred(REFIT_PIP, trainer, re_X_test, re_y_test)
        save_preds(
            f"retrain_final_pred.{fname}.pkl",
            trainer,
            fname,
            re_idx_test,
            re_y_pred,
            "final_retrain_test",
            re_acc,
        )

    else:
        sys.exit(f"invalid step: {my_step}")
    return REFIT_PIP, re_y_pred, re_acc


def execute_model_workflow(
    my_n_features,
    my_n_trials,
    dfs,
    base_excluded,
    cus_drop_cols,
    y_label,
    training_round,
    my_step,
    trainer,
    training_configs,
    debug=False,
    base_n_features=0,
    test_acc=0,
):
    print(f"")
    print(f"")
    print(f"@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    df, y_title = get_df(dfs, y_label, training_round, my_step, trainer, training_configs, debug)
    # make filename contains top_{selected feature N}.{trainer algorithm}.{outcome label}_{base/final}_{predictor set}
    if debug:
        fname = f"debug_{my_n_features}.{'_'.join(trainer)}.{y_title}"
    else:
        if my_step != "final":
            fname = f"top_{my_n_features}.{'_'.join(trainer)}.{y_title}"
        else:
            fname = f"final_top_{my_n_features}.{'_'.join(trainer)}.{y_title}.base_{base_n_features}"
    print(f"Execute {my_step}: {fname}")
    print(f"@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

    drop_cols = base_excluded + cus_drop_cols
    print(f" - preview drop_cols: {drop_cols}")

    if df.shape[0] == 0:
        print(f"Zero individual in the cohort - aborted.")
        return None, None
    if trainer[1] == "classifier" and my_step == "base_fit":
        if df[y_title].value_counts().shape[0] == 1:
            print(f"Monomorphic - training aborted")
            return None, None
        if sum(df[y_title].astype(int) == 1) < 3:
            print(f"Cases less than 3 - training abored")
            return None, None

    if my_step == "base_fit":
        PIP, y_pred, acc = train_opt_pipeline(
            my_step, df, y_title, drop_cols, trainer, fname, my_n_features, my_n_trials
        )
        RE_PIP, re_y_pred, re_acc = retrain_best_pipeline(my_step, df, y_title, drop_cols, trainer, fname, PIP=PIP)
        return None, acc

    elif my_step == "base_predict":
        joblib_fp = f"base_pipeline.{fname}.joblib"
        my_joblib = Path(joblib_fp)
        if my_joblib.is_file():
            PIP = load(joblib_fp)
            PIP, y_pred, acc = train_opt_pipeline(
                my_step,
                df,
                y_title,
                drop_cols,
                trainer,
                fname,
                my_n_features,
                my_n_trials,
                PIP=PIP,
                test_acc=test_acc,
            )
        else:
            print(f"######################################")
            print(f"- {my_step} train_opt_pipeline incompleted: {joblib_fp} not found.")
            print(f"######################################")
            print(f"")
            return None, None

        retrain_joblib_fp = f"retrain_base_pipeline.{fname}.joblib"
        my_retrain_joblib = Path(retrain_joblib_fp)
        if my_retrain_joblib.is_file():
            PIP = load(retrain_joblib_fp)
            RE_PIP, re_y_pred, re_acc = retrain_best_pipeline(
                my_step,
                df,
                y_title,
                drop_cols,
                trainer,
                fname,
                PIP=PIP,
                test_acc=test_acc,
            )
        else:
            print(f"######################################")
            print(f"- {my_step} retrain_best_pipeline incompleted: {retrain_joblib_fp} not found.")
            print(f"######################################")
            print(f"")
            return None, None

        return re_y_pred, re_acc

    elif my_step == "final":
        PIP, y_pred, acc = train_opt_pipeline(
            my_step, df, y_title, drop_cols, trainer, fname, my_n_features, my_n_trials
        )
        RE_PIP, re_y_pred, re_acc = retrain_best_pipeline(my_step, df, y_title, drop_cols, trainer, fname, PIP=PIP)
        return re_y_pred, re_acc

    else:
        sys.exit(f"invalid step: {my_step}")
