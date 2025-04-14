#!/usr/bin/env python3

import argparse
import logging

import numpy as np

from src.utils import execute_model

np.random.seed(19920722)
logging.basicConfig(level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Model training and testing script.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--y_label", required=True, help="Label Y", metavar="LABEL")
    parser.add_argument("--input_pickle_fp", required=True, help="Input PICKLE file path", metavar="INPUT")
    parser.add_argument("--id_col", required=True, help="ID column", metavar="ID_COL")
    parser.add_argument(
        "--pkg",
        default="xgb",
        choices=["xgb", "cat", "lgb"],
        help="Package to use: one of {xgb, cat, lgb}",
        metavar="PKG",
    )
    parser.add_argument(
        "--estimator_type",
        default="classifier",
        choices=["classifier", "regressor"],
        help="Type of estimator: one of {classifier, regressor}",
        metavar="EST_TYPE",
    )
    parser.add_argument("--n_features", type=int, default=10, help="Number of features", metavar="N")
    parser.add_argument("--n_trials", type=int, default=100, help="Number of hyperparameter tuning trials", metavar="N")
    return parser.parse_args()


def main():
    args = parse_arguments()

    execute_model(
        y_label=args.y_label,
        input_pickle_fp=args.input_pickle_fp,
        id_col=args.id_col,
        n_features=args.n_features,
        n_trials=args.n_trials,
        trainer=(args.pkg, args.estimator_type),
    )


if __name__ == "__main__":
    print(
        "\n".join(
            [
                "\t",
                "\t##################################",
                "\t#                                #",
                "\t#      CAD Meta-prediction       #",
                "\t#                                #",
                "\t#      Torkamani_Lab             #",
                "\t#                                #",
                "\t#  Main Contributor: Shaun Chen  #",
                "\t#  Last Modified:    2025-04-14  #",
                "\t#                                #",
                "\t##################################",
                "\t",
            ]
        )
    )

    main()
