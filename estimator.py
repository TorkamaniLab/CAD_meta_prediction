#!/usr/bin/env python3

import argparse
import logging
import sys

import numpy as np

from utils import execute_model_workflow, load_config, load_data


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Model training and testing script.",
        usage=(
            "%(prog)s --y_lbl --step ['base', 'final'] --n_features <int> "
            "--pkg ['xgb', 'cat', 'lgb'] --final_round <str> --test <str> "
            "--base_n_features <int> --n_trials <int>\nUse -h or --help to display help."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,  # ✨ 這行加上去
    )

    parser.add_argument("--y_lbl", required=True, help="Label Y", metavar="LABEL")
    parser.add_argument("--step", default="base", choices=["base", "final"], help="Step to execute", metavar="STEP")
    parser.add_argument("--n_features", type=int, default=10, help="Number of features", metavar="N")
    parser.add_argument("--pkg", default="xgb", choices=["xgb", "cat", "lgb"], help="Package to use", metavar="PKG")
    parser.add_argument("--final_round", default=None, help="Final round configuration", metavar="STR")
    parser.add_argument("--test", default=None, help="Test configuration", metavar="STR")
    parser.add_argument("--base_n_features", type=int, default=50, help="Base number of features", metavar="N")
    parser.add_argument("--n_trials", type=int, default=100, help="Number of trials", metavar="N")
    return parser.parse_args()


def setup_logging():
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")


def main():
    args = parse_arguments()
    setup_logging()

    np.random.seed(19920722)

    if args.step == "base":
        print(f"")
        print(f"######################################")
        print(f"### Base model fitting and transforming: {args.y_lbl}")
        print(f"######################################")
        pre_df, pheno_df, col_dict, base_excluded = load_data(args.step)
    elif args.step == "final":
        print(f"")
        print(f"######################################")
        print(f"### Final model training and testing: {args.y_lbl}")
        print(f"### config: {args.final_round}")
        print(f"######################################")
        pre_df, pheno_df, col_dict, base_excluded = load_data(args.step, args.pkg, args.test, args.base_n_features)
    else:
        sys.exit(f"Invalid step {args.step}")

    if args.step == "base":
        test_acc_dict = {}
        dfs, y_label, cus_drop_cols, trainer, training_rounds, training_configs = load_config(
            "base_fit", args.y_lbl, args.pkg, args.final_round, pre_df, col_dict
        )
        for training_round in training_rounds:
            test_y_pred, test_acc = execute_model_workflow(
                args.n_features,
                args.n_trials,
                dfs,
                base_excluded,
                cus_drop_cols,
                y_label,
                training_round,
                "base_fit",
                trainer,
                training_configs,
            )
            test_acc_dict[training_round] = test_acc

        dfs, y_label, cus_drop_cols, trainer, training_rounds, training_configs = load_config(
            "base_predict", args.y_lbl, args.pkg, args.final_round, pre_df, col_dict
        )
        for training_round in training_rounds:
            y_pred, pred_acc = execute_model_workflow(
                args.n_features,
                args.n_trials,
                dfs,
                base_excluded,
                cus_drop_cols,
                y_label,
                training_round,
                "base_predict",
                trainer,
                training_configs,
                test_acc=test_acc_dict[training_round],
            )

    elif args.step == "final":
        dfs, y_label, cus_drop_cols, trainer, training_rounds, training_configs = load_config(
            "final", args.y_lbl, args.pkg, args.final_round, pre_df, col_dict
        )
        y_pred, acc = execute_model_workflow(
            args.n_features,
            args.n_trials,
            dfs,
            base_excluded,
            cus_drop_cols,
            y_label,
            training_rounds[0],
            "final",
            trainer,
            training_configs,
            base_n_features=args.base_n_features,
        )

    else:
        sys.exit(f"Invalid step {args.step}")


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
                "\t#  Last Modified:    2023-12-05  #",
                "\t#                                #",
                "\t##################################",
                "\t",
            ]
        )
    )

    main()
