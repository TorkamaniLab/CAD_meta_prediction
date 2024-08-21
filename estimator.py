#!/usr/bin/env python3

import sys
import psutil
import argparse
import numpy as np
import logging
from utils import *

def parse_arguments():
    parser = argparse.ArgumentParser(description="Model training and testing script.",
                                     usage="%(prog)s --y_lbl --step ['base', 'final'] --n_features <int> --pkg ['xgb', 'cat', 'lgb']  --final_round <str> --test <str> --base_n_features <int> --n_trials <int>\nUse -h or --help to display help.")
    
    parser.add_argument('--y_lbl', required=True, help='Label Y')
    parser.add_argument('--step', default='base', choices=['base', 'final'], help='Step to execute')
    parser.add_argument('--n_features', type=int, default=10, help='Number of features')
    parser.add_argument('--pkg', default='xgb', choices=['xgb', 'cat', 'lgb'], help='Package to use')
    parser.add_argument('--final_round', default=None, help='Final round configuration')
    parser.add_argument('--test', default=None, help='Test configuration')
    parser.add_argument('--base_n_features', type=int, default=50, help='Base number of features')
    parser.add_argument('--n_trials', type=int, default=100, help='Number of trials')
    return parser.parse_args()

def setup_logging():
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def log_ram_usage(phase):
    ram_usage = round(psutil.virtual_memory()[3]/((2**10)**3), 2)
    ram_percent = psutil.virtual_memory()[2]
    logging.debug(f"RAM memory used in {phase}: {ram_usage}GB ({ram_percent}%)")

def main():
    args = parse_arguments()
    setup_logging()

    np.random.seed(19920722)

    if args.step == "base":
        print(f"")
        print(f"######################################")
        print(f"### Base model fitting and transforming: {my_y_lbl}")
        print(f"######################################")
        pre_df, pheno_df, col_dict, base_excluded = load_data(args.step)
        log_ram_usage("data loading")
    elif args.step == "final":
        print(f"")
        print(f"######################################")
        print(f"### Final model training and testing: {my_y_lbl}")
        print(f"### config: {my_final_round}")
        print(f"######################################")
        pre_df, pheno_df, col_dict, base_excluded = load_data(args.step, args.pkg, args.test, args.base_n_features)
        log_ram_usage("data loading")
    else:
        sys.exit(f"Invalid step {args.step}")

    if args.step == "base":
        test_acc_dict = {}
        dfs, y_label, cus_drop_cols, trainer, training_rounds, training_configs = load_config("base_fit", args.y_lbl, args.pkg, args.final_round, pre_df, col_dict)
        for training_round in training_rounds:
            test_y_pred, test_acc = execute_model_workflow(args.n_features, args.n_trials, dfs, base_excluded, cus_drop_cols, y_label, training_round, "base_fit", trainer, training_configs)
            test_acc_dict[training_round] = test_acc 
            log_ram_usage("base_fit")

        dfs, y_label, cus_drop_cols, trainer, training_rounds, training_configs = load_config("base_predict", args.y_lbl, args.pkg, args.final_round, pre_df, col_dict)
        for training_round in training_rounds:
            y_pred, pred_acc = execute_model_workflow(args.n_features, args.n_trials, dfs, base_excluded, cus_drop_cols, y_label, training_round, "base_predict", trainer, training_configs, test_acc = test_acc_dict[training_round])
            log_ram_usage("base_predict")

    elif args.step == "final":
        dfs, y_label, cus_drop_cols, trainer, training_rounds, training_configs = load_config("final", args.y_lbl, args.pkg, args.final_round, pre_df, col_dict)
        y_pred, acc = execute_model_workflow(args.n_features, args.n_trials, dfs, base_excluded, cus_drop_cols, y_label, training_rounds[0], "final", trainer, training_configs, base_n_features = args.base_n_features)
        log_ram_usage("final")

    else:
        sys.exit(f"Invalid step {args.step}")

if __name__ == "__main__":
    
    print('\n'.join(['\t',
         '\t##################################', 
         '\t#                                #',
         '\t#      CAD Meta-prediction       #',
         '\t#                                #',
         '\t#      Torkamani_Lab             #',
         '\t#                                #',
         '\t#  Main Contributor: Shaun Chen  #',
         '\t#  Last Modified:    2023-12-05  #',
         '\t#                                #',
         '\t##################################',
         '\t',
        ]))
    
    main()
