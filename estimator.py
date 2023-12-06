#!/usr/bin/env python3

# from src.helper_dev import *

# np.random.seed(19920722)

# debug = False

# if debug == True:
#     my_y_lbl =  "sex_f31" 
#     my_step = "base" 
#     my_n_features = 10
#     my_n_trials = 10
#     my_pkg = "xgb"
#     my_final_round = "all" 
#     my_test = "221004_base"
#     my_base_n_features = 50
# else:
#     my_y_lbl = sys.argv[1]
#     my_step = sys.argv[2]
#     my_n_features = int(sys.argv[3])
#     my_pkg = sys.argv[4]
#     my_n_trials = 100
#     if my_step == "final":
#         my_n_trials = 100
#         my_final_round = sys.argv[5] 
#         my_test = sys.argv[6]
#         my_base_n_features = sys.argv[7]


# if my_step == "base":
#     my_final_round = None
#     print(f"")
#     print(f"######################################")
#     print(f"### Base model fitting and transforming: {my_y_lbl}")
#     print(f"######################################")
#     print(f"")
#     pre_df, pheno_df, col_dict, base_excluded = load_data(my_step) # 7min
# elif my_step == "final":
    # print(f"")
    # print(f"######################################")
    # print(f"### Final model training and testing: {my_y_lbl}")
    # print(f"### config: {my_final_round}")
    # print(f"######################################")
#     pre_df, pheno_df, col_dict, base_excluded = load_data(my_step, my_pkg, my_test, my_base_n_features) # 7min
# else:
#     sys.exit(f"invalid step {my_step}")

# # Getting usage of RAM
# print(f"\n\n# RAM memory used for loading inputs: {round(psutil.virtual_memory()[3]/((2**10)**3), 2)}GB ({psutil.virtual_memory()[2]}%)")

# if my_step == "base":
#     # base_fit
#     test_acc_dict = {}
#     dfs, y_label, cus_drop_cols, trainer, training_rounds, training_configs = load_config("base_fit", my_y_lbl, my_pkg, my_final_round, pre_df, col_dict) # 10sec
#     if debug == True:
#         print(f"##############\ndebug mode ON.\n##############")
#         training_rounds = [("base", "age_gender_only")]
#     for training_round in training_rounds:
#         test_y_pred, test_acc = main(my_n_features, my_n_trials, dfs, base_excluded, cus_drop_cols, y_label, training_round, "base_fit", trainer, training_configs, debug)
#         test_acc_dict[training_round] = test_acc 
#         print(f"\n\n# RAM memory used in base_fit: {round(psutil.virtual_memory()[3]/((2**10)**3), 2)}GB ({psutil.virtual_memory()[2]}%)")
        
#     # base_transform
#     dfs, y_label, cus_drop_cols, trainer, training_rounds, training_configs = load_config("base_predict", my_y_lbl, my_pkg, my_final_round, pre_df, col_dict) # 10sec
#     if debug == True:
#         print(f"##############\ndebug mode ON.\n##############")
#         training_rounds = [("base", "age_gender_only")]
#     for training_round in training_rounds:
#         y_pred, pred_acc = main(my_n_features, my_n_trials, dfs, base_excluded, cus_drop_cols, y_label, training_round, "base_predict", trainer, training_configs, debug, test_acc = test_acc_dict[training_round]) 
#         print(f"\n\n# RAM memory used in base_predict: {round(psutil.virtual_memory()[3]/((2**10)**3), 2)}GB ({psutil.virtual_memory()[2]}%)")
        
# elif my_step == "final":
#     dfs, y_label, cus_drop_cols, trainer, training_rounds, training_configs = load_config("final", my_y_lbl, my_pkg, my_final_round, pre_df, col_dict)
#     y_pred, acc = main(my_n_features, my_n_trials, dfs, base_excluded, cus_drop_cols, y_label, training_rounds[0], "final", trainer, training_configs, debug, base_n_features = my_base_n_features)
    
# else:
#     sys.exit(f"invalid step {my_step}")

    
    
    
    
    
#!/usr/bin/env python3

import sys
import psutil
import argparse
import numpy as np
import logging
from utils import *

def parse_arguments():
    parser = argparse.ArgumentParser(description="Model training and testing script.",
                                     usage="%(prog)s --weights <weights_table.csv/.txt> --vcf <imputed_file_chr{}.vcf.gz> --chrom <int> --name <subject_name> --max_chunk_size <int> --wgs_mode <true/false>\nUse -h or --help to display help.")
    
    parser.add_argument('--y_lbl', required=True, help='Label Y')
    parser.add_argument('--step', default='base', choices=['base', 'final'], help='Step to execute')
    parser.add_argument('--n_features', type=int, default=10, help='Number of features')
    parser.add_argument('--pkg', default='xgb', help='Package to use')
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
