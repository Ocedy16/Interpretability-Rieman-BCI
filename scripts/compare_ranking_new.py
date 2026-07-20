from scipy.stats import spearmanr, weightedtau, kendalltau, pearsonr
import numpy as np
import pickle
from moabb.datasets import BNCI2014_001, Dreyer2023C, Beetl2021_A
from src.data.dataset_config import DATASET_CONFIG
from itertools import combinations
import pandas as pd
import rbo
import argparse

def topk_overlap(importance_A, importance_B, k=5):
    top_k_A = set(np.argsort(importance_A)[-k:])
    top_k_B = set(np.argsort(importance_B)[-k:])
    print(SENSORS[np.argsort(importance_A)[:k]])
    print(SENSORS[np.argsort(importance_B)[:k]])
    
    overlap = len(top_k_A & top_k_B)
    return overlap / k 

def importance_to_ranking(importance_values, channel_names):
    sorted_idx = np.argsort(importance_values)[::-1] 
    return [channel_names[i] for i in sorted_idx]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        choices=list(DATASET_CONFIG.keys()),
        required=True,
        help="Dataset à utiliser : BNCI2014_001, Dreyer2023C, Beetl2021_A"
    )
    parser.add_argument(
        "--clf",
        type=str,
        choices=["MDM", "TSClassifier"],
        required=True,
        help="Classifieur à utiliser : MDM, TSClassifier"
    )
    parser.add_argument(
        "--method",
        type = str,
        required = True,
        help = "Méthodes possibles : Shapley ou Permutation"
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    cfg = DATASET_CONFIG[args.dataset]
    dataset = cfg["dataset"]
    SENSORS = np.array(cfg["sensors"])

    OUT_DIR = f'../../Results/Permutation/{args.clf}/{args.dataset}/With_cue'
    OUT_DIR_SHAP = f"../../Results/Shapley/MDM/{args.dataset}"


    if args.method == "Permutation":
        eeg_importance = np.load(f"{OUT_DIR}/Modif_eeg/results_dic.pkl",allow_pickle=True)
        covs_importance =  np.load(f"{OUT_DIR}/Modif_covs/feature_perm_importances_covs_matrix.npy")
        heuristic_importance = np.load(f"{OUT_DIR}/Modif_covs/Heuristic/feature_perm_importances_covs_matrix.npy")
        eeg_importance = [np.mean(eeg_importance['across_times'][i]["importance"],axis=0) for i in dataset.subject_list]
        corrs_importance =  np.load(f"{OUT_DIR}/Correlation/feature_perm_importances_covs_matrix.npy")
        corrs_importance = np.mean(corrs_importance,axis=1)
        covs_importance = np.mean(covs_importance,axis=1)
        heuristic_importance = np.mean(heuristic_importance,axis=1)
        print("hey")
        importance_methods = {
                        'covs':covs_importance,
                        'corrs':corrs_importance}

        corrs = {}

        results = {subject: {} for subject in dataset.subject_list}

        for method_A, method_B in combinations(importance_methods.keys(), 2):
            for i, subject in enumerate(dataset.subject_list):
                importance_A = importance_methods[method_A][i]
                importance_B = importance_methods[method_B][i]
                
                rbo_val = rbo.RankingSimilarity(importance_to_ranking(importance_A,SENSORS), importance_to_ranking(importance_B,SENSORS)).rbo(p=0.8)
                rho,_ = spearmanr(importance_A,importance_B)
                topk = topk_overlap(importance_A, importance_B)
                kentau,_ = kendalltau(importance_A, importance_B)
                weighted_kentau,_ = weightedtau(importance_A, importance_B)
                pearson,_ = pearsonr(importance_A,importance_B)
                
                results[subject][(method_A, method_B)] = {"rho": rho, "topk": topk, "rbo" : rbo_val, 'kentau' : kentau, 'weighted_kentau' : weighted_kentau, 'pearson' : pearson}

        rows = []
        for subject in dataset.subject_list:
            row = {"subject": subject}
            for pair, values in results[subject].items():
                row[f"rho_{pair[0]}_{pair[1]}"] = values["rho"]
                row[f"topk_{pair[0]}_{pair[1]}"] = values["topk"]
                row[f"rbo_{pair[0]}_{pair[1]}"] = values["rbo"]
                row[f"kentau_{pair[0]}_{pair[1]}"] = values["kentau"]
                row[f"weighted_kentau_{pair[0]}_{pair[1]}"] = values["weighted_kentau"]
                row[f"pearson_{pair[0]}_{pair[1]}"] = values["pearson"]
            rows.append(row)

        df = pd.DataFrame(rows)
        print(df)
        df.to_csv(f"{OUT_DIR}/Rankings_spd_corr.csv")


    if args.method == "Shapley" :
        rhos = {}
        topks = {}

        results = {}
        shapley_eeg = np.load(f"{OUT_DIR_SHAP}/New_results_n_samples_1000/all_shap_values.pkl", allow_pickle=True)
        shapley_eeg = -np.mean(np.mean(shapley_eeg,axis=1),axis=1)[:,0,:]
        shapley_test = np.load(f"{OUT_DIR_SHAP}/Tests/all_shap_values.pkl", allow_pickle=True)
        shapley_test= np.mean(np.mean(shapley_test,axis=1),axis=1)[:,0,:]
        shapley_spd = np.load(f"{OUT_DIR_SHAP}/Shapley_SPD/all_shap_values.pkl",allow_pickle=True)
        shapley_spd = np.mean(np.mean(shapley_spd,axis=1),axis=1)[:,0,:]
        shapley_optim = np.load(f"{OUT_DIR_SHAP}/Shapley_optim/Lambda_1/all_subjects_mdm.npy")
        shapley_optim = np.mean(shapley_optim, axis=1)
        shapley_corr = np.load(f"{OUT_DIR_SHAP}/Shapley_SPD/Correlation/all_shap_values.pkl",allow_pickle=True)
        shapley_corr = np.mean(np.mean(shapley_corr,axis=1),axis=1)[:,0,:]
        shapley_heuristic = np.load(f"{OUT_DIR_SHAP}/Shapley_SPD/Heuristic/all_shap_values.pkl",allow_pickle=True)
        shapley_heuri = np.mean(np.mean(shapley_heuristic,axis=1),axis=1)[:,0,:]
        print(shapley_spd.shape,shapley_eeg.shape, shapley_optim.shape)

        importance_methods_shap = {'spd':shapley_spd,
                                    'corr':shapley_corr}

        results = {subject: {} for subject in dataset.subject_list}

        for method_A, method_B in combinations(importance_methods_shap.keys(), 2):
            for i, subject in enumerate(dataset.subject_list):
                importance_A = importance_methods_shap[method_A][i]
                importance_B = importance_methods_shap[method_B][i]
                print(importance_B.shape)
                
                rbo_val = rbo.RankingSimilarity(importance_to_ranking(importance_A,SENSORS), importance_to_ranking(importance_B,SENSORS)).rbo(p=0.8)
                rho,_ = spearmanr(importance_A,importance_B)
                topk = topk_overlap(importance_A, importance_B)
                kentau,_ = kendalltau(importance_A, importance_B)
                weighted_kentau,_ = weightedtau(importance_A, importance_B)
                pearson,_ = pearsonr(importance_A,importance_B)
                
                results[subject][(method_A, method_B)] = {"rho": rho, "topk": topk, "rbo" : rbo_val, 'kentau' : kentau, 'weighted_kentau' : weighted_kentau, 'pearson' : pearson}

        rows = []
        for subject in dataset.subject_list:
            row = {"subject": subject}
            for pair, values in results[subject].items():
                row[f"rho_{pair[0]}_{pair[1]}"] = values["rho"]
                row[f"topk_{pair[0]}_{pair[1]}"] = values["topk"]
                row[f"rbo_{pair[0]}_{pair[1]}"] = values["rbo"]
                row[f"kentau_{pair[0]}_{pair[1]}"] = values["kentau"]
                row[f"weighted_kentau_{pair[0]}_{pair[1]}"] = values["weighted_kentau"]
                row[f"pearson_{pair[0]}_{pair[1]}"] = values["pearson"]
            rows.append(row)

        df = pd.DataFrame(rows)
        print(df)
        df.to_csv(f"{OUT_DIR_SHAP}/Rankings_spd_corrs.csv")

