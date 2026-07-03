from scipy.stats import spearmanr, weightedtau, kendalltau, pearsonr
import numpy as np
import pickle
from moabb.datasets import BNCI2014_001, Dreyer2023C, Beetl2021_A
from itertools import combinations
import pandas as pd
import rbo

DATASET_CONFIG = {
    "BNCI2014_001": dict(
        dataset=BNCI2014_001(),
        session="0train",
        good_subjects=np.array([1, 3, 8, 9]),
        bad_subjects=np.array([2, 4, 5, 6, 7]),
        sensors =  [
        "Fz",
        "FC3",
        "FC1",
        "FCz",
        "FC2",
        "FC4",
        "C5",
        "C3",
        "C1",
        "Cz",
        "C2",
        "C4",
        "C6",
        "CP3",
        "CP1",
        "CPz",
        "CP2",
        "CP4",
        "P1",
        "Pz",
        "P2",
        "POz",
    ]),

    "Dreyer2023C": dict(
        dataset=Dreyer2023C(),
        session="0",
        good_subjects=np.array([83, 85, 87]),
        bad_subjects=np.array([82,84,86]),
        sensors = [
    'Fz', 'FCz', 'Cz', 'CPz', 'Pz', 'C1', 'C3', 'C5', 'C2', 'C4', 'C6', 'F4', 'FC2', 'FC4', 'FC6', 'CP2',
  'CP4', 'CP6', 'P4', 'F3', 'FC1', 'FC3', 'FC5', 'CP1', 'CP3', 'CP5', 'P3'
]
    ),
    "Beetl2021_A": dict(
        dataset=Beetl2021_A(),
        session="0",
        good_subjects=np.array([1, 3]),
        bad_subjects=np.array([2]),
        sensors = [
                "Fp1",
                "Fz",
                "F3",
                "F7",
                "FT9",
                "FC5",
                "FC1",
                "C3",
                "T7",
                "TP9",
                "CP5",
                "CP1",
                "Pz",
                "P3",
                "P7",
                "O1",
                "Oz",
                "O2",
                "P4",
                "P8",
                "TP10",
                "CP6",
                "CP2",
                "C4",
                "T8",
                "FT10",
                "FC6",
                "FC2",
                "F4",
                "F8",
                "Fp2",
                "AF7",
                "AF3",
                "AFz",
                "F1",
                "F5",
                "FT7",
                "FC3",
                "FCz",
                "C1",
                "C5",
                "TP7",
                "CP3",
                "P1",
                "P5",
                "PO7",
                "PO3",
                "POz",
                "PO4",
                "PO8",
                "P6",
                "P2",
                "CPz",
                "CP4",
                "TP8",
                "C6",
                "C2",
                "FC4",
                "FT8",
                "F6",
                "F2",
                "AF4",
                "AF8",
            ],
    ),
}

cfg = DATASET_CONFIG["Dreyer2023C"]
dataset = cfg["dataset"]
session = cfg["session"]
good_subjects = cfg["good_subjects"]
bad_subjects = cfg["bad_subjects"]
SENSORS = np.array(cfg["sensors"])

method = "permutation"

OUT_DIR= "../../Results/Permutation/MDM/Dreyer2023C/With_cue"
OUT_DIR_SHAP = "../../Results/Shapley/MDM/Dreyer2023C"



def topk_overlap(importance_A, importance_B, k=5):
    top_k_A = set(np.argsort(importance_A)[-k:])
    top_k_B = set(np.argsort(importance_B)[-k:])
    print(SENSORS[np.argsort(importance_A)[-k:]])
    print(SENSORS[np.argsort(importance_B)[-k:]])
    
    overlap = len(top_k_A & top_k_B)
    return overlap / k 

def importance_to_ranking(importance_values, channel_names):
    sorted_idx = np.argsort(importance_values)[::-1] 
    return [channel_names[i] for i in sorted_idx]


if method == "permutation":
    eeg_importance = np.load(f"{OUT_DIR}/Modif_EEG/results_dic.pkl",allow_pickle=True)
    covs_importance =  np.load(f"{OUT_DIR}/Modif_covs/feature_perm_importances_covs_matrix.npy")

    eeg_importance = [np.mean(eeg_importance['across_times'][i]["importance"],axis=0) for i in dataset.subject_list]
    covs_importance = np.mean(covs_importance,axis=1)

    importance_methods = {'eeg':eeg_importance,
                      'covs':covs_importance}

    corrs = {}

    results = {subject: {} for subject in dataset.subject_list}

    for method_A, method_B in combinations(importance_methods.keys(), 2):
        for i, subject in enumerate(dataset.subject_list):
            importance_A = importance_methods[method_A][i]
            importance_B = importance_methods[method_B][i]
            
            rbo_val = rbo.RankingSimilarity(importance_to_ranking(importance_A,SENSORS), importance_to_ranking(importance_B,SENSORS)).rbo()
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
    df.to_csv(f"{OUT_DIR}/Rankings.csv")


if method == "shapley" :
    rhos = {}
    topks = {}

    results = {}
    shapley_eeg = np.load(f"{OUT_DIR_SHAP}/New_results_n_samples_1000/all_shap_values.pkl", allow_pickle=True)
    shapley_eeg = np.mean(np.mean(shapley_eeg,axis=1),axis=1)[:,0,:]
    shapley_spd = np.load(f"{OUT_DIR_SHAP}/Shapley_SPD/all_shap_values.pkl",allow_pickle=True)
    shapley_spd = -np.mean(np.mean(shapley_spd,axis=1),axis=1)[:,0,:]
    shapley_optim = np.load(f"{OUT_DIR_SHAP}/Shapley_optim/Lambda_1/all_subjects_mdm.npy")
    shapley_optim = np.mean(shapley_optim, axis=1)
    print(shapley_spd.shape,shapley_eeg.shape, shapley_optim.shape)

    importance_methods_shap = {'eeg':shapley_eeg,
                                'optim':shapley_spd}

    results = {subject: {} for subject in dataset.subject_list}

    for method_A, method_B in combinations(importance_methods_shap.keys(), 2):
        for i, subject in enumerate(dataset.subject_list):
            importance_A = importance_methods_shap[method_A][i]
            importance_B = importance_methods_shap[method_B][i]
            
            rbo_val = rbo.RankingSimilarity(importance_to_ranking(importance_A,SENSORS), importance_to_ranking(importance_B,SENSORS)).rbo()
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
    df.to_csv(f"{OUT_DIR_SHAP}/Rankings_eeg_spd_inverse.csv")

