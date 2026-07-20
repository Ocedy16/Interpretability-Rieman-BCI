import torch
import pymanopt
import numpy as np
from pyriemann.classification import MDM, TSClassifier
from pyriemann.estimation import Covariances
from moabb.paradigms import FilterBankMotorImagery
from moabb.datasets import BNCI2014_001, Dreyer2023C, Beetl2021_A, AlexMI
from sklearn.model_selection import train_test_split
from src.Visualization.topomap import plot_topomap, plot_pannel
from src.Shapley.shapley_spd import (
    proj_on_spd, stable_predict, compute_shapley_heuristic,
    percent_spd, spd_values, eigenval_pos
)
from src.data.dataset_config import DATASET_CONFIG
from sklearn.pipeline import Pipeline
from functools import partial
import math
import matplotlib.pyplot as plt
import os
import pickle


import torch.nn as nn
from src.SPDNet.SPDNet import SPDNetBatchNorm, train_model
from spd_learn.modules import BiMap, CovLayer, LogEig, ReEig, SPDBatchNormMeanVar
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import LabelEncoder
from spd_learn.functional import covariance
from warnings import warn

np.random.seed(3)
torch.manual_seed(0)

import argparse


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
        "--n_splits",
        type=int,
        default=10,
        help="Nombre de splits (défaut: 10)"
    )

    parser.add_argument(
        "--visu_only",
        type=bool,
        default=False,
        help="Si False, le programme générera les valeurs de Shapley et produira les heatmaps correspondantes.\nSi False, il produira des heatmaps à partir de fichiers existants."
    )

    parser.add_argument(
        "--eigenval_info",
        type=bool,
        default=False,
        help="Si True, le programme affichera des informations sur les valeurs propres des matrices avant projection."
    )

    parser.add_argument(
        "--savefile",
        type=str,
        default = None,
        help = "Le chemin où tu veux sauvegarder ton fichier"
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    cfg = DATASET_CONFIG[args.dataset]
    dataset = cfg["dataset"]
    SENSORS = cfg["sensors"]

    OUT_DIR = f"../../Results/Permutation/SPD_Net/{args.dataset}/Heuristique"

    N_SPLITS = args.n_splits
    SCORE_THRESHOLD = 0.75


    paradigm = FilterBankMotorImagery(filters=[[7, 35]], events={"left_hand": 1, "right_hand": 2})
    N_SPLITS = args.n_splits

    os.makedirs(OUT_DIR, exist_ok=True) 

    scores_per_subject = []
    importances_per_subject = []


    for subject in dataset.subject_list:
        print(f"Processing subject {subject}...")
        X, y, meta = paradigm.get_data(dataset=dataset, subjects=[subject])

        le = LabelEncoder()
        y_encoded = torch.tensor(le.fit_transform(y), dtype=torch.long)

        model_config = {
            "n_chans": X.shape[1],
            "n_outputs": 2,
            "subspacedim": 16,
            "bn": None,
        }


        scores_per_split = []
        importances_per_split = []

        for i in range(N_SPLITS):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, stratify = y_encoded, test_size=0.2, random_state=i
            )

            rng = np.random.default_rng(i)
            covs_transformer = Covariances()
            C_train = covs_transformer.fit_transform(X_train) 
            C_test = covs_transformer.transform(X_test)

            C_train_t = torch.tensor(C_train).float()
            C_test_t = torch.tensor(C_test).float()

            model = SPDNetBatchNorm(**model_config, input_type="cov").float()
            train_model(model, C_train_t, y_train, C_test_t, y_test)
            model.eval()
            with torch.no_grad():
                score_initial = (model(C_test_t).argmax(1) == y_test).float().mean().item()

            importances = []
            nb_feat = len(C_train[0])

            for feature in range(nb_feat):
                perm_C_test = []
                for matrix in C_test: 
                    matrix_mod = matrix.copy()
                    matrix_mod[feature,:] = 0
                    matrix_mod[:,feature] = 0
                    matrix_mod[feature,feature] = matrix[feature,feature]
                    matrix_mod = proj_on_spd(matrix_mod)
                    perm_C_test.append(matrix_mod)
                perm_C_test = np.array(perm_C_test)
                perm_C_test_t = torch.tensor(perm_C_test).float()
                model.eval()
                with torch.no_grad():
                    score = (model(perm_C_test_t).argmax(1) == y_test).float().mean().item()
                importances.append(score_initial - score)


            scores_per_split.append(score_initial)
            importances_per_split.append(importances)

        scores_per_subject.append(scores_per_split)
        importances_per_subject.append(importances_per_split)

    np.save(f'{OUT_DIR}/feature_perm_importances_covs_matrix.npy', importances_per_subject)
    np.save(f'{OUT_DIR}/feature_perm_scores_covs_matrix.npy', scores_per_subject)
    np.save(f'{OUT_DIR}/percent_spd.npy',np.array(percent_spd))
    #np.save(f'{OUT_DIR}/spd_values.npy',np.array(spd_values))
    #np.save(f'{OUT_DIR}/eigenval_pos.npy',np.array(eigenval_pos))


    scores = np.load(f'{OUT_DIR}/feature_perm_scores_covs_matrix.npy')
    importances = np.load(f'{OUT_DIR}/feature_perm_importances_covs_matrix.npy')

    scores_reduced = np.mean(scores, axis=1)
    importances_reduced = np.mean(importances, axis=1)


    plot_pannel(importances_reduced, dataset, SENSORS, scores_reduced, OUT_DIR,cbar_type = "Permutation", suptitle = f"Feature Permutation Importance on {args.dataset}")


    good_subjects_perm = []
    good_subjects_scores = []
    bad_subjects_perm = []
    bad_subjects_scores = []

    for i, subject in enumerate(dataset.subject_list): 

        if scores_reduced[i] >= SCORE_THRESHOLD : 
            good_subjects_perm.append(importances_reduced[i])
            good_subjects_scores.append(scores_reduced[i])

        else: 
            bad_subjects_perm.append(importances_reduced[i])
            bad_subjects_scores.append(scores_reduced[i])
        
    np.save(f'{OUT_DIR}/good_subjects.npy', np.array(good_subjects_perm))
    np.save(f'{OUT_DIR}/bad_subjects.npy', np.array(bad_subjects_perm))
    np.save(f'{OUT_DIR}/good_subjects_scores.npy', np.array(good_subjects_scores))
    np.save(f'{OUT_DIR}/bad_subjects_scores.npy', np.array(bad_subjects_scores))


    #good_subjects_perm = np.load(f'{OUT_DIR}/good_subjects.npy')
    #bad_subjects_perm = np.load(f'{OUT_DIR}/bad_subjects.npy')
    #good_subjects_scores = np.load(f'{OUT_DIR}/good_subjects_scores.npy')
    #bad_subjects_scores = np.load(f'{OUT_DIR}/bad_subjects_scores.npy')

    v_max = np.max(np.abs(np.mean(np.array(good_subjects_perm),axis=0)))
    v_min = -v_max

    plot_topomap(np.mean(np.array(good_subjects_perm),axis=0),SENSORS,
                title = f"Subjects with a classif score $\\geq$ {SCORE_THRESHOLD}\n Mean Score : {np.mean(np.array(good_subjects_scores)):.2f}",
                vlim=(v_min,v_max),
                savefile_name=f'{OUT_DIR}/Good_subjects.pdf',
                cbar_type = "Permutation", suptitle = f"Feature Permutation Importance on {args.dataset}")


    plot_topomap(np.mean(np.array(bad_subjects_perm),axis=0),SENSORS,
                title = f"Subjects with a classif score $< {SCORE_THRESHOLD}$\n Mean Score : {np.mean(np.array(bad_subjects_scores)):.2f}",vlim=(v_min,v_max),
                savefile_name=f'{OUT_DIR}/Bad_subjects.pdf',
                cbar_type = "Permutation", suptitle = f"Feature Permutation Importance on {args.dataset}")



