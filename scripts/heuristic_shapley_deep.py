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
        default=1,
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

    OUT_DIR = f"../../Results/Shapley/SPD_Net/{args.dataset}/Heuristic"

    N_SPLITS = args.n_splits
    SCORE_THRESHOLD = 0.75


    paradigm = FilterBankMotorImagery(filters=[[7, 35]], events={"left_hand": 1, "right_hand": 2})
    N_SPLITS = args.n_splits

    os.makedirs(OUT_DIR, exist_ok=True) 

    all_subjects_shap_values = []
    all_mean_shap_values = []
    all_scores = []
    good_subjects_shap = []
    good_subjects_scores = []
    bad_subjects_shap = []
    bad_subjects_scores = []

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

        shap_values, scores = compute_shapley_heuristic(X,y_encoded,N_SPLITS, model_config, deep=True)

        #On moyenne sur tous les splits
        mean_split_shap_values = np.mean(np.array(shap_values),axis=0)

        #On retire l'axe inutile au milieu et on moyenne sur tous les exemples
        mean_shap_values = np.mean(mean_split_shap_values,axis=0)[0]

        mean_score = np.mean(np.array(scores))

        all_scores.append(mean_score)
        all_subjects_shap_values.append(shap_values)
        all_mean_shap_values.append(mean_shap_values)
        if mean_score > SCORE_THRESHOLD : 
            good_subjects_shap.append(mean_shap_values)
            good_subjects_scores.append(mean_score)

        else : 
            bad_subjects_shap.append(mean_shap_values)
            bad_subjects_scores.append(mean_score)

    filename = os.path.join(OUT_DIR, 'all_shap_values.pkl') 
    with open(filename,'wb') as f: 
        pickle.dump(all_subjects_shap_values,f)

    np.save(f"{OUT_DIR}/all_scores.npy", np.array(all_scores))    
    np.save(f"{OUT_DIR}/good_subjects.npy", np.array(good_subjects_shap))
    np.save(f"{OUT_DIR}/bad_subjects.npy", np.array(bad_subjects_shap))
    np.save(f"{OUT_DIR}/good_subjects_scores.npy", np.array(good_subjects_scores))
    np.save(f"{OUT_DIR}/bad_subjects_scores.npy", np.array(bad_subjects_scores))


    plot_pannel(all_mean_shap_values,dataset, SENSORS, all_scores, OUT_DIR, suptitle = f'Shapley values on SPDNet')

    good_subjects_shap = np.load(f"{OUT_DIR}/good_subjects.npy",)
    bad_subjects_shap = np.load(f"{OUT_DIR}/bad_subjects.npy")
    good_subjects_scores = np.load(f"{OUT_DIR}/good_subjects_scores.npy")
    bad_subjects_scores = np.load(f"{OUT_DIR}/bad_subjects_scores.npy")
    
    v_max = np.max(np.abs(np.mean(np.array(good_subjects_shap), axis=0)))
    v_min = -v_max

    plot_topomap(np.mean(np.array(good_subjects_shap),axis=0),SENSORS,
                title = f"Subjects with a classif score $\\geq$ {SCORE_THRESHOLD}\n Mean Score : {np.mean(np.array(good_subjects_scores)):.2f}",
                vlim=(v_min,v_max),
                savefile_name=f"{OUT_DIR}/Good_subjects.pdf",
                cmap = 'PiYG',
                suptitle = f'Shapley values on SPDNet')


    plot_topomap(np.mean(np.array(bad_subjects_shap),axis=0),SENSORS,
                title = f"Subjects with a classif score $< {SCORE_THRESHOLD}$\n Mean Score : {np.mean(np.array(bad_subjects_scores)):.2f}",
                vlim=(v_min,v_max),
                savefile_name=f"{OUT_DIR}/Bad_subjects.pdf",
                cmap = 'PiYG',
                suptitle = f'Shapley values on SPDNet')


    if args.eigenval_info:
        percent = np.load(f'{OUT_DIR}/percent_spd.npy')
        eigenvalues_neg = np.load(f'{OUT_DIR}/spd_values.npy')
        eigenval_pos = np.load(f'{OUT_DIR}/eigenval_pos.npy')
        print("Percentage of SPD matrices", np.mean(percent))
        print("Mean negative eigenvalue", np.mean(eigenvalues_neg))
        print("Mean positive eigenvalue", np.mean(eigenval_pos))
