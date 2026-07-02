import math
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from moabb.datasets import BNCI2014_001, Dreyer2023C, Beetl2021_A
from moabb.paradigms import FilterBankMotorImagery
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from warnings import warn

from spd_learn.functional import covariance
from src.Shapley.shapley_eeg import compute_shapley
from src.Visualization.topomap import plot_topomap, plot_pannel
from src.SPDNet.SPDNet import SPDNetBatchNorm, train_model
np.random.seed(3)

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
        "--clf",
        type=str,
        choices=["MDM", "TSClassifier"],
        required=True,
        help="Classifieur à utiliser : MDM, TSClassifier"
    )
    parser.add_argument(
        "--n_splits",
        type=int,
        default = 1,
        help = "Nombre de splits. Warning, si le nombre de splits est supérieur à 1, des valeurs locales de différents traisl seront moyennés ensemble. Risque de perte d'interprétabilité locale."
    )


    return parser.parse_args()
       

if __name__ == "__main__":
    args = parse_args()
    cfg = DATASET_CONFIG[args.dataset]
    dataset = cfg["dataset"]
    SENSORS = cfg["sensors"]

    OUT_DIR = f"Results/Shapley/SPD_Net/{args.dataset}"

    N_SPLITS = args.n_splits
    SCORE_THRESHOLD = 0.75


    paradigm = FilterBankMotorImagery(filters=[[7, 35]], events={"left_hand": 1, "right_hand": 2})
    N_SPLITS = args.n_split

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

        X = torch.tensor(X, dtype=torch.float32).float()

        le = LabelEncoder()
        y_encoded = torch.tensor(le.fit_transform(y), dtype=torch.long)

        model_config = {
            "n_chans": X.shape[1],
            "n_outputs": 2,
            "subspacedim": 16,
            "bn": None,
        }

        shap_values, scores = compute_shapley(X, y_encoded, N_SPLITS, model_config, deep=True)

        # Average over splits, then over examples; drop the stray middle axis
        mean_split_shap = np.mean(np.array(shap_values), axis=0)  # (58, 1, 22, 2)
        mean_shap = np.mean(mean_split_shap, axis=0)               # (1, 22, 2)
        mean_shap = mean_shap[0, :, 0]  

        mean_score = float(np.mean(scores))
        all_subjects_shap_values.append(mean_split_shap)

        all_scores.append(mean_score)
        all_mean_shap_values.append(mean_shap)

        if mean_score >= SCORE_THRESHOLD:
            good_subjects_shap.append(mean_shap)
            good_subjects_scores.append(mean_score)
        else:
            bad_subjects_shap.append(mean_shap)
            bad_subjects_scores.append(mean_score)


    np.save(f'{OUT_DIR}/all_subjects_shap_values.npy', np.array(all_subjects_shap_values))
    np.save(f'{OUT_DIR}/all_mean_shap_values.npy', np.array(all_mean_shap_values))
    np.save(f'{OUT_DIR}/all_scores.npy', np.array(all_scores))
    # Save results
    np.save(f"{OUT_DIR}/good_subjects.npy", np.array(good_subjects_shap))
    np.save(f"{OUT_DIR}/bad_subjects.npy", np.array(bad_subjects_shap))
    np.save(f"{OUT_DIR}/good_subjects_scores.npy", np.array(good_subjects_scores))
    np.save(f"{OUT_DIR}/bad_subjects_scores.npy", np.array(bad_subjects_scores))


    plot_panel(
        all_subjects_shap_values,
        SENSORS,
        all_scores,
        subject_list=dataset.subject_list,
        save_path=f"{OUT_DIR}/Panel_Tous_Sujets.png",
    )

    v_max = np.max(np.abs(np.mean(np.array(good_subjects_shap), axis=0)))
    v_min = -v_max

    plot_topomap(
        np.mean(np.array(good_subjects_shap), axis=0),
        SENSORS,
        title=(
            f"Subjects with a classif score $\\geq$ {SCORE_THRESHOLD}\n"
            f"Mean Score : {np.mean(good_subjects_scores):.2f}"
        ),
        vlim=(v_min, v_max),
        savefile_name=f"{OUT_DIR}/Good_subjects",
    )

    plot_topomap(
        np.mean(np.array(bad_subjects_shap), axis=0),
        SENSORS,
        title=(
            f"Subjects with a classif score $< {SCORE_THRESHOLD}$\n"
            f"Mean Score : {np.mean(bad_subjects_scores):.2f}"
        ),
        vlim=(v_min, v_max),
        savefile_name=f"{OUT_DIR}/Bad_subjects",
)