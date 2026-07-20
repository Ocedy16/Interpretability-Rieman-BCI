import numpy as np
from src.data.dataset_config import DATASET_CONFIG
from moabb.datasets import BNCI2014_001, Dreyer2023C, Beetl2021_A
from moabb.paradigms import FilterBankMotorImagery
from sklearn.model_selection import train_test_split
from pyriemann.classification import TSClassifier, MDM
from pyriemann.estimation import Covariances
from sklearn.pipeline import Pipeline
from src.Visualization.beeswarm import shap_beeswarm, shap_beeswarm_grouped
import os

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

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    cfg = DATASET_CONFIG[args.dataset]
    dataset = cfg["dataset"]
    SENSORS = cfg["sensors"]

    OUT_DIR = f"../../Results/Shapley/{args.clf}/{args.dataset}/Shapley_SPD/Heuristic"

    SCORE_THRESHOLD = 0.75


    paradigm = FilterBankMotorImagery(filters=[[7, 35]], events={"left_hand": 1, "right_hand": 2})

    os.makedirs(OUT_DIR, exist_ok=True) 

    order = [
        "Fp1","Fp2",
        "AF7","AF3","AFz","AF4","AF8",
        "F7","F3","Fz","F4","F8",
        "FC5","FC3","FCz","FC4","FC6",
        "C5","C3","Cz","C4","C6",
        "CP5","CP3","CPz","CP4","CP6",
        "P7","P3","Pz","P4","P8",
        "O1","Oz","O2"
    ]


    shap_values = np.load(f'{OUT_DIR}/all_shap_values.pkl', allow_pickle=True)

    
    shap_per_subject_list = []

    for i in range (len(shap_values)):
        shap_per_subject = np.array(shap_values[i])
        shap_per_subject = np.squeeze(shap_per_subject)
        shap_per_subject_list.append(shap_per_subject)

    print(np.array(shap_per_subject_list).shape)

    regions = {
        "Frontal": ["Fp1","Fp2","AF7","AF3","AFz","AF4","AF8","F7","F5","F3","F1","Fz","F2","F4","F6","F8"],
        "Frontocentral": ["FC5","FC3","FC1","FCz","FC2","FC4","FC6"],
        "Central": ["C5","C3","C1","Cz","C2","C4","C6"],
        "Temporal": ["FT9","FT7","FT8","FT10","T7","T8"],
        "Parietal": ["CP5","CP3","CP1","CPz","CP2","CP4","CP6","P7","P5","P3","P1","Pz","P2","P4","P6","P8"],
        "Occipital": ["O1","Oz","O2","PO7","PO3","POz","PO4","PO8"],
        "TP": ["TP7","TP8","TP9","TP10"]
    }


    left = ["Fp1","AF7","AF3","F7","F5","F3","F1",
    "FC5","FC3","FC1",
    "C5","C3","C1",
    "CP5","CP3","CP1",
    "P7","P5","P3","P1",
    "PO7","PO3"]

    right = ["Fp2","AF8","AF4","F8","F6","F4","F2", "FC6","FC4","FC2",
    "C6","C4","C2",
    "CP6","CP4","CP2",
    "P8","P6","P4","P2",
    "PO8","PO4"]

    mid = ["Fz","FCz","Cz","CPz","Pz","Oz","AFz","POz"]

    temp = ["FT7","T7","TP7","FT9",
    "FT8","T8","TP8","FT10"]

    channel_to_hemisphere = {}
    channel_to_region = {}

    for ch in left:
        channel_to_hemisphere[ch] = "Left"

    for ch in right:
        channel_to_hemisphere[ch] = "Right"

    for ch in mid:
        channel_to_hemisphere[ch] = "Midline"

    for ch in temp:
        channel_to_hemisphere[ch] = "Temporal"

    for region, chans in regions.items():
        for ch in chans:
            channel_to_region[ch] = region


    all_channels_ordered = list(dict.fromkeys(
        regions["Frontal"] +
        regions["Frontocentral"] +
        regions["Central"] +
        regions["Temporal"] +
        regions["Parietal"] +
        regions["Occipital"]
    ))


    for i, subject in enumerate(dataset.subject_list):
        X,y,meta = paradigm.get_data(dataset=dataset, subjects=[subject])

        n_splits = 1
        X_diff_power_mean = []

        for j in range (n_splits):
            X_train,X_test,y_train,y_test = train_test_split(X,y, train_size=0.8, stratify=y, random_state=3)

            baseline = np.mean(X_train,axis=0)
            power_baseline = np.mean(baseline**2,axis=1)
            print(power_baseline)

            power_per_matrix = [
        np.mean(X**2, axis=1)
        for X in X_test
    ]

            power_per_matrix = np.vstack(power_per_matrix)
            print(power_per_matrix[0])

            power_diff = np.log(
                power_per_matrix
                / (power_baseline[None,:] + 1e-12)
            )
            X_diff_power_mean.append(power_diff)

        shap_beeswarm_grouped(np.array(shap_per_subject_list[i]), SENSORS, args.dataset, subject, regions, channel_to_hemisphere, OUT_DIR)
