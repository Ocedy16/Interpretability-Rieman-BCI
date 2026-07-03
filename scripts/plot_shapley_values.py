from src.Shapley.shapley_eeg import compute_shapley
from src.data.dataset_config import DATASET_CONFIG
from src.Visualization.topomap import plot_pannel, plot_topomap
import numpy as np 
import moabb
from sklearn.model_selection import train_test_split
from moabb.datasets import BNCI2014_001, Dreyer2023C, Beetl2021_A
from moabb.paradigms import FilterBankMotorImagery

import mne
import math
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from pyriemann.classification import MDM, TSClassifier
from pyriemann.estimation import Covariances
from sklearn.model_selection import train_test_split

import os
import pickle

import argparse

np.random.seed(3)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        choices=list(DATASET_CONFIG.keys()),
        required=True,
        help="Dataset à utiliser : BNCI2014_001, Dreyer2023C, Beetl2021_A ou pollution"
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
    paradigm = FilterBankMotorImagery(filters=[[7, 35]],events={"left_hand": 1, "right_hand": 2})
    SENSORS = cfg["sensors"]

    if args.savefile is None:

        OUT_DIR = f'Results/Shapley/{args.clf}/{args.dataset}'

    else :

        OUT_DIR = args.savefile

    N_SPLITS = args.n_splits
    SCORE_THRESHOLD = 0.75

    CLASSIFIERS = {
    "MDM": MDM,
    "TSClassifier": TSClassifier,
}

    classifier = CLASSIFIERS[args.clf]()

    pipeline = Pipeline([
        ('cov', Covariances(),),
        ('clf', classifier)]
    )
    visu_only = False

    if not visu_only:
        all_subjects_shap_values = []
        all_mean_shap_values = []
        all_scores = []
        good_subjects_shap = []
        good_subjects_scores = []
        bad_subjects_shap = []
        bad_subjects_scores = []

        for subject in dataset.subject_list :  
            print("Subject",subject)  
            X,y,meta = paradigm.get_data(dataset=dataset, subjects=[subject])
            print(y)
            shap_values, scores = compute_shapley(X,y,N_SPLITS, pipeline)

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


        plot_pannel(all_mean_shap_values, dataset, SENSORS, all_scores, OUT_DIR)

    good_subjects_shap = np.load(f"{OUT_DIR}/good_subjects.npy",)
    bad_subjects_shap = np.load(f"{OUT_DIR}/bad_subjects.npy")
    good_subjects_scores = np.load(f"{OUT_DIR}/good_subjects_scores.npy")
    bad_subjects_scores = np.load(f"{OUT_DIR}/bad_subjects_scores.npy")

    v_max = np.max(np.abs(np.mean(np.array(good_subjects_shap), axis=0)))
    v_min = -v_max

    plot_topomap(np.mean(np.array(good_subjects_shap),axis=0),SENSORS,
                title = f"Subjects with a classif score $\\geq$ {SCORE_THRESHOLD}\n Mean Score : {np.mean(np.array(good_subjects_scores)):.2f}",
                vlim=(v_min,v_max),
                savefile_name=f"{OUT_DIR}/Good_subjects.pdf")


    plot_topomap(np.mean(np.array(bad_subjects_shap),axis=0),SENSORS,
                title = f"Subjects with a classif score $< {SCORE_THRESHOLD}$\n Mean Score : {np.mean(np.array(bad_subjects_scores)):.2f}",
                vlim=(v_min,v_max),
                savefile_name=f"{OUT_DIR}/Bad_subjects.pdf")
