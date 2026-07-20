import torch
import pymanopt
import numpy as np
from src.data.dataset_config import DATASET_CONFIG
from pyriemann.classification import MDM, TSClassifier
from pyriemann.utils.mean import mean_riemann
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from moabb.paradigms import FilterBankMotorImagery
from moabb.datasets import BNCI2014_001, Dreyer2023C, Beetl2021_A
from sklearn.model_selection import train_test_split

from src.Visualization.topomap import plot_topomap, plot_pannel
from src.Permutation.FeaturePermutationSPD import (proj_on_spd,percent_spd, spd_values, eigenval_pos)
from pyriemann.tangentspace import TangentSpace
from functools import partial
import math
import matplotlib.pyplot as plt

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
        default=10,
        help="Nombre de splits (défaut: 10)"
    )

    
    parser.add_argument(
        "--eigenval_info",
        type=bool,
        default=False,
        help="Si True, le programme affichera des informations sur les valeurs propres des matrices avant projection."
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    cfg = DATASET_CONFIG[args.dataset]
    dataset = cfg["dataset"]
    SENSORS = cfg["sensors"]

    OUT_DIR = f'../../Results/Permutation/{args.clf}/{args.dataset}/With_cue/Modif_covs/Heuristic'

    N_SPLITS = args.n_splits
    SCORE_THRESHOLD = 0.75

    CLASSIFIERS = {
    "MDM": MDM,
    "TSClassifier": TSClassifier,
}

    clf = CLASSIFIERS[args.clf]()

    visu_only = True


    if not visu_only:
        paradigm = FilterBankMotorImagery(filters=[(7, 35)], events ={"left_hand": 1, "right_hand": 2})
        X, y, meta = paradigm.get_data(dataset=dataset)
        scores_per_subject = []
        importances_per_subject = []

        for subject in dataset.subject_list:
            subject_mask = meta.subject == subject
            data = X[subject_mask]
            y_subj = y[subject_mask]

            scores_per_split = []
            importances_per_split = []

            for i in range(N_SPLITS):
                X_train, X_test, y_train, y_test = train_test_split(
                    data, y_subj, stratify = y_subj, test_size=0.2, random_state=i
                )

                rng = np.random.default_rng(i)
                covs_transformer = Covariances()
                C_train = covs_transformer.fit_transform(X_train) 
                C_test = covs_transformer.transform(X_test)
                
                clf.fit(C_train, y_train)
                score_initial = clf.score(C_test, y_test)

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
                    score = clf.score(np.array(perm_C_test), y_test)
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



