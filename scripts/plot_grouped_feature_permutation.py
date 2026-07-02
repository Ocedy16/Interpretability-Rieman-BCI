from src.Permutation.GroupedFeaturePermutation import GroupedFeaturePermutation
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
        "--n_perms",
        type=int,
        default=10,
        help="Nombre de permutations (défaut: 10)"
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    cfg = DATASET_CONFIG[args.dataset]
    dataset = cfg["dataset"]
    SENSORS = cfg["sensors"]

    OUT_DIR = f'Results/Permutation/{args.clf}/{args.dataset}/With_cue'

    N_SPLITS = 10
    N_PERMS = 10
    SCORE_THRESHOLD = 0.75

    classifier = args.clf

    paradigm = FilterBankMotorImagery(filters=[[7, 35]],events={"left_hand": 1, "right_hand": 2})

    pipeline = Pipeline([
        ("cov", Covariances()),
        ("clf", classifier())
    ])

    perm = GroupedFeaturePermutation(dataset,paradigm, SENSORS, classifier)
    perm.fit(n_iter = N_SPLITS, n_perm = N_PERMS)

    results_dic = perm.results_dic
    filename = os.path.join(OUT_DIR, 'results_dic.pkl')
    with open(filename,'wb') as f:
        pickle.dump(results_dic,f)


    all_results = []
    all_scores = []
    good_subjects_imp = []
    good_subjects_scores = []
    bad_subjects_imp = []
    bad_subjects_scores = []

    for subject in dataset.subject_list:    
        results = perm.results_dic['across_times'][subject]['importance']
        scores = perm.results_dic['across_times'][subject]['accuracy']
        all_results.append(np.mean(np.array(results),axis=0))
        all_scores.append(np.mean(np.array(scores),axis=0))

        if np.mean(np.array(scores),axis=0) >= SCORE_THRESHOLD : 
            good_subjects_imp.append(np.mean(np.array(results),axis=0))
            good_subjects_scores.append(np.mean(np.array(scores),axis=0))

        else: 
            bad_subjects_imp.append(np.mean(np.array(results),axis=0))
            bad_subjects_scores.append(np.mean(np.array(scores),axis=0))
        
    np.save(f'{OUT_DIR}/all_scores.npy', np.array(all_scores))  
    np.save(f'{OUT_DIR}/good_subjects.npy', np.array(good_subjects_imp))
    np.save(f'{OUT_DIR}/bad_subjects.npy', np.array(bad_subjects_imp))
    np.save(f'{OUT_DIR}/good_subjects_scores.npy', np.array(good_subjects_scores))
    np.save(f'{OUT_DIR}/bad_subjects_scores.npy', np.array(bad_subjects_scores))

    plot_pannel(all_results, SENSORS, all_scores,cbar_type='Permutation')


    #good_subjects_imp = np.load(f'{OUT_DIR}/good_subjects.npy')
    #bad_subjects_imp = np.load(f'{OUT_DIR}/bad_subjects.npy')
    #good_subjects_score = np.load(f'{OUT_DIR}/good_subjects_scores.npy')
    #bad_subjects_score = np.load(f'{OUT_DIR}/bad_subjects_scores.npy')


    v_max = np.max(np.abs(good_subjects_imp))
    v_min = -v_max

    plot_topomap(np.mean(np.array(good_subjects_imp),axis=0),SENSORS,
                title = f"Subjects with a classif score $\\geq$ {SCORE_THRESHOLD}\n Mean Score : {np.mean(np.array(good_subjects_scores)):.2f}",
                vlim=(v_min,v_max),
                savefile_name=f'{OUT_DIR}/Good_subjects',
                cbar_type='Permutation')


    plot_topomap(np.mean(np.array(bad_subjects_imp),axis=0),SENSORS,
                title = f"Subjects with a classif score $< {SCORE_THRESHOLD}$\n Mean Score : {np.mean(np.array(bad_subjects_scores)):.2f}",
                vlim=(v_min,v_max),
                savefile_name=f'{OUT_DIR}/Bad_subjects',
                cbar_type='Permutation')
