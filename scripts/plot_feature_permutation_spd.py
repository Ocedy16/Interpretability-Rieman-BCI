import torch
import pymanopt
import numpy as np
from pyriemann.classification import MDM, TSClassifier
from pyriemann.utils.mean import mean_riemann
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from moabb.paradigms import FilterBankMotorImagery
from moabb.datasets import BNCI2014_001, Dreyer2023C, Beetl2021_A
from sklearn.model_selection import train_test_split

from src.Visualization.topomap import plot_topomap, plot_pannel
from src.Permutation.FeaturePermutationSPD import permute_sensor_variances, permute_sensor_covariances, proj_on_spd, permute_sensor_across_trials
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
        "--n_perms",
        type=int,
        default=10,
        help="Nombre de permutations (défaut: 10)"
    )

    parser.add_argument(
        "--method",
        type=str,
        default="covs",
        help="Permutation computing method : vars : only variances perm, covs : only covariances perm, trials : permutation vars and covs across trials, tangent : permutation on the tangent space"
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

    OUT_DIR = f'Results/Permutation/{args.clf}/{args.dataset}/With_cue/Modif_covs'

    N_SPLITS = args.n_splits
    N_PERMS = args.n_perms
    SCORE_THRESHOLD = 0.75

    classifier = args.clf

    program = args.method


    if program == "covs":
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
                        data, y_subj, test_size=0.2, random_state=i
                    )

                    rng = np.random.default_rng(i)
                    covs_transformer = Covariances()
                    C_train = covs_transformer.fit_transform(X_train) 
                    C_test = covs_transformer.transform(X_test)
                    
                    clf.fit(C_train, y_train)
                    score_initial = clf.score(C_test, y_test)

                    mean_importances = []

                    for k in range(N_PERMS):
                        perm_seed = rng.integers(0, 1_000_000)
                        perm_rng = np.random.default_rng(perm_seed)

                        importances = []
                        nb_feat = len(C_train[0])

                        for feature in range(nb_feat):
                            perm_C_test = []
                            for matrix in C_test: 
                                matrix_mod = permute_sensor_covariances(matrix, feature, perm_rng)
                                matrix_mod = proj_on_spd(matrix_mod)
                                perm_C_test.append(matrix_mod)
                            score = clf.score(np.array(perm_C_test), y_test)
                            importances.append(score_initial - score)

                        mean_importances.append(importances)

                    mean_importances = np.mean(np.array(mean_importances), axis=0)
                    scores_per_split.append(score_initial)
                    importances_per_split.append(mean_importances)

                scores_per_subject.append(scores_per_split)
                importances_per_subject.append(importances_per_split)

            np.save(f'{OUT_DIR}/feature_perm_importances_covs_matrix.npy', importances_per_subject)
            np.save(f'{OUT_DIR}/feature_perm_scores_covs_matrix.npy', scores_per_subject)
            np.save(f'{OUT_DIR}/percent_spd.npy',np.array(percent_spd))
            np.save(f'{OUT_DIR}/spd_values.npy',np.array(spd_values))
            np.save(f'{OUT_DIR}/eigenval_pos.npy',np.array(eigenval_pos))


        scores = np.load(f'{OUT_DIR}/feature_perm_scores_covs_matrix.npy')
        importances = np.load(f'{OUT_DIR}/feature_perm_importances_covs_matrix.npy')

        scores_reduced = np.mean(scores, axis=1)
        importances_reduced = np.mean(importances, axis=1)


        plot_pannel_topomap(importances_reduced, sensors, scores_reduced)


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

        visualize_channel_importance(np.mean(np.array(good_subjects_perm),axis=0),sensors,
                    title = f"Subjects with a classif score $\\geq$ {SCORE_THRESHOLD}\n Mean Score : {np.mean(np.array(good_subjects_scores)):.2f}",
                    vlim=(v_min,v_max),
                    savefile_name=f'{OUT_DIR}/Good_subjects.pdf',
                    cbar_type = "Permutation")


        visualize_channel_importance(np.mean(np.array(bad_subjects_perm),axis=0),sensors,
                    title = f"Subjects with a classif score $< {SCORE_THRESHOLD}$\n Mean Score : {np.mean(np.array(bad_subjects_scores)):.2f}",vlim=(v_min,v_max),
                    savefile_name=f'{OUT_DIR}/Bad_subjects.pdf',
                    cbar_type = "Permutation")



    if program == "vars" : 

        if not visu_only:
            paradigm = FilterBankMotorImagery(filters=[(7, 35)],events={"left_hand": 1, "right_hand": 2})
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
                        data, y_subj, test_size=0.2, random_state=i
                    )

                    rng = np.random.default_rng(i)
                    covs_transformer = Covariances()
                    C_train = covs_transformer.fit_transform(X_train) 
                    C_test = covs_transformer.transform(X_test)
                    
                    clf.fit(C_train, y_train)
                    score_initial = clf.score(C_test, y_test)

                    mean_importances = []

                    for k in range(N_PERMS):
                        perm_seed = rng.integers(0, 1_000_000)
                        perm_rng = np.random.default_rng(perm_seed)

                        importances = []
                        nb_feat = len(C_train[0])

                        for feature in range(nb_feat):
                            perm_C_test = []
                            for matrix in C_test: 
                                matrix_mod = permute_sensor_variances(matrix, feature, perm_rng)
                                matrix_mod = proj_on_spd(matrix_mod)
                                perm_C_test.append(matrix_mod)
                            score = clf.score(np.array(perm_C_test), y_test)
                            importances.append(score_initial - score)

                        mean_importances.append(importances)

                    mean_importances = np.mean(np.array(mean_importances), axis=0)
                    scores_per_split.append(score_initial)
                    importances_per_split.append(mean_importances)

                scores_per_subject.append(scores_per_split)
                importances_per_subject.append(importances_per_split)

            np.save(f'{OUT_DIR}/feature_perm_importances_var_matrix.npy', importances_per_subject)
            np.save(f'{OUT_DIR}/feature_perm_scores_var_matrix.npy', scores_per_subject)


        scores = np.load(f'{OUT_DIR}/feature_perm_scores_var_matrix.npy')
        importances = np.load(f'{OUT_DIR}/feature_perm_importances_var_matrix.npy')

        scores_reduced = np.mean(scores, axis=1)
        importances_reduced = np.mean(importances, axis=1)

        print(scores_reduced.shape)
        print(importances_reduced.shape)

        plot_pannel_topomap(importances_reduced, sensors, scores_reduced)


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

        visualize_channel_importance(np.mean(np.array(good_subjects_perm),axis=0),sensors,
                    title = f"Subjects with a classif score $\\geq$ {SCORE_THRESHOLD}\n Mean Score : {np.mean(np.array(good_subjects_scores)):.2f}",vlim=(v_min,v_max),savefile_name=f'{OUT_DIR}/Good_subjects.pdf')


        visualize_channel_importance(np.mean(np.array(bad_subjects_perm),axis=0),sensors,
                    title = f"Subjects with a classif score $< {SCORE_THRESHOLD}$\n Mean Score : {np.mean(np.array(bad_subjects_scores)):.2f}",vlim=(v_min,v_max),savefile_name=f'{OUT_DIR}/Bad_subjects.pdf')

    if program == 'combining':

        good_subjects_perm_covs = np.load(f'{OUT_DIR}/Modif_covs/good_subjects.npy')
        bad_subjects_perm_covs = np.load(f'{OUT_DIR}/Modif_covs/bad_subjects.npy')
        good_subjects_perm_var = np.load(f'{OUT_DIR}/good_subjects.npy')
        bad_subjects_perm_var = np.load(f'{OUT_DIR}/bad_subjects.npy')

        good_subjects_scores = np.load(f'{OUT_DIR}/good_subjects_scores.npy')
        bad_subjects_scores = np.load(f'{OUT_DIR}/bad_subjects_scores.npy')

        scores = np.load(f'{OUT_DIR}/feature_perm_scores_var_matrix.npy')
        importances = np.load(f'{OUT_DIR}/feature_perm_importances_var_matrix.npy')

        scores_reduced = np.mean(scores, axis=1)
        importances_reduced = np.mean(importances, axis=1)

        scores_covs = np.load(f'{OUT_DIR}/Modif_covs/feature_perm_scores_covs_matrix.npy')
        importances_covs = np.load(f'{OUT_DIR}/Modif_covs/feature_perm_importances_covs_matrix.npy')


        scores_covs_reduced = np.mean(scores_covs,axis=1)
        importances_covs_reduced = np.mean(importances_covs,axis=1)

        combined = []

        for i in range(len(scores_covs_reduced)):
            combined.append(0.3*importances_reduced[i]+0.5*importances_covs_reduced[i])

        print(combined)

        combined = np.array(combined)
        plot_pannel_topomap(combined, sensors, scores_reduced)

        combined_good_subjects = 0.5*np.mean(np.array(good_subjects_perm_covs),axis=0)+ 0.5*np.mean(np.array(good_subjects_perm_var),axis=0)
        combined_bad_subjects = 0.5*np.mean(np.array(bad_subjects_perm_covs),axis=0)+ 0.5*np.mean(np.array(bad_subjects_perm_var),axis=0)

        v_max = np.max(np.abs(combined_good_subjects))
        v_min = -v_max

        visualize_channel_importance(combined_good_subjects,sensors,
                    title = f"Subjects with a classif score $\\geq$ {SCORE_THRESHOLD}\n Mean Score : {np.mean(np.array(good_subjects_scores)):.2f}",vlim=(v_min,v_max),savefile_name=f'{OUT_DIR}/Good_subjects.pdf')


        visualize_channel_importance(combined_bad_subjects,sensors,
                    title = f"Subjects with a classif score $< {SCORE_THRESHOLD}$\n Mean Score : {np.mean(np.array(bad_subjects_scores)):.2f}",vlim=(v_min,v_max),savefile_name=f'{OUT_DIR}/Bad_subjects.pdf')


    if program == 'trials':
        if not visu_only:
            paradigm = FilterBankMotorImagery(filters=[(7, 35)],events={"left_hand": 1, "right_hand": 2})
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
                        data, y_subj, test_size=0.2, random_state=i
                    )

                    rng = np.random.default_rng(i)
                    covs_transformer = Covariances()
                    C_train = covs_transformer.fit_transform(X_train) 
                    C_test = covs_transformer.transform(X_test)
                    
                    clf.fit(C_train, y_train)
                    score_initial = clf.score(C_test, y_test)

                    mean_importances = []

                    for k in range(N_PERMS):
                        perm_seed = rng.integers(0, 1_000_000)
                        perm_rng = np.random.default_rng(perm_seed)

                        importances = []
                        nb_feat = len(C_train[0])

                        for feature in range(nb_feat):
                            perm_C_test = permute_sensor_across_trials(C_test,feature,rng) 
                            perm_C_test = [proj_on_spd(matrix) for matrix in perm_C_test]
                            score = clf.score(np.array(perm_C_test), y_test)
                            importances.append(score_initial - score)

                        mean_importances.append(importances)

                    mean_importances = np.mean(np.array(mean_importances), axis=0)
                    scores_per_split.append(score_initial)
                    importances_per_split.append(mean_importances)

                scores_per_subject.append(scores_per_split)
                importances_per_subject.append(importances_per_split)

            np.save(f'{OUT_DIR}/feature_perm_importances_covs_matrix.npy', importances_per_subject)
            np.save(f'{OUT_DIR}/feature_perm_scores_covs_matrix.npy', scores_per_subject)


        scores = np.load(f'{OUT_DIR}/feature_perm_scores_covs_matrix.npy')
        importances = np.load(f'{OUT_DIR}/feature_perm_importances_covs_matrix.npy')

        scores_reduced = np.mean(scores, axis=1)
        importances_reduced = np.mean(importances, axis=1)


        plot_pannel_topomap(importances_reduced, sensors, scores_reduced)


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

        visualize_channel_importance(np.mean(np.array(good_subjects_perm),axis=0),sensors,
                    title = f"Subjects with a classif score $\\geq$ {SCORE_THRESHOLD}\n Mean Score : {np.mean(np.array(good_subjects_scores)):.2f}",vlim=(v_min,v_max),savefile_name=f'{OUT_DIR}/Good_subjects.pdf')


        visualize_channel_importance(np.mean(np.array(bad_subjects_perm),axis=0),sensors,
                    title = f"Subjects with a classif score $< {SCORE_THRESHOLD}$\n Mean Score : {np.mean(np.array(bad_subjects_scores)):.2f}",vlim=(v_min,v_max),savefile_name=f'{OUT_DIR}/Bad_subjects.pdf')

    if program == 'tangent':
        if not visu_only:
            paradigm = FilterBankMotorImagery(filters=[(7, 35)],events={"left_hand": 1, "right_hand": 2})
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
                        data, y_subj, test_size=0.2, random_state=i
                    )

                    rng = np.random.default_rng(i)
                    covs_transformer = Covariances()
                    C_train = covs_transformer.fit_transform(X_train) 
                    C_test = covs_transformer.transform(X_test)
                    
                    clf.fit(C_train, y_train)
                    score_initial = clf.score(C_test, y_test)

                    mean_importances = []

                    for k in range(N_PERMS):
                        perm_seed = rng.integers(0, 1_000_000)
                        perm_rng = np.random.default_rng(perm_seed)

                        importances = []
                        nb_feat = len(C_train[0])
                        new_C_test = []
                        for matrix in C_test:
                            eigenvals, eigenvecs = np.linalg.eigh(matrix)
                            log_eigenvals = np.log(eigenvals)
                            new_C_test.append(eigenvecs @ np.diag(log_eigenvals) @ eigenvecs.T)
                        new_C_test = np.array(new_C_test)
                        for feature in range(nb_feat):
                            perm_C_test = []
                            for matrix in new_C_test: 
                                matrix_mod = permute_sensor_covariances(matrix, feature, perm_rng)
                                eigenvals, eigenvecs = np.linalg.eigh(matrix_mod)
                                exp_eigenvals = np.exp(eigenvals)
                                matrix_mod = eigenvecs @ np.diag(exp_eigenvals) @ eigenvecs.T
                                perm_C_test.append(matrix_mod)
                            score = clf.score(np.array(perm_C_test), y_test)
                            importances.append(score_initial - score)

                        mean_importances.append(importances)

                    mean_importances = np.mean(np.array(mean_importances), axis=0)
                    scores_per_split.append(score_initial)
                    importances_per_split.append(mean_importances)

                scores_per_subject.append(scores_per_split)
                importances_per_subject.append(importances_per_split)

            np.save(f'{OUT_DIR}/feature_perm_importances_covs_matrix.npy', importances_per_subject)
            np.save(f'{OUT_DIR}/feature_perm_scores_covs_matrix.npy', scores_per_subject)


        scores = np.load(f'{OUT_DIR}/feature_perm_scores_covs_matrix.npy')
        importances = np.load(f'{OUT_DIR}/feature_perm_importances_covs_matrix.npy')

        scores_reduced = np.mean(scores, axis=1)
        importances_reduced = np.mean(importances, axis=1)


        plot_pannel_topomap(importances_reduced, sensors, scores_reduced)


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

        visualize_channel_importance(np.mean(np.array(good_subjects_perm),axis=0),sensors,
                    title = f"Subjects with a classif score $\\geq$ {SCORE_THRESHOLD}\n Mean Score : {np.mean(np.array(good_subjects_scores)):.2f}",vlim=(v_min,v_max),savefile_name=f'{OUT_DIR}/Good_subjects.pdf')


        visualize_channel_importance(np.mean(np.array(bad_subjects_perm),axis=0),sensors,
                    title = f"Subjects with a classif score $< {SCORE_THRESHOLD}$\n Mean Score : {np.mean(np.array(bad_subjects_scores)):.2f}",vlim=(v_min,v_max),savefile_name=f'{OUT_DIR}/Bad_subjects.pdf')



    if args.eigenval_info:
        percent = np.load(f'{OUT_DIR}/percent_spd.npy')
        eigenvalues_neg = np.load(f'{OUT_DIR}/spd_values.npy')
        eigenval_pos = np.load(f'{OUT_DIR}/eigenval_pos.npy')
        print("Percentage of SPD matrices" : np.mean(percent))
        print("Mean negative eigenvalue", np.mean(eigenvalues_neg))
        print("Mean positive eigenvalue", np.mean(eigenval_pos))