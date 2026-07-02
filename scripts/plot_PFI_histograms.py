import sys, os
import numpy as np


from pyriemann.classification import MDM
from pyriemann.tangentspace import TangentSpace
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.naive_bayes import GaussianNB

from pymanopt.optimizers import ConjugateGradient

import pandas as pd
import time
from tqdm import tqdm
from pyriemann.utils.mean import mean_riemann

from src.Shapley
from utils.FeaturePermutationNotMoabb import FeaturePermutation
from utils.Visualization import visualize_channel_importance
import numpy as np 
import moabb
from sklearn.model_selection import train_test_split
from moabb.datasets import BNCI2014_001, Dreyer2023C, Beetl2021_A
from moabb.paradigms import FilterBankMotorImagery

import mne
import math
import matplotlib.pyplot as plt
from mne.viz.topomap import plot_topomap

from sklearn.pipeline import Pipeline
from pyriemann.classification import MDM, TSClassifier
from pyriemann.estimation import Covariances

import os
import pickle
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        choices=list(DATASET_CONFIG.keys())+["pollution"],
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
        "--subject",
        type=int,
        help="Classifieur à utiliser : MDM, TSClassifier"
    )
    parser.add_argument(
        "--method",
        type=str,
        help="Méthode utilisée : Shapley, Permutation ou GroupedFeaturePermutation"
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

    if dataset == 'pollution' : 

        OUT_DIR = f"Results/{args.method}/{args.clf}/Air_pollution"

        with open(f'{OUT_DIR}/results_dic.pkl', 'rb') as f:
            results = pickle.load(f)

        scores = results['across_times']['accuracy']
        importance = np.array(results['across_times']['importance'])

        mean_importance = np.mean(importance,axis=0)
        components = ['CO', 'NO2', 'O3', 'PM10', "PM2.5", "SO2"]

        plt.bar(components, mean_importance)
        plt.title('Feature Permutation Importance of components')
        plt.xlabel('Components')
        plt.ylabel('Feature Permutation Importance')
        plt.savefig(f'{OUT_DIR}/plot_{args.clf}.pdf')
        plt.show()

    elif method == 'Permutation':
        OUT_DIR = f"Results/Permutation/{args.clf}/{args.dataset}"

        with open(f'{OUT_DIR}/results_dic.pkl', 'rb') as f:
            perm_values = np.array(pickle.load(f))
            perm_values = perm_values[:,0,:,0,:]

        subject_mean_perm_values = np.mean(perm_values[subject],axis=0)
        indices_tries = np.argsort(subject_mean_perm_values)[::-1]
        sensors_tries = np.array(sensors_bnci)[indices_tries]
        perm_values_triees = subject_mean_perm_values[indices_tries]

        plt.figure(figsize=(12, 6))
        plt.bar(sensors_tries, perm_values_triees)
        plt.title(f'Importance values for the sensors of dataset {args.dataset}')
        plt.xlabel('Sensors')
        plt.ylabel('Importance')
        plt.savefig(f'{OUT_DIR}/histogram.pdf')
        plt.show()

    elif method =='Shapley' : 

        OUT_DIR = f"Results/Shapley/{args.clf}/{args.dataset}"

        with open(f'{OUT_DIR}/all_shap_values.pkl', 'rb') as f:
            shap_values = np.array(pickle.load(f))
            shap_values = shap_values[:,0,:,0,:]

        subject_mean_shap_values = np.mean(shap_values[subject],axis=0)
        indices_tries = np.argsort(subject_mean_shap_values)[::-1]
        sensors_tries = np.array(sensors_bnci)[indices_tries]
        shap_values_triees = subject_mean_shap_values[indices_tries]

        plt.figure(figsize=(12, 6))
        plt.bar(sensors_tries, shap_values_triees)
        plt.title(f'Shapley values of {args.dataset} for the class left_hand')
        plt.xlabel('Sensors')
        plt.ylabel('Shapley values')
        plt.savefig(f'{OUT_DIR}/histogram.pdf')
        plt.show()


    
