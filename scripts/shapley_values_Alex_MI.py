from utils.ShapleyValues import shapley_values
from src.Permutation.FeaturePermutation import FeaturePermutation
from src.Shapley.shapley_eeg import KernelShap
from src.Visualization.topomap import visualize_channel_importance
import numpy as np 
import moabb
from sklearn.model_selection import train_test_split
from moabb.datasets import BNCI2014_001, Dreyer2023C, Beetl2021_A, AlexMI
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

np.random.seed(3)


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
            ]),
    "AlexMI": dict(
        dataset=AlexMI(),
        session="0",
        good_subjects=np.array([83, 85, 87]),
        bad_subjects=np.array([82,84,86]),
        sensors =[
                "Fpz",
                "F7",
                "F3",
                "Fz",
                "F4",
                "F8",
                "T7",
                "C3",
                "Cz",
                "C4",
                "T8",
                "P7",
                "P3",
                "Pz",
                "P4",
                "P8",
            ]
    ),
}

N_SPLITS = 1
SCORE_THRESHOLD = 0.75
visu_only = True

classifier = MDM
paradigm = FilterBankMotorImagery(filters=[[7, 35]],n_classes=3)

pipeline = Pipeline([
    ("cov", Covariances()),
    ("clf", classifier())
])

cfg = DATASET_CONFIG["AlexMI"]
dataset = cfg["dataset"]
session = cfg["session"]
good_subjects = cfg["good_subjects"]
bad_subjects = cfg["bad_subjects"]
SENSORS = cfg["sensors"]
OUT_DIR = 'Results/Shapley/MDM/AlexMI/Mean_rest'


def compute_shapley(X,y, n_splits):
    all_shap_values = []
    all_scores = []
    for i in range (n_splits):
        print("Split",i)
        baseline = np.mean(X[np.where(y=='rest')],axis=0)
        X,y = X[np.where(y!="rest")], y[np.where(y!='rest')]
        print(np.unique(y))
        X_train,X_test,y_train,y_test = train_test_split(X,y, train_size=0.8, stratify=y, random_state=i)
        pipeline.fit(X_train,y_train)
        probas = pipeline.predict_proba(X_test)
        shap_values = KernelShap(X_train,X_test, pipeline,baseline=baseline, n_samples=1000)
        all_shap_values.append(shap_values)
        all_scores.append(pipeline.score(X_test,y_test))

    return all_shap_values, all_scores

    
def plot_pannel_shapley(all_shap_values,sensors,all_scores):

    n_subjects = len(all_shap_values)
    n_cols = 4
    n_rows = math.ceil(n_subjects / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))
    axes = axes.flatten()
    v_max = np.max(np.abs(all_shap_values))
    v_min = -v_max

    for i in range(n_subjects):
        shap_data = all_shap_values[i]
        score = all_scores[i]
        
        visualize_channel_importance(
            importance_values=shap_data, 
            sensors=sensors,
            title=f"Subject {dataset.subject_list[i]}\nMean Score {score:.2f} ",
            ax=axes[i],
            show_cbar=True,
            vlim=(v_min,v_max)
        )

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    fig.subplots_adjust(hspace=0.4, wspace=0.3)
    plt.savefig(f"{OUT_DIR}/Panel_Tous_Sujets.pdf")
    plt.show()

visu_only = True

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
        shap_values, scores = compute_shapley(X,y,N_SPLITS)

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


    plot_pannel_shapley(all_mean_shap_values,SENSORS, all_scores)

good_subjects_shap = np.load(f"{OUT_DIR}/good_subjects.npy",)
bad_subjects_shap = np.load(f"{OUT_DIR}/bad_subjects.npy")
good_subjects_scores = np.load(f"{OUT_DIR}/good_subjects_scores.npy")
bad_subjects_scores = np.load(f"{OUT_DIR}/bad_subjects_scores.npy")

v_max = np.max(np.abs(np.mean(np.array(good_subjects_shap), axis=0)))
v_min = -v_max

visualize_channel_importance(np.mean(np.array(good_subjects_shap),axis=0),SENSORS,
               title = f"Subjects with a classif score $\\geq$ {SCORE_THRESHOLD}\n Mean Score : {np.mean(np.array(good_subjects_scores)):.2f}",vlim=(v_min,v_max),savefile_name=f"{OUT_DIR}/Good_subjects.pdf")


visualize_channel_importance(np.mean(np.array(bad_subjects_shap),axis=0),SENSORS,
               title = f"Subjects with a classif score $< {SCORE_THRESHOLD}$\n Mean Score : {np.mean(np.array(bad_subjects_scores)):.2f}",vlim=(v_min,v_max),savefile_name=f"{OUT_DIR}/Bad_subjects.pdf")
