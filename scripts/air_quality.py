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

from utils.ShapleyValues import shapley_values
from utils.FeaturePermutation import FeaturePermutation
from utils.KernelShapAirPollution import KernelShap
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

np.random.seed(3)


N_SPLITS = 10
SCORE_THRESHOLD = 0.75

classifier = TSClassifier

pipeline = Pipeline([
    ("clf", classifier())
])


OUT_DIR = 'Results/Shapley/MDM/Air_pollution'


def compute_shapley(Matrices, y, pipeline, n_splits=10):

    all_shap_values = []
    all_scores = []

    indices = np.arange(len(y))

    covs = np.stack([
    np.cov(X, rowvar=False)
    for X in Matrices
])

    for i in range(n_splits):

        print(f"Split {i}")

        train_idx, test_idx = train_test_split(
            indices,
            train_size=0.8,
            stratify=y,
            random_state=i
        )

        # Covariances pour le modèle
        X_train_cov = np.array(covs[train_idx])
        X_test_cov  = np.array(covs[test_idx])

        y_train = y[train_idx]
        y_test  = y[test_idx]

        # Signaux bruts pour SHAP
        X_train_raw = [Matrices[j] for j in train_idx]
        X_test_raw  = [Matrices[j] for j in test_idx]

        #print(X_train_cov)

        pipeline.fit(X_train_cov, y_train)

        score = pipeline.score(X_test_cov, y_test)

        shap_values = KernelShap(
            X_train_raw,
            X_test_raw,
            pipeline,
            n_samples=1000
        )

        all_shap_values.append(shap_values)
        all_scores.append(score)

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
    plt.savefig(f"{OUT_DIR}/Panel_Tous_Sujets.png")
    plt.show()


if __name__ == "__main__":

    try:
        df = pd.read_csv(
            "Data_pollution/Hourly_data_of_Beijing_from_Jinxi_20210305.csv",
            delimiter=",",
        )
        latlon = pd.read_csv("Data_pollution/latlondata.csv", delimiter=",")
        print("Successfully loaded the data.")
    except Exception as e:
        raise RuntimeError(
            "Failed to open the data, please download it from the link: https://drive.google.com/drive/folders/1qdldmpP-25UozI_wVTRg3D3JZf_4aaGQ?usp=share_link"
        )

    print(df.head())
    df1 = df.iloc[:, 11:17]
    df1 = df1.interpolate(method="linear", axis=0)  
    # Indices for unique values
    Sites = df.Site.unique()
    Dates = df.Date.unique()
    Months = df.Month.unique()
    holi = df.Holiday.unique()
    hours = df.Hour.unique()
    dow = df.DOW.unique()

    Matrices = []
    labels = []
    Covs = []
    coord = []
    numz = []
    for i in tqdm(Sites):
        for j in holi:
            Data = df1.loc[(df["Site"] == i) & (df["Holiday"] == j)]
            label = df.loc[(df["Site"] == i) & (df["Holiday"] == j)].iloc[:, 0]
            lat = latlon.loc[latlon["Site"] == i].iloc[:, 1]
            lon = latlon.loc[latlon["Site"] == i].iloc[:, 2]
            coord.append(np.hstack([lat, lon]))
            Matrices.append(Data.values)
            labels.append(str(j))

    for i in tqdm(Sites):
        for j in holi:
            if j == "holiday":
                numz.append(0)

            elif j == "weekday":
                numz.append(2)

            else:
                numz.append(1)

    from pyriemann.estimation import Covariances

    

    all_subjects_shap_values = []
    all_mean_shap_values = []
    all_scores = []
    good_subjects_shap = []
    good_subjects_scores = []
    bad_subjects_shap = []
    bad_subjects_scores = []

    shap_values, scores = compute_shapley(Matrices, np.array(labels), pipeline, N_SPLITS)

    #On moyenne sur tous les splits
    mean_split_shap_values = np.mean(np.array(shap_values),axis=0)

    #On retire l'axe inutile au milieu et on moyenne sur tous les exemples
    mean_shap_values = np.mean(mean_split_shap_values,axis=0)[0]

    mean_score = np.mean(np.array(scores))

    all_scores.append(mean_score)
    all_subjects_shap_values.append(mean_split_shap_values)
    all_mean_shap_values.append(mean_shap_values)
    if mean_score > SCORE_THRESHOLD : 
        good_subjects_shap.append(mean_shap_values)
        good_subjects_scores.append(mean_score)

    else : 
        bad_subjects_shap.append(mean_shap_values)
        bad_subjects_scores.append(mean_score)

    filename = os.path.join(OUT_DIR, 'all_shap_split_values.pkl') 
    with open(filename,'wb') as f: 
        pickle.dump(all_subjects_shap_values,f)

    np.save(f"{OUT_DIR}/all_scores.npy", np.array(all_scores))    
    np.save(f"{OUT_DIR}/good_subjects.npy", np.array(good_subjects_shap))
    np.save(f"{OUT_DIR}/bad_subjects.npy", np.array(bad_subjects_shap))
    np.save(f"{OUT_DIR}/good_subjects_scores.npy", np.array(good_subjects_scores))
    np.save(f"{OUT_DIR}/bad_subjects_scores.npy", np.array(bad_subjects_scores))
        


  