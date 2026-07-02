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

np.random.seed(3)


N_SPLITS = 10
N_PERMS = 10
SCORE_THRESHOLD = 0.75

classifier = MDM
pipeline = Pipeline([
    ("cov", Covariances()),
    ("clf", classifier())
])


OUT_DIR = 'Results/Permutation/MDM/Air_pollution'


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

    if __name__ =='__main__':
        X, y = Matrices, numz
        perm = FeaturePermutation(X, y, classifier)
        perm.fit(n_iter = N_SPLITS, n_perm = N_PERMS)

        results_dic = perm.results_dic
        filename = os.path.join(OUT_DIR, 'results_dic.pkl')
        with open(filename,'wb') as f:
            pickle.dump(results_dic,f)


        print(results_dic)

