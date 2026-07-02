import torch 
import numpy as np
import torch.nn as nn
#from spd_learn.models import SPDNet
from spd_learn.functional import covariance
from spd_learn.modules import BiMap, CovLayer, LogEig, ReEig, SPDBatchNormMeanVar
from moabb.datasets import BNCI2014_001, Dreyer2023C, Beetl2021_A
from moabb.paradigms import FilterBankMotorImagery
from sklearn.model_selection import train_test_split
import pickle
import os
from warnings import warn
import mne

sensors = [
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
    ]

group_size = 3

montage = mne.channels.make_standard_montage('standard_1020')
all_positions = montage.get_positions()['ch_pos']

sensors_positions = np.array([all_positions[s] for s in sensors])

distances_to_center = np.linalg.norm(sensors_positions, axis=1)
indices_tries = np.argsort(distances_to_center)[::-1]
sensors_tries = [sensors[i] for i in indices_tries]

remaining_sensors = list(sensors)
groups = {}

for sensor in sensors_tries:
    if sensor not in remaining_sensors:
        continue

    dists = []
    for rem_s in remaining_sensors:
        d = np.linalg.norm(all_positions[sensor] - all_positions[rem_s])
        dists.append((rem_s, d))
    
    dists.sort(key=lambda x: x[1])

    current_group_names = [x[0] for x in dists[:group_size]]

    current_group_idx = [sensors.index(s) for s in current_group_names]
    groups[sensor] = current_group_names
    
    for member in current_group_names:
        remaining_sensors.remove(member)
print(groups)

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=len(sensors)//group_size - 1, random_state=0, n_init="auto").fit(sensors_positions)
print(kmeans.labels_)
k_means_groups = {}
for i, label in enumerate(kmeans.labels_):
    if label not in k_means_groups:
        k_means_groups[label] = []
    k_means_groups[label].append(sensors[i])

print(k_means_groups)