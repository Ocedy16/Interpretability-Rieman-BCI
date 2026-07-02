import mne
import torch 
import numpy as np
import torch.nn as nn
from joblib import Parallel, delayed
from pyriemann.estimation import Covariances
from pyriemann.classification import MDM
from sklearn.model_selection import train_test_split
from sklearn.neighbors import kneighbors_graph

class GroupedFeaturePermutation:
    def __init__(self, dataset, paradigm, sensors, classifier=MDM):
        self.dataset = dataset
        self.paradigm = paradigm
        X, y, meta = self.paradigm.get_data(self.dataset)
        #self.X, self.y, self.meta = X[:,:,1024:],y, meta
        #self.X, self.y, self.meta = X[:,:,500:],y, meta
        self.X, self.y, self.meta = X,y,meta
        self.sensor_names = sensors 
        self.clf_class = classifier
        self.results_dic = {}
        self.groups = {}

    def sensors_to_groups(self):
        montage = mne.channels.make_standard_montage('standard_1020')

        all_positions = montage.get_positions()['ch_pos']

        sensors_positions = np.array([all_positions[s] for s in self.sensor_names])

        A = kneighbors_graph(
            sensors_positions,
            n_neighbors=4,
            mode='connectivity',
            include_self=True
        ).toarray()
        
        groups = {}

        for i, sensor in enumerate(self.sensor_names):

            connected_indices = np.where(A[i]>0)[0]

            current_group_names = np.array(self.sensor_names)[connected_indices]

            current_group_idx = [self.sensor_names.index(s) for s in current_group_names]
            groups[sensor] = current_group_idx

        np.save('Results/GroupedPermutationOverlap/TangentSpace/Beetl2021-A/With_cue/groups.npy', groups)

        return groups

    def _permute_multiple_times(self, X, y_true, clf, cov_estimator, group_indices, method, n_perm, seed):
        rng = np.random.default_rng(seed)
        scores = []
        for _ in range(n_perm):
            perm_seed = rng.integers(0, 1_000_000)
            if method == "across_trials":
                score = self.permute_channel_across_trials(X, y_true, clf, cov_estimator, group_indices, perm_seed)
            else:
                score = self.permute_channel_across_times(X, y_true, clf, cov_estimator, group_indices, perm_seed)
            scores.append(score)
        return np.mean(scores)


    def _single_iteration(self, data, y, method, n_perm, seed=None):
        rng = np.random.default_rng(seed)

        indices = rng.permutation(len(data))
        data = data[indices]
        y = y[indices]

        train_eeg, test_eeg, train_y, test_y = train_test_split(
            data,
            y,
            stratify=y,
            random_state=seed,
            test_size=0.2
        )

        cov_estimator = Covariances()
        C_train = cov_estimator.fit_transform(train_eeg)
        C_test = cov_estimator.transform(test_eeg)

        clf = self.clf_class()
        clf.fit(C_train, train_y)
        baseline = clf.score(C_test, test_y)

        importances = np.zeros(len(self.sensor_names))
        counts = np.zeros(len(self.sensor_names))  # pour moyenner les overlaps

        for group_seed, group_indices in self.groups.items():
            mean_perturbed = self._permute_multiple_times(
                test_eeg, test_y, clf, cov_estimator,
                group_indices, method, n_perm, seed
            )
            score = baseline - mean_perturbed
            for idx in group_indices:
                importances[idx] += score
                counts[idx] += 1

        importances /= np.where(counts > 0, counts, 1)

        return baseline, importances

    '''
    #version sans moyenner les overlaps
    def _single_iteration(self, data, y, method, n_perm, seed=None):
        train_eeg, test_eeg, train_y, test_y = train_test_split(
            data, y, stratify=y, random_state=seed, test_size=0.2
        )

        cov_estimator = Covariances()
        C_train = cov_estimator.fit_transform(train_eeg)
        C_test = cov_estimator.transform(test_eeg)

        clf = self.clf_class()
        clf.fit(C_train, train_y)
        baseline = clf.score(C_test, test_y)

        importances = np.zeros(train_eeg.shape[1])

        for group_seed, group_indices in self.groups.items():
            mean_perturbed = self._permute_multiple_times(
                test_eeg, test_y, clf, cov_estimator,
                group_indices, method, n_perm, seed
            )
            sensor_idx = self.sensor_names.index(group_seed)
            importances[sensor_idx] = baseline - mean_perturbed

        return baseline, importances
    '''

    def fit(self, n_iter=5, n_perm=5, method="across_times", n_jobs=-1):
        self.groups = self.sensors_to_groups() # Correctly calling the method

        if method not in self.results_dic:
            self.results_dic[method] = {}

        for subject in self.dataset.subject_list:
            mask = self.meta.subject == subject
            data, y = self.X[mask], self.y[mask]

            results = Parallel(n_jobs=n_jobs)(
                delayed(self._single_iteration)(data, y, method, n_perm, seed=i)
                for i in range(n_iter)
            )

            self.results_dic[method][subject] = {
                "accuracy": [r[0] for r in results],
                "importance": [r[1] for r in results]
            }

    def permute_channel_across_trials(self, X, y_true, clf, cov_estimator, group_indices, random_state):
        X_mod = X.copy()
        rng = np.random.default_rng(random_state)
        for idx in group_indices:
            perm = rng.permutation(X.shape[0])
            print(idx)
            print(perm)
            X_mod[:, idx, :] = X_mod[perm, idx, :]
        
        C_mod = cov_estimator.transform(X_mod)
        return np.mean(clf.predict(C_mod) == y_true)

    def permute_channel_across_times(self, X, y_true, clf, cov_estimator, group_indices, random_state):
        X_mod = X.copy()
        rng = np.random.default_rng(random_state)
        n_times = X.shape[2]
        for trial in range(X.shape[0]):
            for idx in group_indices:
                perm = rng.permutation(n_times)
                X_mod[trial, idx, :] = X_mod[trial, idx, perm]
        
        C_mod = cov_estimator.transform(X_mod)
        return np.mean(clf.predict(C_mod) == y_true)
