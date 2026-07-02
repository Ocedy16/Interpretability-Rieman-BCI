import numpy as np
from sklearn.base import clone
import mne
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from pyriemann.estimation import Covariances
from pyriemann.classification import MDM, TSClassifier
from sklearn.model_selection import train_test_split

class FeaturePermutation:
    def __init__(self, X,y, classifier=MDM):
        self.X, self.y = X,np.array(y)
        self.nb_features = len(X[0])
        self.clf = classifier
        self.results_dic = {}

    def _permute_multiple_times(
        self, X, y_true, clf, feature,
        method, n_perm, seed
    ):
        rng = np.random.default_rng(seed)

        scores = []
        for k in range(n_perm):
            perm_seed = rng.integers(0, 1_000_000)

            if method == "across_trials":
                score = self.permute_channel_across_trials(
                    X, y_true, clf, feature,
                    random_state=perm_seed
                )

            elif method == "across_times":
                score = self.permute_channel_across_times(
                    X, y_true, clf, feature,
                    random_state=perm_seed
                )

            scores.append(score)

        return np.mean(scores)

    def _single_iteration(self, data, y, method, n_perm, seed=None):

        rng = np.random.default_rng(seed)

        indices = np.arange(len(data))

        train_idx, test_idx = train_test_split(
            indices,
            stratify=y,
            random_state=seed,
            test_size=0.2
        )

        train_eeg = [data[i] for i in train_idx]
        test_eeg  = [data[i] for i in test_idx]

        train_y = y[train_idx]
        test_y  = y[test_idx]


        cov = Covariances()
        C_train = np.stack([
            cov.transform(X.T)
            for X in train_eeg
        ])

        C_test = np.stack([
            cov.transform(X.T)
            for X in test_eeg
        ])

        clf = self.clf()
        clf.fit(C_train, train_y)

        baseline = clf.score(C_test, test_y)

        n_channels = train_eeg[0].shape[1]

        feature_importance = np.zeros(n_channels)

        for feature in range(n_channels):

            mean_perturbed = self._permute_multiple_times(
                test_eeg,
                test_y,
                clf,
                feature,
                method,
                n_perm,
                seed
            )

            feature_importance[feature] = baseline - mean_perturbed

        return baseline, feature_importance

    def fit(self, n_iter=10, n_perm=10, method="across_times", n_jobs=-1):
        if method not in self.results_dic:
            self.results_dic[method] = {}

        results = Parallel(n_jobs=n_jobs)(
            delayed(self._single_iteration)(
                self.X, self.y, method, n_perm, seed=i
            )
            for i in range(n_iter)
        )

        accuracies = np.array([r[0] for r in results])
        importances = np.array([r[1] for r in results])

        self.results_dic[method] = {
            "accuracy": accuracies,
            "importance": importances
        }
            

    def permute_channel_across_times(self, X, y_true, clf, feature, random_state=None):
        rng = np.random.default_rng(random_state)
        
        X_modified = [trial.copy() for trial in X]  # liste de (n_times, n_channels)
        
        for trial in X_modified:
            perm = rng.permutation(trial.shape[0])  
            trial[:, feature] = trial[perm, feature] 
        
        cov_estimator = Covariances()
        C_modified = np.stack([
            cov_estimator.transform(trial.T[np.newaxis])[0]
            for trial in X_modified
        ])
        
        return np.mean(clf.predict(C_modified) == y_true)
