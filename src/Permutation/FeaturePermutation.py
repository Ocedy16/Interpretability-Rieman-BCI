import numpy as np
from sklearn.base import clone
import mne
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from pyriemann.estimation import Covariances
from pyriemann.classification import MDM, TSClassifier
from sklearn.model_selection import train_test_split

class FeaturePermutation:
    def __init__(self, dataset, paradigm, classifier=MDM):
        self.dataset = dataset
        self.paradigm = paradigm
        X, y, meta = self.paradigm.get_data(self.dataset)
        #self.X, self.y, self.meta = X[:,:,1024:],y, meta
        #self.X, self.y, self.meta = X[:,:,500:],y, meta
        self.X, self.y, self.meta = X,y,meta
        self.nb_features = self.X.shape[1]
        self.clf = classifier
        self.results_dic = {}

    def _permute_multiple_times(
        self, X, y_true, clf, cov_estimator, feature,
        method, n_perm, seed
    ):
        rng = np.random.default_rng(seed)

        scores = []
        for k in range(n_perm):
            perm_seed = rng.integers(0, 1_000_000)

            if method == "across_trials":
                score = self.permute_channel_across_trials(
                    X, y_true, clf, cov_estimator, feature,
                    random_state=perm_seed
                )

            elif method == "across_times":
                score = self.permute_channel_across_times(
                    X, y_true, clf, cov_estimator, feature,
                    random_state=perm_seed
                )

            scores.append(score)

        return np.mean(scores)

    def _single_iteration(self, data, y, method, n_perm, seed=None):

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

        clf = self.clf()
        clf.fit(C_train, train_y)

        baseline = clf.score(C_test, test_y)

        feature_importance = np.zeros(train_eeg.shape[1])

        for feature in range(train_eeg.shape[1]):
            mean_perturbed = self._permute_multiple_times(
                test_eeg, test_y, clf, cov_estimator,
                feature, method, n_perm, seed
            )

            feature_importance[feature] = baseline - mean_perturbed

        return baseline, feature_importance

    def fit(self, n_iter=10, n_perm=10, method="across_times", n_jobs=-1):
        if method not in self.results_dic:
            self.results_dic[method] = {}

        for subject in self.dataset.subject_list:
            subject_mask = self.meta.subject == subject
            data = self.X[subject_mask]
            y = self.y[subject_mask]

            results = Parallel(n_jobs=n_jobs)(
                delayed(self._single_iteration)(
                    data, y, method, n_perm, seed=i
                )
                for i in range(n_iter)
            )

            accuracies = np.array([r[0] for r in results])
            importances = np.array([r[1] for r in results])

            self.results_dic[method][subject] = {
                "accuracy": accuracies,
                "importance": importances
            }
            
    def permute_channel_across_trials(self, X, y_true, clf, cov_estimator, feature, random_state=None): 
        rng = np.random.default_rng(random_state) 
        X_modified = X.copy() 
        perm = rng.permutation(len(X)) 
        X_modified[:, feature, :] = X_modified[perm, feature, :] 
    
        C_modified = cov_estimator.transform(X_modified) 
        perturbed = np.mean(clf.predict(C_modified) == y_true) 
        self.permutations_across_trials = perturbed 
        return perturbed

    def permute_channel_across_times(self, X, y_true, clf, cov_estimator, feature, random_state=None): 
        rng = np.random.default_rng(random_state) 
        X_modified = X.copy() 
        for trial in range(X.shape[0]): 
            perm = rng.permutation(X.shape[2]) 
            X_modified[trial, feature, :] = X_modified[trial,feature, perm] 
        C_modified = cov_estimator.transform(X_modified) 
        perturbed = np.mean(clf.predict(C_modified) == y_true) 
        self.permutations_across_times = perturbed
        return perturbed

