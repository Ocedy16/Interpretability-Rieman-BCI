import numpy as np
from joblib import Parallel, delayed
from pyriemann.estimation import Covariances
from pyriemann.classification import MDM, TSClassifier
import mne


class FeaturePermutation:
    def __init__(self, dataset, paradigm, classifier=MDM):
        self.dataset = dataset
        self.paradigm = paradigm
        self.X, self.y, self.meta = self.paradigm.get_data(self.dataset)
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
        rng = np.random.default_rng(seed)

        indices = rng.permutation(len(data))
        shuffled_data = data[indices]
        shuffled_y = y[indices]

        split = int(0.8 * len(shuffled_data))
        train_eeg, test_eeg = shuffled_data[:split], shuffled_data[split:]
        train_y, test_y = shuffled_y[:split], shuffled_y[split:]

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


    def visualize_channel_importance(self, subject = 0, montage_type='standard_1020', method = "across_trials"):

        if method not in self.results_dic:
            raise ValueError(f"La méthode {method} n'a pas encore été calculée. Appliquez d'abord fit sur les données avec celle_ci.")

        importance_values = np.mean(self.results_dic[method][subject]['importance'], axis=0) 
        
        montage = mne.channels.make_standard_montage(montage_type)
        valid_eeg = set(montage.ch_names)

        self.ch_names = [
            ch for ch in self.dataset.METADATA.acquisition.sensors
            if ch in valid_eeg
        ]

        info = mne.create_info(ch_names=self.ch_names, sfreq=500, ch_types="eeg")
        
        info.set_montage(montage)
        
        fig, ax = plt.subplots(figsize=(6, 6))

        im, _ = mne.viz.plot_topomap(
            importance_values,
            info,
            axes=ax,
            show=False,
            cmap="RdBu_r",
        )
    
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Importance (baseline - perturbed)")

        acc_mean = np.mean(self.results_dic[method][subject]['accuracy'])
        acc_std = np.std(self.results_dic[method][subject]['accuracy'])
        
        plt.title(f"Sujet {subject} Score : {acc_mean:.2f} ± {acc_std:.2f}")
            
        plt.savefig(f'Sujet_{subject}')
        print(' it is saved')
        plt.show()
