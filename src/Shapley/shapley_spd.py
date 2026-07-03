import numpy as np
import torch
import shap
from shap import KernelExplainer
from functools import partial
from sklearn.model_selection import train_test_split
from pyriemann.estimation import Covariances
from pyriemann.utils.mean import mean_riemann



percent_spd = []
spd_values = []
eigenval_pos = []

def stable_predict(mask_2d,n_channels,current_run_signal,random_reference_signal,clf):
    # mask_2d a une forme (N_simulations, n_channels)
    n_simulations = mask_2d.shape[0]
    X_reconstructed = np.array([current_run_signal for _ in range(n_simulations)])
    
    for i in range(n_simulations):
        for ch in range(n_channels):
            # Si le masque SHAP dit 1, on prend le signal du run actuel
            if mask_2d[i, ch] < 0.5:
                X_reconstructed[i, ch, :] = random_reference_signal[ch, :]
                X_reconstructed[i,:,ch] = random_reference_signal[:, ch]

    print(X_reconstructed.shape)
    X_reconstructed  = np.array([proj_on_spd(x) for x in X_reconstructed])

    return clf.predict_proba(X_reconstructed)[:,0]


def KernelShap(C_train,C_test, clf, baseline,n_samples=2000):
    all_shap_values = []

    n_trials, n_channels, _ = C_train.shape
    random_reference_signal = baseline

    for run_index in range(C_test.shape[0]):
        current_run_signal = C_test[run_index] 
        
        stable_predict_frozen = partial(stable_predict, n_channels=n_channels, 
                                        current_run_signal=current_run_signal, 
                                        random_reference_signal=random_reference_signal, clf=clf)
        explainer = KernelExplainer(stable_predict_frozen, np.zeros((1, n_channels)),seed=run_index)
        
        
        shap_values = explainer.shap_values(np.ones((1, n_channels)), nsamples=n_samples)

        all_shap_values.append(shap_values)

    return all_shap_values


def compute_shapley(X,y, n_splits, clf):
    all_shap_values = []
    all_scores = []
    for i in range (n_splits):
        print("Split",i)
        baseline = X[np.where(y=='rest')]
        X,y = X[np.where(y!="rest")], y[np.where(y!='rest')]
        X_train,X_test,y_train,y_test = train_test_split(X,y, train_size=0.8, stratify=y, random_state=i)
        covs_transformer = Covariances()
        C_train = covs_transformer.fit_transform(X_train) 
        C_test = covs_transformer.transform(X_test)
        C_rest = covs_transformer.transform(baseline)
        y_rest = y[np.where(y=='rest')]
        clf.fit(C_train,y_train)
        probas = clf.predict_proba(C_test)
        shap_values = KernelShap(C_train,C_test, clf, baseline = mean_riemann(C_test), n_samples=1000)
        all_shap_values.append(shap_values)
        all_scores.append(clf.score(C_test,y_test))

    return all_shap_values, all_scores



def proj_on_spd(matrix, epsilon=1e-6):
    #print((matrix == matrix.T).all())
    eigenvals, eigenvecs = np.linalg.eigh(matrix)
    percent_spd.append(np.sum(eigenvals < 0) / len(eigenvals) * 100)
    spd_values.append(eigenvals[np.where(eigenvals<0)])
    eigenval_pos.append(eigenvals[np.where(eigenvals>0)])
    eigenvals_clipped = np.clip(eigenvals, epsilon, None)

    matrix_spd = eigenvecs @ np.diag(eigenvals_clipped) @ eigenvecs.T

    return matrix_spd

