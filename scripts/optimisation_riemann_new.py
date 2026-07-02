import pymanopt
import numpy as np
from pyriemann.classification import MDM
from pyriemann.utils.mean import mean_riemann
from pyriemann.estimation import Covariances
from moabb.paradigms import FilterBankMotorImagery
from moabb.datasets import BNCI2014_001, Dreyer2023C
from sklearn.model_selection import train_test_split
from shap import KernelExplainer
from functools import partial
from pyriemann.utils.distance import distance_riemann
from joblib import Parallel, delayed
import warnings
import time
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)


def solve_riemannian(test_matrix, mask, lmbd=1):
    d = test_matrix.shape[0]
    idx = np.where(mask)[0]
    # M = np.eye(d)[:, mask]
    t_red = test_matrix[np.ix_(idx, idx)]

    # t_red = M.T@test_matrix@M
    @pymanopt.function.numpy(manifold)
    def cost(point):
        """
        mask correspond à l'application de la matrice M
        """
        d_full = distance_riemann(point, mean_matrix) ** 2

        # p_red = M.T@point@M
        p_red = point[np.ix_(idx, idx)]

        d_reduced = distance_riemann(p_red, t_red) ** 2
        return d_full + lmbd * d_reduced

    @pymanopt.function.numpy(manifold)
    def rgrad(point):
        # One eigh(point) → S^{1/2} and S^{-1/2}
        eigvals_S, V_S = np.linalg.eigh(point)
        sqrt_e = np.sqrt(eigvals_S)
        S_sqrt = (V_S * sqrt_e) @ V_S.T
        S_invsqrt = (V_S / sqrt_e) @ V_S.T

        # g1 = log_{point}(mean): W1 = S^{1/2} V_A avoids one matmul
        A = S_invsqrt @ mean_matrix @ S_invsqrt
        eigvals_A, V_A = np.linalg.eigh(A)
        W1 = S_sqrt @ V_A  # (d, d)
        g1 = (W1 * np.log(eigvals_A)) @ W1.T

        # rgrad_2 = point @ g2 @ point, fully fused
        # egrad_P = P^{-1/2} log(P^{-1/2} t_red P^{-1/2}) P^{-1/2}
        # g2 = M @ egrad_P @ M^T  →  rgrad_2 = point @ g2 @ point
        # p_red = M.T @ point @ M
        p_red = point[np.ix_(idx, idx)]

        eigvals_P, V_P = np.linalg.eigh(p_red)
        P_invsqrt = (V_P / np.sqrt(eigvals_P)) @ V_P.T
        B = P_invsqrt @ t_red @ P_invsqrt
        eigvals_B, V_B = np.linalg.eigh(B)
        # W2 = point @ M @ (P_invsqrt @ V_B)              # (d, d-1)
        W2 = point[:, idx] @ (P_invsqrt @ V_B)
        rgrad_2 = (W2 * np.log(eigvals_B)) @ W2.T

        return -2 * (g1 + lmbd * rgrad_2)

    problem = pymanopt.Problem(manifold, cost, riemannian_gradient=rgrad)
    # check_gradient(problem)
    optimizer = pymanopt.optimizers.ConjugateGradient(
        verbosity=0, log_verbosity=2, max_iterations=10
    )
    result = optimizer.run(problem, initial_point=test_matrix)
    # all_losses.append(result.log["iterations"]["cost"])
    return result.point


def shap_predict(mask_2d, test_matrix, pipeline: MDM) -> np.ndarray:
    n_simulations = mask_2d.shape[0]
    probas = np.zeros(n_simulations)
    cache = {}

    for i in range(n_simulations):
        mask_key = tuple(mask_2d[i].astype(int).tolist())

        if mask_key not in cache:
            mask_tensor = np.array(mask_key, dtype=bool)

            if mask_tensor.sum() < 2:
                sol = mean_matrix
            else:
                sol = solve_riemannian(test_matrix, mask_tensor)

            cache[mask_key] = sol

        sol = cache[mask_key]

        sol_3d = sol[np.newaxis, :, :]
        probas[i] = pipeline.predict_proba(sol_3d)[0, 0]

    return probas


def KernelShap(C_test, pipeline: MDM, n_samples: int = 200):

    n_trials, dim, _ = C_test.shape

    def _explain_one(run_index):
        test_matrix = C_test[run_index]

        predict_fn = partial(shap_predict, test_matrix=test_matrix, pipeline=pipeline)

        baseline = np.zeros((1, dim))
        foreground = np.ones((1, dim))

        explainer = KernelExplainer(predict_fn, baseline, seed=run_index)
        shap_values = explainer.shap_values(foreground, nsamples=n_samples, silent=True)

        print(f"Trial {run_index+1}/{n_trials} — SHAP terminé")
        return shap_values[0]

    # Each trial is independent; mean_matrix / manifold are read-only globals.
    # seed=run_index keeps results deterministic regardless of execution order.
    all_shap_values = Parallel(n_jobs=-1)(
        delayed(_explain_one)(run_index) for run_index in range(n_trials)
    )
    # all_shap_values = []
    # for run_index in range(n_trials):
    #     all_shap_values.append(_explain_one(run_index))

    return all_shap_values


np.random.seed(3)

paradigm = FilterBankMotorImagery(filters=[(7, 35)])
dataset = BNCI2014_001()
for subject in dataset.subject_list[2:] : 
    t0 = time.perf_counter()
    X, y, meta = paradigm.get_data(dataset=dataset, subjects=[subject])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)

    covs = Covariances()
    C_train = covs.fit_transform(X_train)
    C_test = covs.fit_transform(X_test)

    mdm = MDM()
    mdm.fit(C_train, y_train)

    dim = X.shape[1]
    manifold = pymanopt.manifolds.SymmetricPositiveDefinite(dim)

    mean_matrix = mean_riemann(C_train)
    shap_values = KernelShap(C_test, mdm, n_samples=1000)
    elapsed = time.perf_counter() - t0
    print(f"KernelShap: {elapsed:.2f} s")

    shap_array = np.array(shap_values)
    np.save(f"Results/Shapley/MDM/BNCI2014_001/Shapley_optim/shap_values_subject_{subject}_mdm.npy", shap_array)
    print("Shape SHAP values :", shap_array.shape)
