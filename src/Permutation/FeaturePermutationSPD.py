import numpy as np


percent_spd = []
spd_values = []
eigenval_pos = []


def permute_sensor_variances(matrix, sensor_idx, rng):
    matrix_mod = matrix.copy()
    n_features = matrix.shape[0]

    other_indices = np.delete(np.arange(n_features), sensor_idx)
    target_idx = rng.choice(other_indices)

    matrix_mod[sensor_idx, sensor_idx] = matrix[target_idx, target_idx]
    matrix_mod[target_idx, target_idx] = matrix[sensor_idx, sensor_idx]
    return matrix_mod

def permute_sensor_covariances(matrix, sensor_idx, rng):
    matrix_mod = matrix.copy()
    n_features = matrix.shape[0]
    
    other_indices = np.delete(np.arange(n_features), sensor_idx)
    perm_indices = rng.permutation(other_indices)
    
    matrix_mod[sensor_idx, other_indices] = matrix[sensor_idx, perm_indices]
    matrix_mod[other_indices, sensor_idx] = matrix[perm_indices, sensor_idx]
    
    return matrix_mod

percent_spd = []
spd_values = []
eigenval_pos = []

def proj_on_spd(matrix, epsilon=1e-6):
    #print((matrix == matrix.T).all())
    eigenvals, eigenvecs = np.linalg.eigh(matrix)
    percent_spd.append(np.sum(eigenvals < 0) / len(eigenvals) * 100)
    spd_values.append(eigenvals[np.where(eigenvals<0)])
    eigenval_pos.append(eigenvals[np.where(eigenvals>0)])
    eigenvals_clipped = np.clip(eigenvals, epsilon, None)

    matrix_spd = eigenvecs @ np.diag(eigenvals_clipped) @ eigenvecs.T

    return matrix_spd


def permute_sensor_across_trials(C_matrices, sensor_idx, rng):
    n_trials = len(C_matrices)
    C_mod = [m.copy() for m in C_matrices]
    
    perm = rng.permutation(n_trials)
    n_features = C_matrices[0].shape[0]
    other_indices = np.delete(np.arange(n_features), sensor_idx)
    
    for i in range(n_trials):
        source_matrix = C_matrices[perm[i]]
        C_mod[i][sensor_idx, other_indices] = source_matrix[sensor_idx, other_indices]
        C_mod[i][other_indices, sensor_idx] = source_matrix[other_indices, sensor_idx]
    
    return C_mod
