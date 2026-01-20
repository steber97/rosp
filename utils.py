import numpy as np
from typing import Tuple, List

EPS = 1e-5

def zero_one_with_probability(prob: float = 0.5) -> int:
    if np.random.rand() <= prob:
        return 1
    return 0


def create_rand_matrix(n: int, prob: float=0.5) -> np.ndarray:
    M = np.identity(n)
    for i in range(n):
        M[i,i] = n-2
    for i in range(n):
        for j in range(i+1, n):
            M[i,j] = 2 * (-0.5 + zero_one_with_probability(prob))
            M[j,i] = M[i,j]
    return M


def diagonally_dominant(M: np.ndarray) -> bool:
    return bool(np.all(
        (2*np.sum((M * np.eye(M.shape[0])), axis=0) - np.sum(np.abs(M), axis=0)) >= 0
    ))


def psd(M: np.ndarray) -> bool:
    return np.linalg.eig(M)[0].min() >= 0


def is_vect_eigenv(M: np.ndarray, v: np.ndarray) -> bool:
    """
    Check if v is an eigenvector for M
    """
    w = M @ v
    w = w / v
    w -= w[0]
    return bool(np.all(np.abs(w) < 1e-6))


def create_rand_symmetric_matrix(n: int, range_values: Tuple[int, int], sign_perc: float, diag_boost: float = 0):
    M = np.random.randint(low=range_values[0], high=range_values[1], size=(n, n))
    # Make it symmetric.
    for i in range(n):
        M[i,i] += diag_boost
        for j in range(i):
            if np.random.rand() < sign_perc:
                M[j,i] = -M[j,i]
            M[i,j] = M[j,i]
    return M


def create_rand_symmetric(n: int) -> np.ndarray:
    M = (np.random.rand(n**2).reshape(n, n) - 0.5) * 2
    for i in range(n):
        # M[i,i] = np.abs(M[i,i])
        for j in range(i):
            M[i,j] = M[j,i]
    return M


def create_rand_dd(n):
    M = create_rand_symmetric(n)
    for i in range(n):
        M[i,i] += np.sum(np.abs(np.concatenate([M[i,:i], ([] if i == n-1 else M[i+1, :])]))) - M[i,i]
    return M

def create_rand_dd_plus_ros(n: int, alpha_dd: float = 0.5) -> np.ndarray:
    v = (np.random.rand(n) -0.5) * 2
    dd = create_rand_dd(n)
    return alpha_dd * dd + (1-alpha_dd) * np.outer(v, v)
