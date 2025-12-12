import numpy as np
from typing import Tuple, List

EPS = 1e-5

def zero_one_with_probability(prob: float = 0.5) -> int:
    if np.random.rand() <= prob:
        return 1
    return 0


def create_rand_matrix(n: int, prob: float=0.5) -> np.array:
    M = np.identity(n)
    for i in range(n):
        M[i,i] = n-2
    for i in range(n):
        for j in range(i+1, n):
            M[i,j] = 2 * (-0.5 + zero_one_with_probability(prob))
            M[j,i] = M[i,j]
    return M

def diagonally_dominant(M: np.array) -> bool:
    return np.all(
        (2*np.sum((M * np.eye(M.shape[0])), axis=0) - np.sum(np.abs(M), axis=0)) >= 0
    )

def psd(M: np.array) -> bool:
    return np.linalg.eig(M)[0].min() >= 0

def is_vect_eigenv(M: np.array, v: np.array) -> bool:
    """
    Check if v is an eigenvector for M
    """
    w = M @ v
    w = w / v
    w -= w[0]
    return np.all(np.abs(w) < 1e-6)
