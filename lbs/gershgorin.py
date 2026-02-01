import numpy as np


def gershgorin_lb(M: np.ndarray, *args) -> float:
    """
    Simple gershgorin circle theorem: the smallest eigenvalue is >= than
    the smallest circle (difference between diagonal element and the absolute value
    of the off-diagonal elements).
    """
    n = len(M)
    return np.min([M[i,i] - np.sum(np.abs(np.concatenate([M[i,:i], M[i,i+1:]]))) for i in range(n)])