import numpy as np

def eig_lb(M: np.ndarray, *args) -> float:
    """
    Compute the smallest true eigenvalue of M.
    """
    return np.linalg.eig(M)[0].min()