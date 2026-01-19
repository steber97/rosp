import numpy as np

def eig_lb(M: np.ndarray) -> float:
    """
    Compute the smallest true eigenvalue of M.
    """
    return np.linalg.eig(M)[0].min()