import numpy as np

from utils import psd

def eig_lb(M: np.ndarray, *args) -> float:
    """
    Compute the smallest true eigenvalue of M.
    """
    assert psd(M)
    return np.min([x.real for x in np.linalg.eig(M)[0]])