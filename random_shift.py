import math
import numpy as np

from gershgorin import gershgorin_lb
from piecewise_linear_maximize import maximize_x
from utils import EPS


def single_random_lb(M: np.ndarray) -> float:
    shift = np.random.rand(len(M))
    x = maximize_x(M, np.outer(shift, shift))
    if x > EPS:
        SM = M - x * np.outer(shift, shift)
        assert gershgorin_lb(SM) >= gershgorin_lb(M) - EPS
        return gershgorin_lb(SM)
    return gershgorin_lb(M)


def random_lb(M: np.ndarray) -> float:
    rep = int(np.ceil(np.log(len(M))))
    best_lb = -math.inf
    for r in range(rep):
        lb = single_random_lb(M)
        if lb > best_lb:
            best_lb = lb
    return best_lb
