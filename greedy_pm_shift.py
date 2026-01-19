import numpy as np
from typing import Tuple

from utils import EPS

import deville
from gershgorin import gershgorin_lb
from piecewise_linear_maximize import maximize_x


def max_direction_lb(M: np.ndarray, stop_early: bool = True) -> float:
    """
    Simple heuristic: given a matrix, it produces a shift (+ Gershgorin circle theorem).
    The shift is computed to improve greedily the worst row.
    The time complexity is O(n^2 * log(n)).
    """
    n = len(M)
    dd = np.array([M[i,i] - np.sum(np.abs(np.concatenate([M[i, :i], M[i, i+1:]]))) for i in range(n)])
    min_dd = np.min(dd)
    best_shift = np.ones(n)
    best_lb = deville.deville_lb(M)
    best_x = maximize_x(M, np.outer(best_shift, best_shift))
    for i in range(n):
        if np.abs(dd[i] - min_dd) < EPS:
            shift_v = np.ones_like(M[i])
            for j in range(n):
                if j != i:
                    shift_v[j] = shift_v[j] * (1 if M[i,j] > EPS else -1 if M[i,j] < -EPS else 0)
            x = maximize_x(M, np.outer(shift_v, shift_v))
            res = gershgorin_lb(M - x * np.outer(shift_v, shift_v))
            if res > best_lb:
                best_x = x
                best_lb = res
            best_shift = shift_v
            if stop_early:
                break
    return best_lb
