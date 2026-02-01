import numpy as np
import typing
import math
from lbs.gershgorin import gershgorin_lb

def sos_lb(M: np.ndarray, *args) -> float:
    repetitions = 10
    n = len(M)
    v = np.random.rand(n)
    best_v = v.copy()
    best_lb = -math.inf
    for k in range(repetitions):
        for i in range(n):
            N = np.sum([M[i][j] * v[j] for j in range(n) if j != i])
            D = np.sum([v[j]**2 for j in range(n) if j != i])
            v[i] = N / D
        new_lb = gershgorin_lb(M - np.outer(v,v))
        if new_lb > best_lb:
            best_v = v.copy()
            best_lb = new_lb
    return best_lb