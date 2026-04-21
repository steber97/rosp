import numpy as np
from pprint import pprint
from typing import Tuple
from time import time

from utils import EPS, create_rand_psd_matrix
from lbs.gershgorin import gershgorin_lb
from piecewise_linear_maximize import maximize_x
from lbs.greedy_pm_shift import max_direction_lb
from scipy.optimize import minimize_scalar 

def argmax_x(M, V, bounds=None):
    """
    Solve

        argmax_x  min_i [ M[i,i] - x*V[i,i] - sum_{j!=i} |M[i,j] - x*V[i,j]| ]

    Parameters
    ----------
    M : (n,n) ndarray
    V : (n,n) ndarray
    bounds : tuple (lo, hi), optional
        Search interval for x. If given, bounded optimization is used.

    Returns
    -------
    x_star : float
    f_star : float
    """

    M = np.asarray(M, dtype=float)
    V = np.asarray(V, dtype=float)
    n = M.shape[0]

    def objective(x):
        row_vals = np.empty(n)

        for i in range(n):
            # diagonal term
            val = M[i, i] - x * V[i, i]

            # off-diagonal absolute value sum
            s = 0.0
            for j in range(n):
                if j != i:
                    s += abs(M[i, j] - x * V[i, j])

            row_vals[i] = val - s

        # maximize min_i(...)
        return np.min(row_vals)

    # scipy minimizes, so minimize negative objective
    def neg_objective(x):
        return -objective(x)

    if bounds is None:
        res = minimize_scalar(neg_objective, method='brent')
    else:
        res = minimize_scalar(
            neg_objective,
            bounds=bounds,
            method='bounded'
        )

    x_star = res.x
    f_star = objective(x_star)

    return x_star

def avg_direction_v2_lb(M: np.ndarray, *args) -> float:
    """
    args: repetitions. Namely, we try to write M = sum_{i=1}^{rep} x_i x_i^T 
            as the sum of rank-repetitions vectors.
    """
    n = len(M)
    repetitions = int(args[0]) if len(args) > 0 else 1  # Use as default 2
    M_copy = M.copy()
    for rank_k_approx in range(repetitions):
        dd_value = [(
            i, 
            M_copy[i,i] - np.sum(np.abs(np.concatenate((M_copy[i, :i], [] if i == n-1 else M_copy[i, i+1:]))))
            ) for i in range(n)]
        dd_value = sorted(dd_value, key=lambda x: x[1])
        # pprint(dd_value)
        rep = int(np.sum([x[1] < -EPS for x in dd_value]))
        directions = []
        if rep > 0:
            for k in range(rep):
                directions.append(np.ones(n))
                i, dd_val = dd_value[k]
                directions[-1][i] = min(1, M_copy[i,i])
                for j in range(n):
                    if j != i:
                        directions[-1][j] = M_copy[i,j] / (min(1, M_copy[i,i]))
                directions[-1] /= np.sqrt(directions[-1] @ directions[-1])
                assert abs(directions[-1]@directions[-1] - 1) < EPS
                if k > 0:
                    if directions[-1] @ directions[0] < EPS:
                        directions[-1] = -directions[-1]
            # pprint(directions)
            tot = np.sum([dd_value[k][1] for k in range(rep)])
            # print(tot)
            direction = np.sum([directions[k] * dd_value[k][1] / tot for k in range(rep)], axis=0)
            # print(direction)
            direction /= np.sqrt(direction @ direction)
            # print(direction)
            # print(direction, direction @ direction)
            x = maximize_x(M_copy, np.outer(direction, direction))
            
            # x = argmax_x(M_copy, np.outer(direction, direction), bounds=[0,np.max(M_copy)])
            # print(x)
            # x = 1
            M_copy -= x * np.outer(direction, direction)
            # print(M_copy)
    
    return gershgorin_lb(M_copy)

if __name__=="__main__":
    n = 500
    M = create_rand_psd_matrix(n)
    # print(M)
    print(gershgorin_lb(M))
    time_start = time()
    print(avg_direction_v2_lb(M))
    print("time total:", time() - time_start)


