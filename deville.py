import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from typing import Tuple
import pandas as pd
import time
from tqdm import tqdm

from utils import EPS
from piecewise_linear_maximize import maximize_x
import greedy_pm_shift

def gap_diagonally_dominance(x: float, M: np.ndarray, v: np.ndarray) -> float:
    """
    Compute the max dd-gap on any row of M, when applying the shift x * vv^T.
    """
    n = len(v)
    assert len(M) == n and x >= -EPS
    SM = M - x * np.outer(v, v)
    dd = [SM[i,i] - np.sum(np.abs(np.concatenate([SM[i, :i], SM[i, i+1:]]))) for i in range(n)]
    return min(dd)

def maximize_gershgoryn_circle(M, v):
    # maximize f by minimizing -f(x)
    # res = minimize_scalar(lambda x: -gap_diagonally_dominance(x, M, v), bounds=[0, np.max(M)], )
    # return res.x, gap_diagonally_dominance(res.x, M, v)
    # if abs(res.x) > EPS and abs(gap_diagonally_dominance(res.x, M, v) - gap_diagonally_dominance(x, M, v)) > EPS:
    #     print(res.x, x, gap_diagonally_dominance(res.x, M, v), gap_diagonally_dominance(x, M, v), M, v)
    #     raise AssertionError

    x = maximize_x(M, np.outer(v, v))
    return x, gap_diagonally_dominance(x, M, v)

def delta_i_k(M:  np.ndarray, i: int, k: int) -> float:
    """
    Def 3.3 of paper.
    Careful because all indices are shifted by one.
    Time complexity O(n * log(n))
    """
    n = len(M)
    assert 0 <= i < n and 0 <= k < n
    y = sorted([M[i,j] for j in range(n) if i != j])
    return np.sum(y[:k]) - np.sum(y[k:])

def s_i_k(M: np.ndarray, i: int, k: int) -> float:
    """
    Def 3.3 of paper
    Careful because all indices are shifted by one.
    Time complexity O(n * log(n)) (due to delta_i_k)
    """
    n = len(M)
    assert 0 <= i < n and 0 <= k < n
    return M[i,i] + delta_i_k(M, i, k)

def s_k(M: np.ndarray, k: int) -> float:
    """
    Def 3.3 of paper
    Careful because all indices are shifted by one.
    Time complexity O(n^2 * log(n))
    """
    n = len(M)
    assert 0 <= k < n
    return np.min([s_i_k(M, i, k) for i in range(n)])

def compute_x(M: np.ndarray) -> float:
    """
    Corollary 3.5 of paper
    """
    n = len(M)
    Q = [(k, l) for k in range(1, n+1) for l in range(1, n+1) if k <= (n/2) and l > (n/2) and s_k(M, l-1) >= s_k(M, k-1) - EPS]
    if len(Q) == 0:
        # No way to improve it.
        return 0.0
    min_l_k = np.argmin([((n/2 - k) * s_k(M, l-1) + (l - n/2) * s_k(M, k-1))/(l-k) for (k, l) in Q])
    l, k = Q[min_l_k]
    x = (s_k(M, l-1) - s_k(M, k-1))/(2*(l-k))
    return x

def gershgorin_lb(M: np.ndarray) -> float:
    n = len(M)
    return np.min([M[i,i] - np.sum(np.abs(np.concatenate([M[i,:i], M[i,i+1:]]))) for i in range(n)])

def R_i(M: np.ndarray, i: int) -> float:
    """
    Simple sum of off-diagonal elements
    """
    return np.sum(np.abs(np.concatenate([M[i,:i], M[i, i+1:]])))

def brauers_lb(M: np.ndarray) -> float:
    n = len(M)
    return np.min([((M[i,i] + M[j,j]) / 2) - (np.sqrt((M[i,i] - M[j,j])**2 + R_i(M, i) * R_i(M, j))) for i in range(n) for j in range(n) if i != j])

def deville_lb(M: np.ndarray) -> float:
    n = len(M)
    # x = compute_x(M)
    x_2, min_gersh_circle = maximize_gershgoryn_circle(M, np.ones(n))
    
    # Here it looks that x_2 optimized using the optimizer is better for some reason? :/
    # if np.abs(gap_diagonally_dominance(x_2, M, np.ones(n)) - gap_diagonally_dominance(x, M, np.ones(n)))>EPS:
    #     print(x, x_2)
    #     print(gap_diagonally_dominance(x, M, np.ones(n)), gap_diagonally_dominance(x_2  , M, np.ones(n)))
    if x_2 > EPS:
        SM = M - x_2 * np.ones_like(M)
        assert gershgorin_lb(SM) >= gershgorin_lb(M) - EPS
        return gershgorin_lb(SM)
    return gershgorin_lb(M)


def eig_lb(M: np.ndarray) -> float:
    """
    Compute the smallest true eigenvalue of M.
    """
    return np.linalg.eig(M)[0].min()


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

if __name__ == "__main__":
    
    np.random.seed(42)
    n = 10
    attempts = 10
    range_values = (0,11)  # inclusive, exclusive
    sign_perc = 0.5
    diag_boost = 20
    
    lb_functions = [
        # (gershgorin_lb, "gershgorin"),
        (deville_lb, "deville"),
        # (brauers_lb, "brauers"),
        (eig_lb, "eigenvalue"),
        (greedy_pm_shift.shift_as_max_direction, "greedy")
    ]
    df_result = pd.DataFrame(columns=[lb_f[1] for lb_f in lb_functions] + [lb_f[1] + "_time" for lb_f in lb_functions])

    for att in tqdm(range(attempts)):
        M = create_rand_symmetric_matrix(n, range_values, sign_perc, diag_boost)
        if att == 0:
            print(M)
        row = {}
        for lb_f, lb_name in lb_functions:
            start = time.time()
            row[lb_name] = lb_f(M)
            end = time.time() - start
            row[lb_name+"_time"] = end
        df_result.loc[len(df_result)] = row
    print(df_result.describe())

    #plt.boxplot(np.array(df_result['greedy']) - np.array(df_result['deville']))
    # plt.ylabel("Difference between DeVille lb and our lb")

    f, (ax1, ax2) = plt.subplots(1, 2)

    ax1.boxplot(df_result[[lb_name + '_time' for lb_f, lb_name in lb_functions]], 
                labels=[lb_name for lb_f, lb_name in lb_functions])
    ax1.set_ylabel("time")

    ax2.boxplot(df_result[[lb_name for lb_f, lb_name in lb_functions]], 
                labels=[lb_name for lb_f, lb_name in lb_functions])
    ax2.set_ylabel("LB")
    plt.show()



