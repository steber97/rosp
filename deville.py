import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from typing import Tuple

from utils import EPS
from piecewise_linear_maximize import maximize_x
import greedy_pm_shift

def gap_diagonally_dominance(x: float, M: np.array, v: np.array) -> float:
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
    x = maximize_x(M, np.outer(v, v))
    # if abs(res.x) > EPS and abs(gap_diagonally_dominance(res.x, M, v) - gap_diagonally_dominance(x, M, v)) > EPS:
    #     print(res.x, x, gap_diagonally_dominance(res.x, M, v), gap_diagonally_dominance(x, M, v), M, v)
    #     raise AssertionError
    return x, gap_diagonally_dominance(x, M, v)

def delta_i_k(M: np.array, i: int, k: int) -> float:
    """
    Def 3.3 of paper.
    Careful because all indices are shifted by one.
    """
    n = len(M)
    assert 0 <= i < n and 0 <= k < n
    y = sorted([M[i,j] for j in range(n) if i != j])
    return np.sum(y[:k]) - np.sum(y[k:])

def s_i_k(M: np.array, i: int, k: int) -> float:
    """
    Def 3.3 of paper
    Careful because all indices are shifted by one.
    """
    n = len(M)
    assert 0 <= i < n and 0 <= k < n
    return M[i,i] + delta_i_k(M, i, k)

def s_k(M: np.array, k: int) -> float:
    """
    Def 3.3 of paper
    Careful because all indices are shifted by one.
    """
    n = len(M)
    assert 0 <= k < n
    return np.min([s_i_k(M, i, k) for i in range(n)])

def compute_x(M: np.array) -> float:
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

def gershgorin_lb(M: np.array) -> float:
    n = len(M)
    return np.min([M[i,i] - np.sum(np.abs(np.concatenate([M[i,:i], M[i,i+1:]]))) for i in range(n)])

def R_i(M: np.array, i: int) -> float:
    """
    Simple sum of off-diagonal elements
    """
    return np.sum(np.abs(np.concatenate([M[i,:i], M[i, i+1:]])))

def brauers_lb(M: np.array) -> float:
    n = len(M)
    return np.min([((M[i,i] + M[j,j]) / 2) - (np.sqrt((M[i,i] - M[j,j])**2 + R_i(M, i) * R_i(M, j))) for i in range(n) for j in range(n) if i != j])

def deville_lb(M: np.array) -> float:
    n = len(M)
    x = compute_x(M)
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


def create_rand_symmetric_matrix(n: int, range_values: Tuple[int, int], sign_perc: float, diag_boost: float = 0):
    M = np.random.randint(np.ones(n**2) * range_values[0], np.ones(n**2) * range_values[1]).reshape(n, n)
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
    n = 5
    attempts = 1000
    range_values = (0,11)  # inclusive, exclusive
    sign_perc = 0.5
    diag_boost = 15
        
    g_lb = []
    d_lb = []
    b_lb = []
    pm_greedy_lb = []
    pm_lb = []
    eigvals = []
    values = []
    for att in range(attempts):
        M = create_rand_symmetric_matrix(n, range_values, sign_perc, diag_boost)
        if att == 0:
            print(M)
        
        g_lb.append(gershgorin_lb(M))
        d_lb.append(deville_lb(M))
        b_lb.append(brauers_lb(M))
        eigvals.append(np.linalg.eig(M)[0].min())
        pm_greedy_lb.append(greedy_pm_shift.shift_as_max_direction(M))
        pm_lb.append(greedy_pm_shift.shift_as_max_direction(M, stop_early=False))
        values.append((g_lb[-1], d_lb[-1], b_lb[-1], eigvals[-1], pm_lb[-1], pm_greedy_lb[-1]))
    
    values = sorted(values, key=lambda x: x[2])

    legend_names = ['gersh', 'deville', 'brauer', 'eigv', 'greedy_pm_se', 'greedy_pm']
    # for j in range(5):
    # for j in [1, 4]:
    #     plt.scatter([i for i in range(attempts)], [values[i][j] for i in range(attempts)], label=legend_names[j], alpha=0.5)
    #     plt.plot([i for i in range(attempts)], [0 for i in range(attempts)])
    # plt.ylabel("lowest eigenvalue")
    # plt.xlabel("random attempt")
    # plt.legend()
    # plt.show()

    plt.boxplot(np.array(pm_lb) - np.array(d_lb))
    plt.ylabel("Difference between DeVille lb and our lb")
    plt.show()



