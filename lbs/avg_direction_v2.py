import numpy as np
from typing import Tuple

from utils import EPS, create_rand_dd_plus_ros
from lbs.gershgorin import gershgorin_lb
from piecewise_linear_maximize import maximize_x
from lbs.greedy_pm_shift import max_direction_lb

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
        rep = int(np.sum([x[1] < EPS for x in dd_value]))
        directions = []
        if rep > 0:
            for k in range(rep):
                directions.append(np.ones(n))
                i, dd_val = dd_value[k]
                for j in range(n):
                    if j != i:
                        directions[-1][j] = M_copy[i,j]
                directions[-1] /= np.sqrt(directions[-1] @ directions[-1])
                assert abs(directions[-1]@directions[-1] - 1) < EPS
                if k > 0:
                    if directions[-1] @ directions[0] < EPS:
                        directions[-1] = -directions[-1]
            
            tot = np.sum([dd_value[k][1] for k in range(rep)])
            direction = np.sum([directions[k] * dd_value[k][1] / tot for k in range(rep)], axis=0)
            direction /= np.sqrt(direction @ direction)
            
            # print(direction, direction @ direction)
            x = maximize_x(M_copy, np.outer(direction, direction))
            # x = 1
            M_copy -= x * np.outer(direction, direction)

    return gershgorin_lb(M_copy)

if __name__=="__main__":
    n = 10
    M = create_rand_dd_plus_ros(n, 0.1)
    print(M)
    lb = avg_direction_v2_lb(M)
    print(lb)

