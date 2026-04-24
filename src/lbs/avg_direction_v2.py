import numpy as np
from pprint import pprint
from typing import Tuple
from time import time

from utils import EPS, create_rand_psd_matrix
from lbs.gershgorin import gershgorin_lb
from piecewise_linear_maximize import maximize_x, argmax_x
import optimization_module


def avg_direction_v2_lb(M: np.ndarray, *args) -> float:
    """
    args: repetitions. Namely, we try to write M = sum_{i=1}^{rep} x_i x_i^T 
            as the sum of rank-repetitions vectors.
    """
    total_time_opt_x = 0
    n = len(M)
    repetitions = int(args[0]) if len(args) > 0 else 1  # Use as default 2
    M_copy = M.copy()
    total_direction = np.zeros(n)
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
                dir = np.ones(n)
                i, dd_val = dd_value[k]
                dir[i] = min(1, M_copy[i,i])
                for j in range(n):
                    if j != i:
                        dir[j] = M_copy[i,j] / (min(1, M_copy[i,i]))
                dir /= np.sqrt(dir @ dir)
                # assert abs(directions[-1]@directions[-1] - 1) < EPS
                if k > 0:
                    if dir @ directions[0] < EPS:
                        dir = -dir
                directions.append(dir)
            tot = np.sum([dd_value[k][1] for k in range(rep)])
            direction = np.sum([directions[k] * dd_value[k][1] / tot for k in range(rep)], axis=0)
            direction /= np.sqrt(direction @ direction)
            S = np.outer(direction, direction)
            start = time()
            x = optimization_module.maximize_x_cpp(M_copy, S)
            total_time_opt_x += time() - start
            # x2 = argmax_x(M_copy, S)
            # print(x, x2)
            M_copy -= x * S
    # print("time opt x:", total_time_opt_x)
    return gershgorin_lb(M_copy)

if __name__=="__main__":
    n = 500
    M = create_rand_psd_matrix(n)
    # print(M)
    print(gershgorin_lb(M))
    time_start = time()
    print(avg_direction_v2_lb(M))
    print("time total:", time() - time_start)


