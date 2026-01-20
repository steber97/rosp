import numpy as np

from utils import EPS, create_rand_dd_plus_ros
from gershgorin import gershgorin_lb
from piecewise_linear_maximize import maximize_x
from greedy_pm_shift import max_direction_lb

def rand_cluster_lb(M: np.ndarray, *args) -> float:
    """
    args: number of repetitions. Needs to be a constant. Defaults to 2
    """
    n = len(M)
    rep = int(args[0]) if len(args) > 0 else 2  # Use as default 2
    dd_value = [(
        i, 
        M[i,i] - np.sum(np.abs(np.concatenate((M[i, :i], [] if i == n-1 else M[i, i+1:]))))
        ) for i in range(n)]
    dd_value = sorted(dd_value, key=lambda x: x[1])
    
    for k in range(rep):
        if dd_value[k][1] >= -EPS:
            rep = k
            break
    directions = []
    for k in range(rep):
        directions.append(np.ones(n))
        i, dd_val = dd_value[k]
        for j in range(n):
            if j != i:
                directions[-1][j] = directions[-1][j] * (1 if M[i,j] >= -EPS else -1)  # Here we would like to have zeros :/
        directions[-1] /= np.sqrt(directions[-1] @ directions[-1])
        if k > 0:
            if directions[-1] @ directions[0] >= EPS:
                directions[0][i] = np.abs(directions[0][i])
            else:
                directions[-1] = -directions[-1]
                directions[0][i] = - np.abs(directions[0][i])

    tot = np.sum([dd_value[k][1] for k in range(rep)])
    direction = np.sum([directions[k] * dd_value[k][1] / tot for k in range(rep)], axis=0)
    x = maximize_x(M, np.outer(direction, direction))

    return gershgorin_lb(M - x * np.outer(direction, direction))


if __name__ == "__main__":
    M = create_rand_dd_plus_ros(10, 0.3)
    print(M)
    rand_cluster_lb(M)