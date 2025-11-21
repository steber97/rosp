import numpy as np

from utils import EPS

from deville import maximize_gershgoryn_circle, gershgorin_lb, deville_lb, create_rand_symmetric_matrix


def shift_as_max_direction(M: np.array) -> float:
    n = len(M)
    dd = np.array([M[i,i] - np.sum(np.abs(np.concatenate([M[i, :i], M[i, i+1:]]))) for i in range(n)])
    min_dd = np.min(dd)
    best_shift = np.ones(n)
    best_lb = deville_lb(M)
    best_x = maximize_gershgoryn_circle(M, best_shift)
    for i in range(n):
        if np.abs(dd[i] - min_dd) < EPS:
            shift_v = np.ones_like(M[i])
            for j in range(n):
                if j != i:
                    shift_v[j] = shift_v[j] * (1 if M[i,j] > EPS else -1 if M[i,j] < -EPS else 0)
            x, res = maximize_gershgoryn_circle(M, shift_v)
            # print(x, res, i)
            if res > best_lb:
                best_x = x
                best_lb = res
                best_shift = shift_v
    # print(best_shift, best_x, best_lb)
    return best_lb


        


if __name__=='__main__':
    n = 5
    range_vals = [0, 11]
    sign_perc = 0.5
    M = create_rand_symmetric_matrix(n, range_vals, sign_perc)
    print(M)
    print(shift_as_max_direction(M))
    print(deville_lb(M))