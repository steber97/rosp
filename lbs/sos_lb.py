import numpy as np
import typing
import math
from tqdm import tqdm
import pandas as pd
import time
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar 

from lbs.gershgorin import gershgorin_lb
from utils import EPS, create_rand_psd_matrix

def sos_lb(M: np.ndarray, *args) -> float:
    repetitions = 1000
    n = len(M)
    v = (np.random.rand(n) - 0.5) * 2
    v_abs = np.copy(v)
    best_v = v.copy()
    best_lb = -math.inf

    lr = 0.1
    lbs_sos = []
    lbs_abs = []

    for k in range(repetitions):
        M_v = M - np.outer(v, v)
        dd_values = [(
            i, 
            M_v[i,i] - np.sum(
                np.abs(np.concatenate((M_v[i, :i], [] if i == n-1 else M_v[i, i+1:]))))
            ) for i in range(n)]
        dd_values = sorted(dd_values, key=lambda x: x[1])
        # print(dd_values)
        v_new = v.copy()

        for i in range(n):
            if dd_values[i][1] < 0:
                idx = dd_values[i][0]
                dd_val = dd_values[i][1]
                N = np.sum([M[idx][j] * v[j] for j in range(n) if j != idx])
                # Just to avoid division by zero.
                D = np.sum([v[j]**2 for j in range(n) if j != idx]) + EPS 
                v_new[idx] = (lr * (dd_val/dd_values[0][1])) * (N / D) +\
                      (1-(lr * ((dd_val/dd_values[0][1])))) * (v[idx])
                # print(dd_val/dd_values[0][1], v[idx], N/D, v_new[idx])
        v = v_new
        lbs_sos.append(gershgorin_lb(M - np.outer(v,v)))
        

        M_v = M - np.outer(v_abs, v_abs)
        dd_values = [(
            i, 
            M_v[i,i] - np.sum(
                np.abs(np.concatenate((M_v[i, :i], [] if i == n-1 else M_v[i, i+1:]))))
            ) for i in range(n)]
        dd_values = sorted(dd_values, key=lambda x: x[1])
        # print(dd_values)
        v_new = v_abs.copy()
        for i in range(n):
            if dd_values[i][1] < 0:
                def f(x):
                    return M[i,i] - x**2 - np.sum(
                        [np.abs(M[i,j] - v_abs[j] * x) for j in range(n) if i != j])
                res = minimize_scalar(lambda x: -f(x), method='brent')
                newval = res.x
                idx = dd_values[i][0]
                dd_val = dd_values[i][1]
                v_new[idx] = (lr * (dd_val/dd_values[0][1])) * (newval) +\
                    (1-(lr * ((dd_val/dd_values[0][1])))) * (v_abs[idx])
                # print(dd_val/dd_values[0][1], v[idx], N/D, v_new[idx])
        v_abs = v_new
        lbs_abs.append(gershgorin_lb(M - np.outer(v_abs,v_abs)))
        new_lb = max(best_lb, lbs_sos[-1])
        if new_lb > best_lb:
            best_v = v.copy()
            best_lb = new_lb
    print(np.max(lbs_sos), np.max(lbs_abs))
    plt.plot([i for i in range(len(lbs_sos))], lbs_sos, label='sos')
    plt.plot([i for i in range(len(lbs_abs))], lbs_abs, label='abs')
    plt.legend()
    plt.show()

    return best_lb

if __name__ == '__main__':
    attempts = 1
    n = 15
    df_result = pd.DataFrame(columns=['sos'] + ["sos_time"])
    for att in tqdm(range(attempts)):
        M = create_rand_psd_matrix(n)
        row = {}
        start = time.time()
        row['sos'] = sos_lb(M)
        end = time.time() - start
        row["sos_time"] = end
        
        df_result.loc[len(df_result)] = row