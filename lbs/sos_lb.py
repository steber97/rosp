import numpy as np
import typing
import math
from tqdm import tqdm
import pandas as pd
import time
import matplotlib.pyplot as plt

from lbs.gershgorin import gershgorin_lb
from utils import EPS, create_rand_psd_matrix

def sos_lb(M: np.ndarray, *args) -> float:
    repetitions = 1000
    n = len(M)
    v = np.random.rand(n) - 0.5
    best_v = v.copy()
    best_lb = -math.inf

    lr = 0.1
    lbs = []
    for k in range(repetitions):
        dd_values = [(
            i, 
            M[i,i] - np.sum(np.abs(np.concatenate((M[i, :i], [] if i == n-1 else M[i, i+1:]))))
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
                v_new[idx] = (lr * (dd_val/dd_values[0][1])) * (N / D) + (1-(lr * ((dd_val/dd_values[0][1])))) * (v[idx])
                # print(dd_val/dd_values[0][1], v[idx], N/D, v_new[idx])
        v = v_new
        lbs.append(gershgorin_lb(M - np.outer(v,v)))
        new_lb = max(best_lb, lbs[-1])
        if new_lb > best_lb:
            best_v = v.copy()
            best_lb = new_lb
    # plt.plot([i for i in range(len(lbs))], lbs)
    # plt.show()
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