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
    v = args[0] if len(args) > 0 else []  # Use as default 2
    
    if len(v) < len(M):
         v = (np.random.rand(len(M)) - 0.5) * 2
    repetitions = 10 * int(np.ceil(np.log(len(M))))
    # repetitions = 1000
    n = len(M)
    # v = (np.random.rand(n) - 0.5) * 2
    best_lb = max(gershgorin_lb(M), gershgorin_lb(M-np.outer(v,v)))

    lr = 0.1
    lbs_sos = []
    lbs_abs = []

    for k in range(repetitions):
        for i in range(n):
                dd_i =  M[i,i] - v[i]**2 - np.sum([np.abs(M[i,j]-v[i]*v[j]) for j in range(n) if j != i])
                if dd_i < 0:
                    N = np.sum([M[i][j] * v[j] for j in range(n) if j != i])
                    # Just to avoid division by zero.
                    D = np.sum([v[j]**2 for j in range(n) if j != i])
                    if D < EPS:
                         D = EPS
                    v[i] = min(N/D, np.sqrt(M[i,i]))
        lbs_sos.append(gershgorin_lb(M - np.outer(v,v)))
        
        new_lb = max(best_lb, lbs_sos[-1])
        if new_lb > best_lb:
            best_lb = new_lb
    
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