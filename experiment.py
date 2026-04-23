import numpy as np
from tqdm import tqdm
import pandas as pd
import time
import matplotlib.pyplot as plt


from lbs.deville import deville_lb, brauers_lb
from lbs.eig import eig_lb
from lbs.gershgorin import gershgorin_lb
from lbs.greedy_pm_shift import max_direction_lb
from lbs.random_shift import random_lb
from lbs.avg_direction import avg_direction_lb
from lbs.avg_direction_v2 import avg_direction_v2_lb
from lbs.sos_lb import sos_lb
from lbs.abs_lb import abs_lb
from utils import create_rand_symmetric_matrix, create_rand_dd_plus_ros, create_rand_psd_matrix, EPS

np.set_printoptions(precision=2, suppress=True)

def run_experiment(lb_functions, n, attempts, sparsity, diag_eps, rank, rangeval, sortby):
    df_result = pd.DataFrame(columns=[lb_f[1] for lb_f in lb_functions] + [lb_f[1] + "_time" for lb_f in lb_functions])

    for att in tqdm(range(attempts)):
        M = create_rand_psd_matrix(
            n=n, 
            sparsity=sparsity, 
            diag_eps=diag_eps, 
            rank=rank, 
            rangeval=rangeval)
    
        if att == 0:
            print("n=", n)
            print(M)
        row = {}
        for lb_f, lb_name, args in lb_functions:
            start = time.time()
            row[lb_name] = lb_f(M, args)
            end = time.time() - start
            row[lb_name+"_time"] = end
        df_result.loc[len(df_result)] = row
    print(df_result.describe())

    df_result = df_result.sort_values(by=sortby)  
    return df_result  


if __name__ == "__main__":
    
    np.random.seed(42)
    n = 5000
    attempts = 1
    
    lb_functions = [
        (gershgorin_lb, "Gershgorin", ()),
        (deville_lb, "DeVille", ()),
        (avg_direction_v2_lb, "Algorithm 2(k=1)", (1)),
        (avg_direction_v2_lb, "Algorithm 2(k=4)", (4)),
        # (sos_lb, "sos", ()),
        # (abs_lb, "abs", ()),
        (eig_lb, "eigenvalue", ()),
    ]
    
    df_result = run_experiment(
        lb_functions=lb_functions, 
        n=n,
        attempts=attempts, 
        sparsity=0,
        diag_eps=0.5,
        rank=1,
        rangeval=(-1,1))
