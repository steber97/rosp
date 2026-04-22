import numpy as np
from tqdm import tqdm
import pandas as pd
import time
import matplotlib.pyplot as plt
from pprint import pprint

from experiment import run_experiment
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

if __name__ == "__main__":
    
    np.random.seed(42)
    ns = [i*1000 for i in range(1,11)]
    
    lb_functions = [
        (gershgorin_lb, "Gershgorin", ()),
        (deville_lb, "DeVille", ()),
        (avg_direction_v2_lb, "Algorithm 2(k=1)", (1)),
        (avg_direction_v2_lb, "Algorithm 2(k=4)", (4)),
        # (sos_lb, "sos", ()),
        # (abs_lb, "abs", ()),
        (eig_lb, "eigenvalue", ()),
    ]
    
    rows = pd.DataFrame()
    for i, n in enumerate(ns):

        df_result = run_experiment(
            lb_functions=lb_functions, 
            n=n,
            attempts=1, 
            sparsity=0,
            diag_eps=2,
            rank=2,
            rangeval=(-1,1))
        
        df_result['n'] = n
        if len(rows) == 0:
            rows = df_result
        else:
            rows = pd.concat([rows, df_result], ignore_index=True)

    rows.to_csv("experiment3.csv")