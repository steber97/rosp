import numpy as np
from tqdm import tqdm
import pandas as pd
import time
import matplotlib.pyplot as plt

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
    print("Experiment 2")
    np.random.seed(42)
    ns = [i*1000 for i in range(1,11)]
    
    lb_functions = [
        (gershgorin_lb, "Gershgorin", ()),
        (deville_lb, "DeVille", ()),
        (avg_direction_v2_lb, "Algorithm 2(k=1)", (1)),
        (avg_direction_v2_lb, "Algorithm 2(k=2)", (2)),
        # (sos_lb, "sos", ()),
        # (abs_lb, "abs", ()),
        (eig_lb, "eigenvalue", ()),
    ]
    
    df = pd.DataFrame()
    for i, n in enumerate(ns):

        df_result = run_experiment(
            lb_functions=lb_functions, 
            n=n,
            attempts=1, 
            sparsity=0,
            diag_eps=1,
            rank=2,
            rangeval=(-1,1),
            sortby="Algorithm 2(k=2)")
        df_result['n'] = [n for i in range(len(df_result))]
        df = pd.concat([df_result, df], ignore_index=True)
    
    for lb, lb_name, args in lb_functions:
        times = [np.mean(df[df['n']==n][lb_name+"_time"]) for n in ns]
        plt.plot(ns, times, label=lb_name)
    plt.xlabel("n")
    plt.ylabel("Time (s)")
    plt.title("Running time for increasing n")
    plt.legend()
    plt.savefig("figures/experiment2/experiment2.png", dpi=1600)
    df.to_csv("figures/experiment2/experiment2.csv")
    print("Experiment 2 finished!")