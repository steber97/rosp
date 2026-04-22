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
    np.random.seed(42)
    n = 10
    attempts = 100
    
    lb_functions = [
        (gershgorin_lb, "Gershgorin", ()),
        (deville_lb, "DeVille", ()),
        (avg_direction_v2_lb, "Algorithm 2(k=1)", (1)),
        (avg_direction_v2_lb, "Algorithm 2(k=4)", (4)),
        # (sos_lb, "sos", ()),
        # (abs_lb, "abs", ()),
        (eig_lb, "eigenvalue", ()),
    ]
    diag_eps_v = [0, 0.1, 0.5, 1]
    fig, axes = fig, axs = plt.subplots(2, 2, figsize=(10, 4))

    for i, diag_eps in enumerate(diag_eps_v):

        df_result = run_experiment(
            lb_functions=lb_functions, 
            n=n,
            attempts=attempts, 
            sparsity=0,
            diag_eps=diag_eps,
            rank=n,
            rangeval=(-1,1))
    
        for lb_f, lb_name, args in lb_functions:
            axes[i//2][i%2].scatter(
                [i for i in range(len(df_result))],
                df_result[lb_name], label=lb_name
            )
        
        axes[i//2][i%2].set_title("eps={}".format(diag_eps))
        axes[i//2][i%2].plot(
            [i for i in range(len(df_result))],
            [0 for i in range(len(df_result))], label='zero')
        # axes[i//2][i%2].legend()
    
        # df_result[[col for col in df_result.columns if "time" in col]].boxplot(ax=axs[1])

        df_lbs = df_result[[col for col in df_result.columns if "time" not in col]]
        print("diag_eps={}, correct={}".format(diag_eps, (df_lbs>0-EPS).sum(axis=0)))
    fig.supxlabel('Attempt', fontsize=12)
    fig.supylabel('LB on smallest eigenvalue', fontsize=12)
    plt.legend()
    plt.show()