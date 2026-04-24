import numpy as np
from tqdm import tqdm
import pandas as pd
import time
import matplotlib.pyplot as plt

from lbs.deville import deville_lb, brauers_lb
from lbs.eig import eig_lb
from lbs.gershgorin import gershgorin_lb
from lbs.avg_direction_v2 import avg_direction_v2_lb
from utils import create_rand_symmetric_matrix, create_rand_dd_plus_ros, create_rand_psd_matrix, EPS

np.set_printoptions(precision=2, suppress=True)

def experiment1():
    from experiment import run_experiment # for circular import
    print("Experiment 1")
    np.random.seed(42)
    n = 10
    attempts = 100
    
    lb_functions = [
        (gershgorin_lb, "Gershgorin", ()),
        (deville_lb, "DeVille", ()),
        (avg_direction_v2_lb, "Algorithm 2(k=1)", (1)),
        (avg_direction_v2_lb, "Algorithm 2(k=2)", (2)),
        (eig_lb, "eigenvalue", ()), 
    ]
    diag_eps_v = [0, 0.1, 0.5, 1]
    fig, axes = fig, axs = plt.subplots(2, 2, figsize=(20, 10))

    df = pd.DataFrame()

    for i, diag_eps in enumerate(diag_eps_v):

        df_result = run_experiment(
            lb_functions=lb_functions, 
            n=n,
            attempts=attempts, 
            sparsity=0,
            diag_eps=diag_eps,
            rank=n,
            rangeval=(-1,1),
            sortby="Algorithm 2(k=2)")
        df_result['eps'] = [diag_eps for i in range(len(df_result))]
        df = pd.concat([df_result, df], ignore_index=True)
        for lb_f, lb_name, args in lb_functions:
            j = 0
            # The shift just helps visualizing overlapping scatters.
            shift = 0.1
            if lb_name=="Gershgorin":
                j += shift
            elif lb_name=="DeVille":
                j -= shift
            elif lb_name=="Algorithm 2(k=1)":
                j += shift
            elif lb_name=="Algorithm 2(k=2)":
                j -= shift
            axes[i//2][i%2].scatter(
                [i+j for i in range(len(df_result))],
                df_result[lb_name], label=lb_name
            )
        
        axes[i//2][i%2].set_title("eps={}".format(diag_eps))
        axes[i//2][i%2].plot(
            [i for i in range(len(df_result))],
            [0 for i in range(len(df_result))], label='zero')
        # axes[i//2][i%2].legend()
    
        # df_result[[col for col in df_result.columns if "time" in col]].boxplot(ax=axs[1])

        df_lbs = df_result[[col for col in df_result.columns if "time" not in col and "eps" not in col]]
        print("diag_eps={}\ncorrect=\n{}".format(diag_eps, (df_lbs>0-EPS).sum(axis=0)))
    fig.supxlabel('Attempt', fontsize=12)
    fig.supylabel('LB on smallest eigenvalue', fontsize=12)
    plt.legend()
    plt.savefig('figures/experiment1/experiment1.png', dpi=1600)
    df.to_csv("figures/experiment1/experiment1.csv")

    print("Experiment 1 finished!")