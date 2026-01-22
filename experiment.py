import numpy as np
from tqdm import tqdm
import pandas as pd
import time
import matplotlib.pyplot as plt


from deville import deville_lb, brauers_lb
from eig import eig_lb
from gershgorin import gershgorin_lb
from greedy_pm_shift import max_direction_lb
from random_shift import random_lb
from avg_direction import avg_direction_lb
from avg_direction_v2 import avg_direction_v2_lb
from utils import create_rand_symmetric_matrix, create_rand_dd_plus_ros

np.set_printoptions(precision=2, suppress=True)

if __name__ == "__main__":
    
    np.random.seed(42)
    n = 15
    attempts = 10
    range_values = (0,11)  # inclusive, exclusive
    sign_perc = 0.5
    diag_boost = 20
    
    lb_functions = [
        (gershgorin_lb, "gershgorin", ()),
        (deville_lb, "deville", ()),
        (brauers_lb, "brauers", ()),
        (random_lb, "random", ()),
        (max_direction_lb, "greedy", (1)),
        (avg_direction_lb, "avg_direction_15", (15)),
        (avg_direction_v2_lb, "avg_direction_v2", ()),
        (eig_lb, "eigenvalue", ()),
    ]
    df_result = pd.DataFrame(columns=[lb_f[1] for lb_f in lb_functions] + [lb_f[1] + "_time" for lb_f in lb_functions])

    for att in tqdm(range(attempts)):
        M = create_rand_dd_plus_ros(n, 0.1)
        row = {}
        for lb_f, lb_name, args in lb_functions:
            start = time.time()
            row[lb_name] = lb_f(M, args)
            end = time.time() - start
            row[lb_name+"_time"] = end
        df_result.loc[len(df_result)] = row
    print(df_result.describe())

    #plt.boxplot(np.array(df_result['greedy']) - np.array(df_result['deville']))
    # plt.ylabel("Difference between DeVille lb and our lb")

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 9))

    ax1.boxplot(df_result[[lb_name + '_time' for lb_f, lb_name, args in lb_functions]], 
                labels=[lb_name for lb_f, lb_name, args in lb_functions])
    ax1.set_ylabel("time")

    ax2.boxplot(df_result[[lb_name for lb_f, lb_name, args in lb_functions]], 
                labels=[lb_name for lb_f, lb_name, args in lb_functions])
    ax2.set_ylabel("LB")
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha="right")
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha="right")
    plt.show()