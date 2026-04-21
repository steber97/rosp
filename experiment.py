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
from utils import create_rand_symmetric_matrix, create_rand_dd_plus_ros, create_rand_psd_matrix, EPS

np.set_printoptions(precision=2, suppress=True)

if __name__ == "__main__":
    
    np.random.seed(42)
    n = 5
    attempts = 50
    
    lb_functions = [
        (gershgorin_lb, "Gershgorin", ()),
        (deville_lb, "DeVille", ()),
        # (brauers_lb, "brauers", ()),
        # (random_lb, "random", ()),
        # (max_direction_lb, "greedy", (1)),
        # (avg_direction_lb, "avg_direction_15", (n)),
        (avg_direction_v2_lb, "Algorithm 2(k=1)", (1)),
        # (avg_direction_v2_lb, "avg_direction_v2_rep2", (2)),
        (avg_direction_v2_lb, "Algorithm 2(k=3)", (3)),
        (sos_lb, "sos", ()),
        (eig_lb, "eigenvalue", ()),
    ]
    df_result = pd.DataFrame(columns=[lb_f[1] for lb_f in lb_functions] + [lb_f[1] + "_time" for lb_f in lb_functions])

    for att in tqdm(range(attempts)):
        M = create_rand_psd_matrix(n)
        if att == 0:
            print(M)
        row = {}
        for lb_f, lb_name, args in lb_functions:
            start = time.time()
            row[lb_name] = lb_f(M, args)
            end = time.time() - start
            row[lb_name+"_time"] = end
        df_result.loc[len(df_result)] = row
    print(df_result.describe())

    df_result = df_result.sort_values(by='Algorithm 2(k=1)')
    for lb_f, lb_name, args in lb_functions:
        plt.scatter(
            [i for i in range(len(df_result))],
            df_result[lb_name], label=lb_name
        )
    
    plt.plot(
        [i for i in range(len(df_result))],
        [0 for i in range(len(df_result))], label='zero')
    
    df_lbs = df_result[[col for col in df_result.columns if "time" not in col]]
    print((df_lbs>0).sum(axis=0))

    plt.legend()
    plt.show()
    with open("plot.tex", mode='w') as f:
        # Reset index so rows are 0,1,2,...
        df_lbs = df_lbs.reset_index(drop=True)

        print(r"\begin{tikzpicture}", file=f)
        print(r"\begin{axis}[", file=f)
        print(r"    width=12cm,", file=f)
        print(r"    height=7cm,", file=f)
        print(r"    xlabel={Attempt},", file=f)
        print(r"    ylabel={LB for diagonally dominance},", file=f)
        print(r"    grid=major,", file=f)
        print(r"    legend style={at={(1.05,1)}, anchor=north west},", file=f)
        print(r"]", file=f)

        # Styles for the 3 columns
        styles = [
            "only marks, mark=*,          mark size=2.5pt, color=blue,            fill=blue,            draw=white, line width=0.4pt",
            "only marks, mark=square*,    mark size=2.5pt, color=red,             fill=red,             draw=white, line width=0.4pt",
            "only marks, mark=triangle*,  mark size=3.0pt, color=green!60!black,  fill=green!60!black,  draw=white, line width=0.4pt",
            "only marks, mark=diamond*,   mark size=2.8pt, color=orange,          fill=orange,          draw=white, line width=0.4pt",
            "only marks, mark=pentagon*,  mark size=2.8pt, color=purple,          fill=purple,          draw=white, line width=0.4pt",
            "only marks, mark=x,          mark size=3.0pt, color=black,                                line width=0.8pt"
        ]

        for k, col in enumerate(df_lbs.columns):
            safe_col = str(col).replace("_", r"\_")
            style = styles[k % len(styles)]

            print(rf"\addplot[{style}] coordinates {{", file=f)
            for i, val in enumerate(df_lbs[col]):
                print(f"({i},{float(val):.6f})", file=f)
            print(r"};", file=f)
            print(rf"\addlegendentry{{{safe_col}}}", file=f)

        print(r"\end{axis}", file=f)
        print(r"\end{tikzpicture}", file=f)