import numpy as np
import typing
import math
from tqdm import tqdm
import pandas as pd
import time
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar 
from piecewise_linear_maximize import create_piecewise_linear, binary_search_max

from lbs.gershgorin import gershgorin_lb
from utils import EPS, create_rand_psd_matrix

def abs_lb(M: np.ndarray, *args) -> float:
    v = args[0] if len(args) > 0 else []  # Use as default 2
    
    if len(v) < len(M):
         v = (np.random.rand(len(M)) - 0.5) * 2

    if type(v) != np.ndarray:
         v = np.array(v)

    repetitions = 10 * int(np.ceil(np.log(len(M))))
    # repetitions = 1000
    n = len(M)
    
    best_lb = max(gershgorin_lb(M), gershgorin_lb(M-np.outer(v,v)))

    lbs_abs = []

    for k in range(repetitions):
        for i in range(n):
                dd_i =  M[i,i] - v[i]**2 - np.sum([np.abs(M[i,j]-v[i]*v[j]) for j in range(n) if j != i])
                if dd_i < 0:
                    m_i = M[i, :].copy()
                    m_i[i] = 0
                    v_i = v.copy()
                    v_i[i]=0

                    pl = create_piecewise_linear(m_i, v_i, i)
                    x = binary_search_max(pl)
                    v_i[i]=x
                    if (v_i[i])**2 > M[i,i]:
                         v_i[i] = M[i,i]
                    v = v_i

        lbs_abs.append(gershgorin_lb(M - np.outer(v,v)))
        
        new_lb = max(best_lb, lbs_abs[-1])
        if new_lb > best_lb:
            best_lb = new_lb
    
    return best_lb