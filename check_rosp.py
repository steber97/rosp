import numpy as np
import itertools
from tqdm import tqdm

from utils import create_rand_matrix, zero_one_with_probability, diagonally_dominant, psd

if __name__ == "__main__":
    tot = 0
    repetitions = 100
    psd_tot = 0
    n = 16
    for i in tqdm(range(repetitions)):
        M = create_rand_matrix(n, 0.5)
        psd_tot += psd(M)
        vectors = itertools.product([0,1], repeat=n)
        rosp = False
        for v in vectors:
            v = np.array([-1 if x==0 else 1 for x in v])
            S = np.outer(v, v)
            if diagonally_dominant(M-S):
                rosp = True
                break
        tot += 1 if rosp else 0
    print(tot, repetitions, psd_tot)
        
