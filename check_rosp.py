import numpy as np
import itertools
from tqdm import tqdm

from utils import create_rand_matrix, zero_one_with_probability, diagonally_dominant, psd

if __name__ == "__main__":
    tot = 0
    tot_shift_by_row = 0
    repetitions = 100
    psd_tot = 0
    n = 15
    print_rosp = False
    print_unrosp = False
    for i in tqdm(range(repetitions)):
        M = create_rand_matrix(n, 0.5)
        psd_tot += psd(M)
        vectors = itertools.product([0,1], repeat=n)
        rosp = False
        shift_by_row = False
        v_shift = None
        for v in vectors:
            v = np.array([-1 if x==0 else 1 for x in v])
            S = np.outer(v, v)
            if diagonally_dominant(M-S):
                rosp = True
                v_shift = v
                break
        if not print_unrosp and not rosp:
            print("Rosp?", rosp)
            print(M)
            print_unrosp = True
        for i in range(n):
            v = np.array([x for x in M[i,:]])
            v[i] = 1
            S = np.outer(v, v)
            if diagonally_dominant(M-S):
                shift_by_row = True
                break
        if not print_rosp and rosp and not shift_by_row:
            print("Rosp?", rosp)
            print(M)
            print(v_shift)
            print_rosp = True
        tot_shift_by_row += shift_by_row
        tot += 1 if rosp else 0
    print(tot, repetitions, psd_tot, tot_shift_by_row)
        
