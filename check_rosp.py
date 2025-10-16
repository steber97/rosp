import numpy as np
import itertools
from tqdm import tqdm

from utils import create_rand_matrix, zero_one_with_probability, \
        diagonally_dominant, psd, is_vect_eigenv

if __name__ == "__main__":
    tot = 0
    tot_shift_by_row = 0
    repetitions = 100
    psd_tot = 0
    tot_eigv = 0
    n = 15
    prob = 0.5
    print_rosp = False
    print_unrosp = False
    for i in tqdm(range(repetitions)):
        M = create_rand_matrix(n, prob)
        psd_tot += psd(M)
        vectors = itertools.product([-1,1], repeat=n)
        rosp = False
        shift_by_row = False
        v_shift = None
        for v in vectors:
            v = np.array(v)
            S = np.outer(v, v)
            if diagonally_dominant(M-S):
                rosp = True
                v_shift = v
                if is_vect_eigenv(M, v):
                    tot_eigv += 1
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
    print(
        f"total rosp:\t{tot}\ntot repetitions:\t{repetitions}\ntot ",
        f"psd:\t{psd_tot}\ntot shifted by row:\t{tot_shift_by_row}\n",
        f"tot eigenvect:\t{tot_eigv}")
        
