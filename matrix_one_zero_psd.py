import numpy as np

from typing import List, Tuple

from utils import psd

def build_zero_one_mat_with_pairs(n: int, pairs: List[Tuple[int, int]]) -> np.array:
    M = np.ones((n, n))
    for pair in pairs:
        M[pair[0], pair[1]] = 0
        M[pair[1], pair[0]] = 0
    return M

def compute_zero_one_mat_recursively(n: int, pairs_so_far: List[Tuple[int, int]], remaining_elements: List[int]) -> Tuple[int,int]:
    """
    Given n, the size of the matrix of zeros and ones, compute how many symmetric matrices
    with at most one zero per row (different from diagonal) are psd.
    @arg pairs_so_far
    @arg remaining_elements: sorted in increasing order

    @returns psd matrices, total matrices
    """
    counter_psd = 0
    counter_total = 0
    if len(remaining_elements) < 2:
        return 0, 0
    i = 0
    for j in range(1, len(remaining_elements)):
        pair = (remaining_elements[i], remaining_elements[j])
        rem_el = [x for x in remaining_elements if x != remaining_elements[i] and x != remaining_elements[j]]
        pair_sf = pairs_so_far + [pair]
        # print(pair, pairs_so_far, remaining_elements, pair_sf, rem_el)
        M = build_zero_one_mat_with_pairs(n, pair_sf)
        counter_total += 1
        if psd(M):
            counter_psd += 1
        psd_rec, tot_rec = compute_zero_one_mat_recursively(n, pair_sf, rem_el)
        counter_psd += psd_rec
        counter_total += tot_rec
    return counter_psd, counter_total


if __name__ == "__main__":
    n = 14
    psd_counter, tot_counter = compute_zero_one_mat_recursively(n, [], [x for x in range(n)])
    print("psd matrices {} out of {}.".format(psd_counter, tot_counter))