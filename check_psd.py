import numpy as np

from utils import create_rand_matrix

# This code runs for a while, and testifies that the vast majority 
# of random matrices filled with +-1, and n-2 on the diagonal are 
# indeed psd.

if __name__ == "__main__":
    failed = 0
    for prob in range(1,10):
        for i in range(20, 1200, 100):
            for j in range(10):
                if np.linalg.eig(create_rand_matrix(i, 1/prob))[0].min() < 0:
                    failed += 1
                    print(i)
    print(failed)