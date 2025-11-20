import numpy as np
import matplotlib.pyplot as plt

def delta_i_k(M: np.array, i: int, k: int) -> float:
    """
    Def 3.3 of paper.
    Careful because all indices are shifted by one.
    """
    n = len(M)
    assert 0 <= i < n and 0 <= k < n
    y = sorted([M[i,j] for j in range(n) if i != j])
    return np.sum(y[:k]) - np.sum(y[k:])

def s_i_k(M: np.array, i: int, k: int) -> float:
    """
    Def 3.3 of paper
    Careful because all indices are shifted by one.
    """
    n = len(M)
    assert 0 <= i < n and 0 <= k < n
    return M[i,i] + delta_i_k(M, i, k)

def s_k(M: np.array, k: int) -> float:
    """
    Def 3.3 of paper
    Careful because all indices are shifted by one.
    """
    n = len(M)
    assert 0 <= k < n
    return np.min([s_i_k(M, i, k) for i in range(n)])

def compute_x(M: np.array) -> float:
    """
    Corollary 3.5 of paper
    """
    n = len(M)
    Q = [(k, l) for k in range(n) for l in range(n) if k < n//2 and l >= n//2 and s_k(M, l) >= s_k(M, k)]
    if len(Q) == 0:
        # No way to improve it.
        return 0.0
    min_l_k = np.argmin([((n//2 - 1 - k)*s_k(M, l) + (l - (n//2 - 1)))/(l-k) for (k, l) in Q])
    l, k = Q[min_l_k]
    x = (s_k(M, l) - s_k(M, k))/(2*(l-k))
    return x

def gershgorin_lb(M: np.array) -> float:
    n = len(M)
    return np.min([M[i,i] - np.sum(np.abs(np.concatenate([M[i,:i], M[i,i+1:]]))) for i in range(n)])

def R_i(M: np.array, i: int) -> float:
    """
    Simple sum of off-diagonal elements
    """
    return np.sum(np.abs(np.concatenate([M[i,:i], M[i, i+1:]])))

def brauers_lb(M: np.array) -> float:
    n = len(M)
    return np.min([((M[i,i] + M[j,j]) / 2) - (np.sqrt((M[i,i] - M[j,j])**2 + R_i(M, i) * R_i(M, j))) for i in range(n) for j in range(n) if i != j])

def shifted_gershgorin_lb(M: np.array) -> float:
    n = len(M)
    x = compute_x(M)
    if x < -1e-5:
        print(x, M)
    assert x > -1e-5
    SM = M - x * np.ones_like(M)
    return gershgorin_lb(SM)


if __name__ == "__main__":
    n = 5
    attempts = 1000
    g_lb = []
    sg_lb = []
    b_lb = []
    eigvals = []
    values = []
    for att in range(attempts):
        range_values = [0,11]  # inclusive, exclusive
        sign_perc = 0
        M = np.random.randint(np.ones(n**2) * range_values[0], np.ones(n**2) * range_values[1]).reshape(n, n)
        # Make it symmetric.
        for i in range(n):
            for j in range(i):
                if np.random.rand() < sign_perc:
                    M[j,i] = -M[j,i]
                M[i,j] = M[j,i]
        if att == 0:
            print(M)
        
        g_lb.append(gershgorin_lb(M))
        sg_lb.append(shifted_gershgorin_lb(M))
        b_lb.append(brauers_lb(M))
        eigvals.append(np.linalg.eig(M)[0].min())
        values.append((g_lb[-1], sg_lb[-1], b_lb[-1], eigvals[-1]))
    
    values = sorted(values, key=lambda x: x[2])

    legend_names = ['gersh', 'sgers', 'brauer', 'eigv']
    for j in range(4):
        plt.plot([i for i in range(attempts)], [values[i][j] for i in range(attempts)], label=legend_names[j])
        plt.ylabel("a")
    plt.legend()
    plt.show()




