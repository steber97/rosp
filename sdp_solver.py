import numpy as np
from scipy.linalg import eigh

def psd_projection(W, eps=1e-8):
    # Full eigen-decomposition
    eigvals, eigvecs = eigh(W)
    eigvals_clipped = np.maximum(eigvals, 0.0)
    return eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T

def psd_projection_heuristic(W):
    # Example cheap test: Gershgorin lower bound
    diag = np.diag(W)
    off_diag_sum = np.sum(np.abs(W), axis=1) - np.abs(diag)

    if np.all(diag - off_diag_sum >= -1e-6):
        # Likely PSD, skip eigendecomposition
        return W
    else:
        return psd_projection(W)
    
def maxcut_sdp_admm(W, rho=1.0, max_iter=1000, tol=1e-6):
    """
    We are trying to optimize:
    max <W,X>
    s.t. diag(X) = 1
         X >= 0 (psd)
    
    :param W: Description
    :param rho: Description
    :param max_iter: Description
    :param tol: Description
    """
    W_copy = -W.copy()
    n = W_copy.shape[0]

    # Variables
    X = np.eye(n)
    Z = np.eye(n)
    Y = np.zeros((n, n))  # dual variable

    for k in range(max_iter):
        # ----- X-update (affine + diag constraint) -----
        X = Z - Y + (1 / rho) * W_copy
        np.fill_diagonal(X, 1.0)

        # ----- Z-update (PSD projection) -----
        Z_prev = Z.copy()
        Z = psd_projection_heuristic(X + Y)

        # ----- Dual update -----
        Y += X - Z

        # ----- Convergence check -----
        primal_res = np.linalg.norm(X - Z, 'fro')
        dual_res = np.linalg.norm(Z - Z_prev, 'fro')

        if primal_res < tol and dual_res < tol:
            print(f"Converged in {k} iterations")
            break

    return Z

def factor_psd_eig(X, tol=1e-8):
    # Eigen-decomposition
    eigvals, eigvecs = np.linalg.eigh(X)

    # Keep only nonzero eigenvalues
    idx = eigvals > tol
    sqrt_vals = np.sqrt(eigvals[idx])

    # B has shape (r, n)
    B = np.diag(sqrt_vals) @ eigvecs[:, idx].T
    return B

def max_cut(W):
    # IMPORTANT: we are minimizing -W!!
    X = maxcut_sdp_admm(W, rho=1, max_iter=10**4, tol=1e-5)
    B = factor_psd_eig(X)
    # print(B)
    # TODO: here I would need to recover the node embedding, and draw the random hyperplane.
    return B

if __name__=="__main__":
    np.random.seed(42)

    n = 50

    # Random symmetric weights (spin-glass style)
    W = np.random.randint(0, n, size=(n, n))
    W = (W + W.T) / 2
    np.fill_diagonal(W, 0.0)

    # # Add low-rank structure to create frustration
    # k = 3
    # U = np.random.randn(n, k)
    # W += 0.8 * (U @ U.T)

    # print(W[:10, :10])
    max_cut(W)