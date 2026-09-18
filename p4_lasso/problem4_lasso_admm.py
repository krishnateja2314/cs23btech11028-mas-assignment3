"""
Problem 4: LASSO via ADMM.

Problem:
    min_x  (1/2) || A x - b ||_2^2 + lambda || x ||_1

References:
- Helper notes, pages 5, 7, 11-14  (soft thresholding proximal, ISTA,
  variable splitting, augmented Lagrangian, scaled ADMM).
- ADMM Lecture, pages 4-6  (augmented Lagrangian and update equations).

Variable splitting (Helper page 11):  introduce z, rewrite as
    min f(x) + g(z)   s.t.  x - z = 0
    f(x) = (1/2) || A x - b ||^2 ,   g(z) = lambda || z ||_1

Scaled-form ADMM (Helper page 13):
    x^{k+1} = argmin  f(x) + (rho/2) || x - z^k + u^k ||^2
    z^{k+1} = argmin  g(z) + (rho/2) || x^{k+1} - z + u^k ||^2
    u^{k+1} = u^k + (x^{k+1} - z^{k+1})

Closed forms:
    x^{k+1} = ( A^T A + rho I )^{-1} ( A^T b + rho (z^k - u^k) )
    z^{k+1} = S_{lambda/rho} ( x^{k+1} + u^k )
where S_kappa(v)_i = sign(v_i) max(|v_i| - kappa, 0) is soft thresholding
(Helper page 5).

Author: Krishna Teja (CS23BTECH11028)
"""

import os
import numpy as np
import matplotlib.pyplot as plt

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

def soft_thresh(v, kappa):
    return np.sign(v) * np.maximum(np.abs(v) - kappa, 0.0)

def lasso_admm(A, b, lam, rho=1.0, iters=500, tol=1e-6):
    m, n = A.shape
    AtA = A.T @ A
    Atb = A.T @ b
    L = np.linalg.cholesky(AtA + rho * np.eye(n))
    x = np.zeros(n); z = np.zeros(n); u = np.zeros(n)
    hist = {'obj': [], 'r': [], 's': []}
    for k in range(iters):
        # x-update (Cholesky-based solve of the normal equation)
        rhs = Atb + rho * (z - u)
        y   = np.linalg.solve(L, rhs)
        x   = np.linalg.solve(L.T, y)
        # z-update  (soft thresholding: prox of ell_1)
        z_old = z.copy()
        z = soft_thresh(x + u, lam / rho)
        # dual update
        u = u + (x - z)
        # residuals + objective
        r = float(np.linalg.norm(x - z))
        s = float(rho * np.linalg.norm(z - z_old))
        obj = 0.5*float(np.linalg.norm(A @ x - b)**2) + lam*float(np.abs(z).sum())
        hist['obj'].append(obj); hist['r'].append(r); hist['s'].append(s)
        if r < tol and s < tol:
            print(f"converged at iter {k}")
            break
    return x, z, hist

def run():
    rng = np.random.default_rng(0)
    m, n = 200, 500
    A = rng.standard_normal((m, n))
    # normalize columns (standard preprocessing for LASSO)
    A = A / np.linalg.norm(A, axis=0, keepdims=True)
    x_true = np.zeros(n)
    k_nonzero = 20
    idx = rng.choice(n, k_nonzero, replace=False)
    x_true[idx] = rng.standard_normal(k_nonzero)
    b = A @ x_true + 0.01 * rng.standard_normal(m)
    # standard heuristic: lambda ~ small fraction of ||A^T b||_inf
    lam = 0.02 * float(np.max(np.abs(A.T @ b)))
    print(f"m={m}, n={n}, k_true nnz={k_nonzero}, lambda={lam:.4f}")

    x_out, z_out, hist = lasso_admm(A, b, lam, rho=1.0, iters=500)

    nnz = int(np.sum(np.abs(z_out) > 1e-6))
    # support overlap
    true_supp = set(idx.tolist())
    est_supp  = set(np.where(np.abs(z_out) > 1e-6)[0].tolist())
    tp = len(true_supp & est_supp)
    fp = len(est_supp - true_supp)
    fn = len(true_supp - est_supp)
    rec_err = float(np.linalg.norm(z_out - x_true) / max(np.linalg.norm(x_true), 1e-12))
    print(f"non-zeros in z = {nnz}, true support size = {k_nonzero}")
    print(f"support: true positives = {tp}, false positives = {fp}, "
          f"false negatives = {fn}")
    print(f"relative recovery error ||z - x_true|| / ||x_true|| = {rec_err:.4f}")

    # -------- figure : true vs recovered, objective, residuals --------
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    axes[0, 0].stem(x_true, markerfmt='bo', linefmt='b-', basefmt=' ')
    axes[0, 0].set_title("True sparse x  (k = 20 non-zeros)")
    axes[0, 0].set_xlabel("index"); axes[0, 0].set_ylabel("value")
    axes[0, 1].stem(z_out, markerfmt='rx', linefmt='r-', basefmt=' ')
    axes[0, 1].set_title(f"LASSO estimate z  ({nnz} non-zeros)")
    axes[0, 1].set_xlabel("index"); axes[0, 1].set_ylabel("value")
    axes[1, 0].semilogy(hist['obj'])
    axes[1, 0].set_title("Objective vs iteration")
    axes[1, 0].set_xlabel("iter"); axes[1, 0].set_ylabel("(1/2)||Ax-b||^2 + lam ||z||_1")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 1].semilogy(hist['r'], label='primal ||x-z||')
    axes[1, 1].semilogy(hist['s'], label='dual rho ||z_k - z_{k-1}||')
    axes[1, 1].set_title("Primal and dual residuals")
    axes[1, 1].set_xlabel("iter"); axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "lasso.png"), dpi=150)
    plt.close(fig)

    # ---- overlay: true vs estimate on the same axes ----
    fig2, ax = plt.subplots(figsize=(9, 3.2))
    ax.stem(np.arange(n), x_true, markerfmt='bo', linefmt='b-', basefmt=' ',
            label='true')
    ax.stem(np.arange(n), z_out, markerfmt='rx', linefmt='r--', basefmt=' ',
            label='ADMM estimate')
    ax.set_title("Overlay: true x versus ADMM LASSO estimate z")
    ax.set_xlabel("index"); ax.set_ylabel("value")
    ax.legend()
    fig2.tight_layout()
    fig2.savefig(os.path.join(OUT_DIR, "lasso_overlay.png"), dpi=150)
    plt.close(fig2)

if __name__ == "__main__":
    run()
