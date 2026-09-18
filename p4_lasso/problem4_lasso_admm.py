import os
import numpy as np
import matplotlib.pyplot as plt

OUT = os.path.dirname(os.path.abspath(__file__))


def soft(v, k):
    # soft thresholding, prox of ell_1
    return np.sign(v) * np.maximum(np.abs(v) - k, 0.0)


def lasso_admm(A, b, lam, rho=1.0, iters=500, tol=1e-6):
    m, n = A.shape
    AtA = A.T @ A
    Atb = A.T @ b
    L = np.linalg.cholesky(AtA + rho*np.eye(n))
    x = np.zeros(n); z = np.zeros(n); u = np.zeros(n)
    hist = {'obj': [], 'r': [], 's': []}

    for k in range(iters):
        # x update, closed form via cached Cholesky
        rhs = Atb + rho*(z - u)
        y = np.linalg.solve(L, rhs)
        x = np.linalg.solve(L.T, y)

        # z update
        z_old = z.copy()
        z = soft(x + u, lam/rho)

        # dual update
        u = u + (x - z)

        r = float(np.linalg.norm(x - z))
        s = float(rho * np.linalg.norm(z - z_old))
        obj = 0.5*float(np.linalg.norm(A @ x - b)**2) + lam*float(np.abs(z).sum())
        hist['obj'].append(obj); hist['r'].append(r); hist['s'].append(s)
        if r < tol and s < tol:
            print("converged at", k)
            break

    return x, z, hist


def main():
    rng = np.random.default_rng(0)
    m, n = 200, 500
    A = rng.standard_normal((m, n))
    A = A / np.linalg.norm(A, axis=0, keepdims=True)   # normalize columns

    x_true = np.zeros(n)
    k_nz = 20
    idx = rng.choice(n, k_nz, replace=False)
    x_true[idx] = rng.standard_normal(k_nz)
    b = A @ x_true + 0.01*rng.standard_normal(m)

    lam = 0.02 * float(np.max(np.abs(A.T @ b)))
    print("lambda =", lam)

    x, z, hist = lasso_admm(A, b, lam, rho=1.0, iters=500)

    nnz = int(np.sum(np.abs(z) > 1e-6))
    true_supp = set(idx.tolist())
    est_supp  = set(np.where(np.abs(z) > 1e-6)[0].tolist())
    tp = len(true_supp & est_supp)
    fp = len(est_supp - true_supp)
    fn = len(true_supp - est_supp)
    err = float(np.linalg.norm(z - x_true) / max(np.linalg.norm(x_true), 1e-12))
    print("nnz est =", nnz, "true =", k_nz)
    print("tp =", tp, "fp =", fp, "fn =", fn)
    print("rel err =", err)

    # summary plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    axes[0, 0].stem(x_true, markerfmt='bo', linefmt='b-', basefmt=' ')
    axes[0, 0].set_title("true sparse x, 20 non zero")
    axes[0, 0].set_xlabel("index"); axes[0, 0].set_ylabel("value")
    axes[0, 1].stem(z, markerfmt='rx', linefmt='r-', basefmt=' ')
    axes[0, 1].set_title("LASSO estimate z, " + str(nnz) + " non zero")
    axes[0, 1].set_xlabel("index"); axes[0, 1].set_ylabel("value")
    axes[1, 0].semilogy(hist['obj'])
    axes[1, 0].set_title("objective vs iter")
    axes[1, 0].set_xlabel("iter"); axes[1, 0].set_ylabel("(1/2)||Ax-b||^2 + lam ||z||_1")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 1].semilogy(hist['r'], label='primal ||x-z||')
    axes[1, 1].semilogy(hist['s'], label='dual rho ||z_k - z_{k-1}||')
    axes[1, 1].set_title("residuals")
    axes[1, 1].set_xlabel("iter"); axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "lasso.png"), dpi=150)
    plt.close(fig)

    # overlay plot
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.stem(np.arange(n), x_true, markerfmt='bo', linefmt='b-', basefmt=' ', label='true')
    ax.stem(np.arange(n), z, markerfmt='rx', linefmt='r--', basefmt=' ', label='ADMM')
    ax.set_title("true x vs ADMM LASSO estimate z")
    ax.set_xlabel("index"); ax.set_ylabel("value")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "lasso_overlay.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
