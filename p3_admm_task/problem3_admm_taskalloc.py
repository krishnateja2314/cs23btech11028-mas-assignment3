"""
Problem 3: Distributed task allocation via ADMM.

References:
- ADMM Lecture, Consensus ADMM (page 8): min sum_i f_i(x_i) s.t. x_i = z.
- ADMM Lecture, "ADMM hack for any distributed decision making problem"
  (page 11): each agent updates only its own row and uses a neighbourhood
  averaging for the consensus variable.
- Helper notes on augmented Lagrangian and scaled form ADMM (pages 12-14).

Extra material (not spelled out in the lecture, declared explicitly):
- LP convex relaxation of the binary linear program (0-1 -> [0,1] box).
- Integer rounding via the Hungarian algorithm on an expanded cost matrix,
  which recovers a valid binary assignment with row sums b and column sums 1.

Binary LP:
    min   sum_{i,j} C_ij x_ij
    s.t.  sum_i x_ij = 1        for every task j     (task covered once)
          sum_j x_ij = b_i      for every agent i    (capacity)
          x_ij in {0,1}
LP relaxation:
          x_ij in [0,1]

Consensus ADMM formulation:
    min sum_i f_i(X^{(i)})      s.t.  X^{(i)} = Z for all i
    f_i(X) = <C_i,:, X_{i,:}> + I_{sum_j X_{i,j}=b_i} + I_{[0,1] box on X}
    plus a task-cover constraint I_{sum_i Z_{i,j}=1 for every j} on Z.

Author: Krishna Teja (CS23BTECH11028)
"""

import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

def connected_er(N, p, seed):
    rng = np.random.default_rng(seed)
    for _ in range(500):
        G = nx.erdos_renyi_graph(N, p, seed=int(rng.integers(0, 1_000_000)))
        if nx.is_connected(G):
            return G
    raise RuntimeError("could not get a connected ER graph")

def random_capacities(N, M, seed):
    """Random positive composition of M into N parts (so sum(b) = M, b_i >= 1)."""
    rng = np.random.default_rng(seed)
    cuts = np.sort(rng.choice(np.arange(1, M), size=N-1, replace=False))
    parts = np.diff(np.concatenate([[0], cuts, [M]]))
    return parts.astype(int)

def project_capped_simplex(v, s, lo=0.0, hi=1.0, tol=1e-10, itmax=100):
    """Project v onto { r : sum(r) = s, lo <= r_j <= hi } via bisection."""
    n = len(v)
    if s < lo*n - 1e-9 or s > hi*n + 1e-9:
        raise ValueError(f"capped simplex infeasible: s={s}, n={n}")
    def f(tau): return np.clip(v - tau, lo, hi).sum() - s
    lo_t, hi_t = v.min() - hi - 1.0, v.max() - lo + 1.0
    for _ in range(itmax):
        mid = 0.5*(lo_t + hi_t)
        if f(mid) > 0: lo_t = mid
        else:          hi_t = mid
        if hi_t - lo_t < tol: break
    return np.clip(v - 0.5*(lo_t + hi_t), lo, hi)

def project_col_sum_one(Z):
    """Project each column of Z onto { v : sum(v) = 1 }."""
    N = Z.shape[0]
    return Z + (1.0 - Z.sum(axis=0, keepdims=True)) / N

def round_to_binary(X, b):
    """Round fractional X to a binary matrix with row sums b, col sums 1
       via the Hungarian algorithm on an expanded cost matrix.
       Uses -X as the reward (Hungarian minimises)."""
    N, M = X.shape
    assert b.sum() == M, "capacities must sum to M"
    # expand each agent i into b_i duplicated rows
    cost_exp = np.vstack([np.tile(-X[i:i+1], (b[i], 1)) for i in range(N)])
    row_ind, col_ind = linear_sum_assignment(cost_exp)   # M x M
    Xb = np.zeros_like(X)
    idx = np.concatenate([np.full(b[i], i) for i in range(N)])
    for r, c in zip(row_ind, col_ind):
        Xb[idx[r], c] = 1.0
    return Xb

def admm_task_alloc(C, G, b, rho=1.0, iters=400):
    """
    Distributed consensus ADMM with neighbourhood averaging.

    Each agent i owns a full N x M matrix X^{(i)} and a scaled dual U^{(i)}.
    The consensus variable z_i (row of Z) is what agent i is responsible for
    within the global Z; z_i is the target that x^{(i)}_{i,:} must agree on.

    Updates:
      X^{(i),k+1}_{i,:}   = capped simplex projection with row sum b_i
      X^{(i),k+1}_{k,:}   = clip(Z_k[k,:] - U^{(i),k}[k,:], 0, 1) for k != i
      Z^{k+1}[i,:]        = (1/|N_i+1|) sum_{j in N_i union {i}} (X^{(j),k+1}[i,:] + U^{(j),k}[i,:])
      then column-sum-to-one projection on Z
      U^{(i),k+1}         = U^{(i),k} + (X^{(i),k+1} - Z^{k+1})
    """
    N, M = C.shape
    Xs = [np.full((N, M), 1.0/N) for _ in range(N)]
    Us = [np.zeros((N, M))       for _ in range(N)]
    Z  = np.full((N, M), 1.0/N)

    # neighbour list including self
    nbrs = [list(G.neighbors(i)) + [i] for i in range(N)]
    primal_res_hist = []
    obj_hist        = []

    for k in range(iters):
        new_Xs = [np.empty((N, M)) for _ in range(N)]
        for i in range(N):
            X_new = np.empty((N, M))
            # own row: capped simplex projection with row sum b_i
            v = Z[i] - Us[i][i] - C[i]/rho
            X_new[i] = project_capped_simplex(v, s=float(b[i]), lo=0.0, hi=1.0)
            # other rows: just clip
            for r in range(N):
                if r == i: continue
                v2 = Z[r] - Us[i][r]
                X_new[r] = np.clip(v2, 0.0, 1.0)
            new_Xs[i] = X_new
        Xs = new_Xs

        # z-update: only agent i's row of Z is needed; use its neighbours'
        # copies of that row (this is the distributed "ADMM hack", ADMM
        # Lecture page 11).
        newZ = np.zeros((N, M))
        for i in range(N):
            acc = np.zeros(M); cnt = 0
            for j in nbrs[i]:
                acc += Xs[j][i] + Us[j][i]
                cnt += 1
            newZ[i] = acc / cnt
        Z = project_col_sum_one(newZ)

        # dual update
        for i in range(N):
            Us[i] = Us[i] + (Xs[i] - Z)

        # residual and objective
        primal_res_hist.append(
            np.mean([np.linalg.norm(Xs[i] - Z) for i in range(N)]))
        obj_hist.append(float((C * Z).sum()))
    return Z, primal_res_hist, obj_hist

def solve_lp_reference(C, b):
    """Solve the relaxed LP centrally, for cross-checking."""
    from scipy.optimize import linprog
    N, M = C.shape
    c = C.reshape(-1)
    # sum_i x_ij = 1  ->  M rows
    A1 = np.zeros((M, N*M))
    for j in range(M):
        for i in range(N):
            A1[j, i*M + j] = 1.0
    b1 = np.ones(M)
    # sum_j x_ij = b_i -> N rows
    A2 = np.zeros((N, N*M))
    for i in range(N):
        for j in range(M):
            A2[i, i*M + j] = 1.0
    b2 = b.astype(float)
    A_eq = np.vstack([A1, A2])
    b_eq = np.concatenate([b1, b2])
    res = linprog(c, A_eq=A_eq, b_eq=b_eq,
                  bounds=[(0, 1)]*N*M, method='highs')
    return res.x.reshape(N, M), res.fun

def run():
    N, M = 5, 12
    G = connected_er(N, 0.5, seed=3)
    b = random_capacities(N, M, seed=3)
    print("N =", N, "M =", M, "capacities b =", b, "sum(b) =", b.sum())

    # ----- three cost matrices, same N, M, b, G -----
    rng = np.random.default_rng(3)
    trials = []
    for t in range(3):
        C = rng.uniform(0.1, 10.0, size=(N, M))
        Z_admm, r_hist, o_hist = admm_task_alloc(C, G, b, rho=1.0, iters=400)
        X_bin = round_to_binary(Z_admm, b)
        # verify constraints
        assert np.allclose(X_bin.sum(axis=0), 1.0), "task-cover violated"
        assert np.array_equal(X_bin.sum(axis=1).astype(int), b), "capacity violated"
        total_cost = float((C * X_bin).sum())
        # LP central reference
        X_lp, lp_cost = solve_lp_reference(C, b)
        trials.append(dict(C=C, Z=Z_admm, X=X_bin,
                           cost=total_cost, lp_cost=lp_cost,
                           r_hist=r_hist, o_hist=o_hist))
        print(f"Trial {t+1}: ADMM rounded cost = {total_cost:.3f}, "
              f"LP relaxation cost = {lp_cost:.3f}")

    # -------- figure : the communication graph --------
    fig_g, ax = plt.subplots(figsize=(4.5, 4.5))
    pos = nx.spring_layout(G, seed=1)
    nx.draw(G, pos, ax=ax, with_labels=True, node_color='tab:green',
            node_size=420, font_color='white', font_size=11, edge_color='gray')
    ax.set_title(f"Communication graph N={N}, p=0.5")
    fig_g.tight_layout()
    fig_g.savefig(os.path.join(OUT_DIR, "graph.png"), dpi=150)
    plt.close(fig_g)

    # -------- figure : allocation heatmaps for 3 cost matrices --------
    fig, axes = plt.subplots(2, 3, figsize=(15, 6))
    for t, tr in enumerate(trials):
        # top row: cost matrix
        im0 = axes[0, t].imshow(tr['C'], aspect='auto', cmap='viridis')
        axes[0, t].set_title(f"Cost matrix C, trial {t+1}")
        axes[0, t].set_xlabel("task j"); axes[0, t].set_ylabel("agent i")
        plt.colorbar(im0, ax=axes[0, t], fraction=0.046)
        # bottom row: binary allocation
        axes[1, t].imshow(tr['X'], aspect='auto', cmap='Blues', vmin=0, vmax=1)
        for i in range(N):
            for j in range(M):
                if tr['X'][i, j] > 0.5:
                    axes[1, t].text(j, i, "1", ha='center', va='center',
                                    color='red', fontsize=9)
        axes[1, t].set_title(f"Allocation, cost = {tr['cost']:.2f} "
                             f"(LP = {tr['lp_cost']:.2f})")
        axes[1, t].set_xlabel("task j"); axes[1, t].set_ylabel("agent i")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "allocations.png"), dpi=150)
    plt.close(fig)

    # -------- figure : primal residual and objective vs iter (trial 1) --------
    fig2, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    axes[0].semilogy(trials[0]['r_hist'])
    axes[0].set_title("Primal residual (trial 1)")
    axes[0].set_xlabel("iter"); axes[0].set_ylabel("mean ||X^(i) - Z||_F")
    axes[0].grid(True, alpha=0.3)
    for t, tr in enumerate(trials):
        axes[1].plot(tr['o_hist'], label=f"trial {t+1}")
    axes[1].set_title("Objective <C, Z> vs iter")
    axes[1].set_xlabel("iter"); axes[1].set_ylabel("objective")
    axes[1].grid(True, alpha=0.3); axes[1].legend()
    fig2.tight_layout()
    fig2.savefig(os.path.join(OUT_DIR, "convergence.png"), dpi=150)
    plt.close(fig2)

if __name__ == "__main__":
    run()
