import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment, linprog

OUT = os.path.dirname(os.path.abspath(__file__))


def get_graph(N, p, seed):
    rng = np.random.default_rng(seed)
    for _ in range(500):
        G = nx.erdos_renyi_graph(N, p, seed=int(rng.integers(0, 1_000_000)))
        if nx.is_connected(G):
            return G
    raise RuntimeError("not connected")


def rand_caps(N, M, seed):
    # random positive parts that sum to M
    rng = np.random.default_rng(seed)
    cuts = np.sort(rng.choice(np.arange(1, M), size=N-1, replace=False))
    return np.diff(np.concatenate([[0], cuts, [M]])).astype(int)


def project_capped_simplex(v, s, lo=0.0, hi=1.0):
    # project onto { r : sum(r)=s, lo<=r<=hi } by bisecting tau
    n = len(v)
    if s < lo*n - 1e-9 or s > hi*n + 1e-9:
        raise ValueError("infeasible")
    f = lambda tau: np.clip(v - tau, lo, hi).sum() - s
    lot, hit = v.min() - hi - 1, v.max() - lo + 1
    for _ in range(80):
        mid = 0.5*(lot + hit)
        if f(mid) > 0: lot = mid
        else:          hit = mid
    return np.clip(v - 0.5*(lot+hit), lo, hi)


def proj_col_one(Z):
    N = Z.shape[0]
    return Z + (1.0 - Z.sum(axis=0, keepdims=True))/N


def round_bin(X, b):
    # duplicate row i by b_i times, run Hungarian
    N, M = X.shape
    assert b.sum() == M
    cost = np.vstack([np.tile(-X[i:i+1], (b[i], 1)) for i in range(N)])
    ri, ci = linear_sum_assignment(cost)
    Xb = np.zeros_like(X)
    idx = np.concatenate([np.full(b[i], i) for i in range(N)])
    for r, c in zip(ri, ci):
        Xb[idx[r], c] = 1.0
    return Xb


def admm(C, G, b, rho=1.0, iters=400):
    N, M = C.shape
    Xs = [np.full((N, M), 1.0/N) for _ in range(N)]
    Us = [np.zeros((N, M)) for _ in range(N)]
    Z  = np.full((N, M), 1.0/N)

    nb = [list(G.neighbors(i)) + [i] for i in range(N)]
    r_hist, o_hist = [], []

    for k in range(iters):
        # x update, one per agent
        new_Xs = [np.empty((N, M)) for _ in range(N)]
        for i in range(N):
            Xn = np.empty((N, M))
            v = Z[i] - Us[i][i] - C[i]/rho
            Xn[i] = project_capped_simplex(v, float(b[i]))
            for r in range(N):
                if r == i: continue
                Xn[r] = np.clip(Z[r] - Us[i][r], 0.0, 1.0)
            new_Xs[i] = Xn
        Xs = new_Xs

        # z update, neighbourhood average of the i-th row
        newZ = np.zeros((N, M))
        for i in range(N):
            acc = np.zeros(M); c = 0
            for j in nb[i]:
                acc += Xs[j][i] + Us[j][i]
                c += 1
            newZ[i] = acc/c
        Z = proj_col_one(newZ)

        # dual update
        for i in range(N):
            Us[i] = Us[i] + (Xs[i] - Z)

        r_hist.append(np.mean([np.linalg.norm(Xs[i]-Z) for i in range(N)]))
        o_hist.append(float((C*Z).sum()))

    return Z, r_hist, o_hist


def lp_ref(C, b):
    N, M = C.shape
    c = C.reshape(-1)
    A1 = np.zeros((M, N*M))
    for j in range(M):
        for i in range(N):
            A1[j, i*M + j] = 1.0
    A2 = np.zeros((N, N*M))
    for i in range(N):
        for j in range(M):
            A2[i, i*M + j] = 1.0
    Aeq = np.vstack([A1, A2])
    beq = np.concatenate([np.ones(M), b.astype(float)])
    res = linprog(c, A_eq=Aeq, b_eq=beq, bounds=[(0,1)]*N*M, method='highs')
    return res.x.reshape(N, M), res.fun


def main():
    N, M = 5, 12
    G = get_graph(N, 0.5, seed=3)
    b = rand_caps(N, M, seed=3)
    print("b =", b, "sum =", b.sum())

    rng = np.random.default_rng(3)
    trials = []
    for t in range(3):
        C = rng.uniform(0.1, 10.0, size=(N, M))
        Z, rh, oh = admm(C, G, b, rho=1.0, iters=400)
        Xb = round_bin(Z, b)
        assert np.allclose(Xb.sum(axis=0), 1.0)
        assert np.array_equal(Xb.sum(axis=1).astype(int), b)
        tot = float((C*Xb).sum())
        _, lp = lp_ref(C, b)
        trials.append(dict(C=C, X=Xb, cost=tot, lp=lp, rh=rh, oh=oh))
        print(f"trial {t+1}: ADMM = {tot:.3f}, LP = {lp:.3f}")

    # graph
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    pos = nx.spring_layout(G, seed=1)
    nx.draw(G, pos, ax=ax, with_labels=True, node_color='tab:green',
            node_size=420, font_color='white', font_size=11, edge_color='gray')
    ax.set_title(f"graph N={N}, p=0.5")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "graph.png"), dpi=150)
    plt.close(fig)

    # cost + allocation heatmaps for 3 trials
    fig, axes = plt.subplots(2, 3, figsize=(15, 6))
    for t, tr in enumerate(trials):
        im = axes[0, t].imshow(tr['C'], aspect='auto', cmap='viridis')
        axes[0, t].set_title("cost matrix C, trial " + str(t+1))
        axes[0, t].set_xlabel("task j"); axes[0, t].set_ylabel("agent i")
        plt.colorbar(im, ax=axes[0, t], fraction=0.046)

        axes[1, t].imshow(tr['X'], aspect='auto', cmap='Blues', vmin=0, vmax=1)
        for i in range(N):
            for j in range(M):
                if tr['X'][i, j] > 0.5:
                    axes[1, t].text(j, i, "1", ha='center', va='center',
                                    color='red', fontsize=9)
        axes[1, t].set_title(f"allocation, cost = {tr['cost']:.2f} (LP = {tr['lp']:.2f})")
        axes[1, t].set_xlabel("task j"); axes[1, t].set_ylabel("agent i")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "allocations.png"), dpi=150)
    plt.close(fig)

    # convergence
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    axes[0].semilogy(trials[0]['rh'])
    axes[0].set_title("primal residual, trial 1")
    axes[0].set_xlabel("iter"); axes[0].set_ylabel("mean ||X - Z||")
    axes[0].grid(True, alpha=0.3)
    for t, tr in enumerate(trials):
        axes[1].plot(tr['oh'], label="trial " + str(t+1))
    axes[1].set_title("objective vs iter")
    axes[1].set_xlabel("iter"); axes[1].set_ylabel("<C, Z>")
    axes[1].grid(True, alpha=0.3); axes[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "convergence.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
