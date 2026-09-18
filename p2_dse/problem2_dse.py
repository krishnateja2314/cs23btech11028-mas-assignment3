import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
import networkx as nx

OUT = os.path.dirname(os.path.abspath(__file__))


def get_graph(N, p, seed):
    rng = np.random.default_rng(seed)
    for _ in range(500):
        G = nx.erdos_renyi_graph(N, p, seed=int(rng.integers(0, 1_000_000)))
        if nx.is_connected(G):
            return G
    raise RuntimeError("not connected")


def metropolis(G):
    N = G.number_of_nodes()
    W = np.zeros((N, N))
    deg = dict(G.degree())
    for (i, j) in G.edges():
        w = 1.0 / (1 + max(deg[i], deg[j]))
        W[i, j] = w
        W[j, i] = w
    for i in range(N):
        W[i, i] = 1.0 - W[i].sum()
    return W


def main():
    rng = np.random.default_rng(42)
    N = 8
    G = get_graph(N, 0.4, seed=42)
    W = metropolis(G)

    # sanity check on W
    assert np.allclose(W.sum(axis=1), 1.0)
    assert np.allclose(W.sum(axis=0), 1.0)

    d = 3
    zbar0 = np.zeros(d)
    S0 = 4.0 * np.eye(d)
    Sw = 0.05 * np.eye(d)
    T = 50

    # sensor positions and per sensor noise
    X = rng.uniform(-5.0, 5.0, size=(N, d))
    sig = rng.uniform(0.3, 0.8, size=N)
    Sv_inv = [(1.0/sig[i]**2)*np.eye(d) for i in range(N)]

    # step size, safe wrt lipschitz of grad
    L = float(sum(1.0/sig[i]**2 for i in range(N)))
    alpha = 0.5 / L
    Kin = 120
    print("L =", L, "alpha =", alpha)

    # true trajectory
    z_true = np.zeros((T+1, d))
    z_true[0] = rng.multivariate_normal(zbar0, S0)
    for t in range(T):
        z_true[t+1] = z_true[t] + rng.multivariate_normal(np.zeros(d), Sw)

    # local estimates, one row per agent
    Z = np.tile(zbar0, (N, 1))
    zhat = np.zeros((T+1, d))
    gap = np.zeros(T+1)

    for t in range(T+1):
        # measurements at time t
        noise = np.stack([rng.multivariate_normal(np.zeros(d), (sig[i]**2)*np.eye(d))
                          for i in range(N)])
        Y = X - z_true[t] + noise

        # DGD inner loop
        for _ in range(Kin):
            grads = np.stack([Sv_inv[i] @ (Y[i] - X[i] + Z[i]) for i in range(N)])
            Z = W @ Z - alpha * grads

        zhat[t] = Z.mean(axis=0)
        gap[t] = np.max(np.linalg.norm(Z - Z.mean(axis=0), axis=1))

    e = zhat - z_true
    e_norm = np.linalg.norm(e, axis=1)
    print("final err:", e_norm[-1])
    print("mean err :", e_norm.mean())
    print("max gap  :", gap.max())

    # graph
    fig, ax = plt.subplots(figsize=(5,5))
    pos = nx.spring_layout(G, seed=2)
    nx.draw(G, pos, ax=ax, with_labels=True, node_color='tab:orange',
            node_size=380, font_color='white', font_size=10, edge_color='gray')
    ax.set_title("drone graph, N=8, p=0.4")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "graph.png"), dpi=150)
    plt.close(fig)

    # 3d plus error norm
    fig = plt.figure(figsize=(12,5))
    ax = fig.add_subplot(1,2,1, projection='3d')
    ax.plot(z_true[:,0], z_true[:,1], z_true[:,2], 'b-o', ms=3, label='true')
    ax.plot(zhat[:,0], zhat[:,1], zhat[:,2], 'r--x', ms=3, label='estimate')
    ax.scatter(X[:,0], X[:,1], X[:,2], marker='^', color='k', s=60, label='sensors')
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
    ax.set_title('true vs estimated trajectory')
    ax.legend(loc='upper left', fontsize=8)

    ax2 = fig.add_subplot(1,2,2)
    ax2.plot(np.arange(T+1), e_norm, 'k-')
    ax2.set_xlabel('time t'); ax2.set_ylabel('||e(t)||')
    ax2.set_title('estimation error')
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "estimate.png"), dpi=150)
    plt.close(fig)

    # per coord plot
    fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
    lbl = ['x', 'y', 'z']
    for i in range(3):
        axes[i].plot(z_true[:, i], 'b-o', label='true '+lbl[i], ms=3)
        axes[i].plot(zhat[:, i], 'r--x', label='est '+lbl[i], ms=3)
        axes[i].set_ylabel(lbl[i])
        axes[i].grid(True, alpha=0.3); axes[i].legend()
    axes[-1].set_xlabel('time t')
    fig.suptitle('true vs estimated coordinates')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "coord_trace.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
