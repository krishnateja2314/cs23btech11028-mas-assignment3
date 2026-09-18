"""
Problem 2: Distributed State Estimation of an intruder using DGD.

References:
- Distributed Learning Lecture, "Motivating Problem 2 -- Distributed State
  Estimation" (page 12) and "Consensus + Gradient Descent" (pages 13-14).
- DGD update rule (Distributed Learning Lecture, page 7):
    theta_i[k+1] = sum_j W_ij theta_j[k] - alpha * grad_i( theta_i[k] )
- The mixing matrix W is chosen as Metropolis-Hastings weights (declared as
  extra material because the lecture does not spell out an explicit formula).

Measurement model:
    y_i(t) = x_i - z(t) + v_i,      v_i ~ N(0, Sigma_v^{(i)})
Local negative-log-likelihood (up to constants):
    f_i(z; t) = (1/2) * (y_i(t) - x_i + z)^T Sigma_v^{(i)-1} (y_i(t) - x_i + z)
    grad_z f_i(z; t) = Sigma_v^{(i)-1} (y_i(t) - x_i + z)

Author: Krishna Teja (CS23BTECH11028)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
import networkx as nx

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

def connected_er(N, p, seed):
    rng = np.random.default_rng(seed)
    for _ in range(500):
        G = nx.erdos_renyi_graph(N, p, seed=int(rng.integers(0, 1_000_000)))
        if nx.is_connected(G):
            return G
    raise RuntimeError("could not get a connected ER graph")

def metropolis_weights(G):
    """Metropolis-Hastings weights, doubly stochastic, W_ij=0 if j not neighbour."""
    N = G.number_of_nodes()
    W = np.zeros((N, N))
    deg = dict(G.degree())
    for (i, j) in G.edges():
        w = 1.0 / (1 + max(deg[i], deg[j]))
        W[i, j] = w; W[j, i] = w
    for i in range(N):
        W[i, i] = 1.0 - W[i].sum()
    return W

def main():
    rng = np.random.default_rng(42)
    N = 8
    G = connected_er(N, 0.4, seed=42)
    W = metropolis_weights(G)
    assert np.allclose(W.sum(axis=1), 1.0)   # row-stochastic
    assert np.allclose(W.sum(axis=0), 1.0)   # column-stochastic (doubly)

    d = 3
    zbar0  = np.zeros(d)
    Sigma0 = 4.0 * np.eye(d)
    Sigma_w = 0.05 * np.eye(d)
    T = 50

    # sensor locations x_i and per-sensor noise cov Sigma_v^(i)
    X = rng.uniform(-5.0, 5.0, size=(N, d))
    sigmas = rng.uniform(0.3, 0.8, size=N)
    Sig_v_inv = [(1.0 / sigmas[i]**2) * np.eye(d) for i in range(N)]

    # Lipschitz constant of the global gradient sum_i grad f_i is
    #   L_lip = sum_i lambda_max(Sigma_v^{(i)-1}) = sum_i 1/sigma_i^2
    # For stability of DGD we take alpha < 2 / L_lip.
    L_lip = float(sum(1.0/sigmas[i]**2 for i in range(N)))

    # true trajectory
    z_true = np.zeros((T + 1, d))
    z_true[0] = rng.multivariate_normal(zbar0, Sigma0)
    for t in range(T):
        z_true[t+1] = z_true[t] + rng.multivariate_normal(np.zeros(d), Sigma_w)

    # local estimates z_i in R^3 (rows)
    Z = np.tile(zbar0, (N, 1))

    # Step size chosen inversely proportional to the Lipschitz constant of
    # the local gradients (safe DGD step). Small constant step so we get
    # the classical DGD behaviour (Distributed Learning Lecture, page 7-8).
    alpha  = 0.5 / L_lip
    K_inner = 120
    print(f"L_lip = {L_lip:.3f},  alpha = {alpha:.4f}")

    zhat = np.zeros((T + 1, d))
    consensus_gap = np.zeros(T + 1)

    for t in range(T + 1):
        # measurements at time t
        noise = np.stack([rng.multivariate_normal(np.zeros(d),
                                                  (sigmas[i]**2)*np.eye(d))
                          for i in range(N)])
        Y = X - z_true[t] + noise

        # DGD inner iterations at fixed time t
        for _ in range(K_inner):
            grads = np.stack([Sig_v_inv[i] @ (Y[i] - X[i] + Z[i])
                              for i in range(N)])
            Z = W @ Z - alpha * grads

        zhat[t] = Z.mean(axis=0)
        consensus_gap[t] = np.max(np.linalg.norm(Z - Z.mean(axis=0), axis=1))

    err = zhat - z_true
    err_norm = np.linalg.norm(err, axis=1)
    print(f"final ||e(T)||  = {err_norm[-1]:.4f}")
    print(f"mean  ||e(t)||  = {err_norm.mean():.4f}")
    print(f"max consensus gap across time = {consensus_gap.max():.4e}")

    # --------- figure 1: communication graph ---------
    fig1, ax1 = plt.subplots(figsize=(5, 5))
    pos = nx.spring_layout(G, seed=2)
    nx.draw(G, pos, ax=ax1, with_labels=True,
            node_color='tab:orange', node_size=380,
            font_color='white', font_size=10, edge_color='gray')
    ax1.set_title(f"Drone communication graph (N={N}, p=0.4)")
    fig1.tight_layout()
    fig1.savefig(os.path.join(OUT_DIR, "graph.png"), dpi=150)
    plt.close(fig1)

    # --------- figure 2: true vs estimated 3D trajectory ---------
    fig2 = plt.figure(figsize=(12, 5))
    ax = fig2.add_subplot(1, 2, 1, projection='3d')
    ax.plot(z_true[:,0], z_true[:,1], z_true[:,2], 'b-o',
            markersize=3, label='true z(t)')
    ax.plot(zhat[:,0],   zhat[:,1],   zhat[:,2], 'r--x',
            markersize=3, label='estimate z_hat(t)')
    ax.scatter(X[:,0], X[:,1], X[:,2], marker='^', color='k',
               s=60, label='sensors x_i')
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
    ax.set_title('True vs estimated intruder trajectory')
    ax.legend(loc='upper left', fontsize=8)

    ax2 = fig2.add_subplot(1, 2, 2)
    ax2.plot(np.arange(T + 1), err_norm, 'k-', label='||e(t)||_2')
    ax2.set_xlabel('time t'); ax2.set_ylabel('estimation error norm')
    ax2.set_title('Estimation error over time')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(os.path.join(OUT_DIR, "estimate.png"), dpi=150)
    plt.close(fig2)

    # --------- figure 3: per-coordinate trace ---------
    fig3, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
    labels = ['x', 'y', 'z']
    for i in range(3):
        axes[i].plot(z_true[:, i], 'b-o', label=f'true {labels[i]}', ms=3)
        axes[i].plot(zhat[:, i],   'r--x', label=f'estimate {labels[i]}', ms=3)
        axes[i].set_ylabel(f'{labels[i]} coord')
        axes[i].grid(True, alpha=0.3); axes[i].legend()
    axes[-1].set_xlabel('time t')
    fig3.suptitle('True vs estimated intruder coordinates')
    fig3.tight_layout()
    fig3.savefig(os.path.join(OUT_DIR, "coord_trace.png"), dpi=150)
    plt.close(fig3)

if __name__ == "__main__":
    main()
