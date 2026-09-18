import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx

OUT = os.path.dirname(os.path.abspath(__file__))


def get_graph(N, p, seed):
    rng = np.random.default_rng(seed)
    for _ in range(500):
        G = nx.erdos_renyi_graph(N, p, seed=int(rng.integers(0, 1_000_000)))
        if nx.is_connected(G):
            return G
    raise RuntimeError("graph not connected, try bigger p")


def line(a, b, n=30):
    t = np.linspace(0, 1, n)
    return np.stack([a[0] + t*(b[0]-a[0]), a[1] + t*(b[1]-a[1])], axis=1)


def arc(c, r, a0, a1, n=40):
    a = np.linspace(a0, a1, n)
    return np.stack([c[0] + r*np.cos(a), c[1] + r*np.sin(a)], axis=1)


# each letter is just a list of strokes
def strokes_of(L):
    if L == 'K':
        return [line((0,0),(0,1)), line((0,0.5),(1,1)), line((0,0.5),(1,0))]
    if L == 'R':
        return [line((0,0),(0,1)),
                arc((0.4,0.75), 0.25, np.pi/2, -np.pi/2),
                line((0,0.5),(1,0))]
    if L == 'I':
        return [line((0,1),(1,1)), line((0.5,1),(0.5,0)), line((0,0),(1,0))]
    if L == 'S':
        return [arc((0.5,0.75), 0.25, 0, np.pi),
                arc((0.5,0.75), 0.25, np.pi, 3*np.pi/2),
                arc((0.5,0.25), 0.25, np.pi/2, -np.pi/2),
                arc((0.5,0.25), 0.25, -np.pi/2, 0)]
    if L == 'H':
        return [line((0,0),(0,1)), line((1,0),(1,1)), line((0,0.5),(1,0.5))]
    if L == 'N':
        return [line((0,0),(0,1)), line((0,1),(1,0)), line((1,0),(1,1))]
    if L == 'A':
        return [line((0,0),(0.5,1)), line((0.5,1),(1,0)),
                line((0.25,0.5),(0.75,0.5))]
    raise ValueError(L)


def sample_one(poly, n):
    d = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(d)])
    ts = np.linspace(0, s[-1], n)
    out = np.zeros((n, 2))
    for k, t in enumerate(ts):
        idx = np.searchsorted(s, t)
        idx = min(max(idx, 1), len(poly)-1)
        a = (t - s[idx-1]) / max(s[idx] - s[idx-1], 1e-12)
        out[k] = poly[idx-1] + a * (poly[idx] - poly[idx-1])
    return out


def sample_letter(strokes, N, floor=4):
    S = len(strokes)
    if S*floor > N:
        floor = max(1, N//S)
    lens = np.array([np.linalg.norm(np.diff(p, axis=0), axis=1).sum() for p in strokes])
    left = N - S*floor
    prop = lens / lens.sum()
    extra = np.floor(prop * left).astype(int)
    # hand out any remainder to the longest strokes
    order = np.argsort(-lens)
    for k in range(left - extra.sum()):
        extra[order[k % S]] += 1
    counts = extra + floor
    return np.vstack([sample_one(strokes[i], counts[i]) for i in range(S)])


def main():
    np.random.seed(7)
    N = 20
    p = 0.3
    G = get_graph(N, p, seed=7)
    A = nx.to_numpy_array(G)

    name = "KRISHNA"

    # target points for each letter, centered around origin
    targets = []
    for L in name:
        pts = sample_letter(strokes_of(L), N, floor=4) * 1.2 - np.array([0.6, 0.6])
        targets.append(pts)

    # random start
    P = np.random.uniform(-1.6, 1.6, size=(N, 2))

    dt = 0.05
    steps = 300
    frames = [P.copy()]
    letter_end = [0]
    err_hist = []

    # formation control from MAS Algorithms lecture, Algorithm 2
    for r in targets:
        for _ in range(steps):
            u = np.zeros_like(P)
            for i in range(N):
                for j in np.where(A[i] > 0)[0]:
                    u[i] += (P[j] - P[i]) - (r[j] - r[i])
            P = P + dt*u
            frames.append(P.copy())
            err_hist.append(np.max(np.linalg.norm(P - r, axis=1)))
        letter_end.append(len(frames)-1)

    print("frames:", len(frames))
    print("final error:", err_hist[-1])

    # graph plot
    fig, ax = plt.subplots(figsize=(5,5))
    pos = nx.spring_layout(G, seed=1)
    nx.draw(G, pos, ax=ax, with_labels=True, node_color='tab:blue',
            node_size=350, font_color='white', font_size=9, edge_color='gray')
    ax.set_title(f"ER graph N={N}, p={p}")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "graph.png"), dpi=150)
    plt.close(fig)

    # animation
    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_xlim(-1.8, 1.8); ax.set_ylim(-1.8, 1.8); ax.set_aspect('equal')
    ax.set_title("Formation control : KRISHNA")

    edges = []
    for (i, j) in G.edges():
        ln, = ax.plot([], [], color='lightgray', lw=0.6, zorder=1)
        edges.append((i, j, ln))
    dots = ax.scatter([], [], s=45, c='tab:blue', zorder=2,
                      edgecolors='k', linewidths=0.5)
    txt = ax.text(0.02, 0.95, "", transform=ax.transAxes, fontsize=14, weight='bold')

    step = 4
    ids = list(range(0, len(frames), step))

    def which(k):
        for li in range(len(letter_end)-1):
            if letter_end[li] <= k < letter_end[li+1]:
                return name[li]
        return name[-1]

    def update(idx):
        k = ids[idx]
        Pk = frames[k]
        dots.set_offsets(Pk)
        for (i, j, ln) in edges:
            ln.set_data([Pk[i,0], Pk[j,0]], [Pk[i,1], Pk[j,1]])
        txt.set_text("letter: " + which(k))
        return [dots, txt] + [ln for _, _, ln in edges]

    ani = animation.FuncAnimation(fig, update, frames=len(ids), interval=40, blit=False)
    ani.save(os.path.join(OUT, "krishna_formation.gif"),
             writer=animation.PillowWriter(fps=25))
    plt.close(fig)
    print("saved gif")

    # stills of all 7 letters
    fig, axes = plt.subplots(1, len(name), figsize=(2.6*len(name), 3.0))
    for li, L in enumerate(name):
        ax = axes[li]
        k = letter_end[li+1] - 1
        Pk = frames[k]
        for (i, j) in G.edges():
            ax.plot([Pk[i,0], Pk[j,0]], [Pk[i,1], Pk[j,1]], color='lightgray', lw=0.5)
        ax.scatter(Pk[:,0], Pk[:,1], s=32, c='tab:blue',
                   edgecolors='k', linewidths=0.4)
        ax.set_title("letter " + L)
        ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2); ax.set_aspect('equal')
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "letter_stills.png"), dpi=150)
    plt.close(fig)

    # error plot
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(err_hist, color='tab:red')
    for b in letter_end[1:-1]:
        ax.axvline(b, color='k', lw=0.4, ls='--')
    ax.set_xlabel("iteration"); ax.set_ylabel("max_i || p_i - r_i ||")
    ax.set_title("formation error")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "formation_error.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
