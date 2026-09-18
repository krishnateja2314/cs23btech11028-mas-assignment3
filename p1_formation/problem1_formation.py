"""
Problem 1: Formation Control to spell KRISHNA TEJA with N=20 agents.

Reference:
- MAS Algorithms Lecture, Algorithm 2 (Distributed Formation Control), page 7.
  u_i[k] = sum_{j in N_i} a_ij [(p_j - p_i) - (r_j - r_i)]
  p_i[k+1] = p_i[k] + dt * u_i[k]

Graph: Erdos-Renyi G(N, p), resampled until connected.
Author: Krishna Teja (CS23BTECH11028)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ----------------------- graph -----------------------
def connected_er_graph(N, p, seed):
    rng = np.random.default_rng(seed)
    for _ in range(500):
        G = nx.erdos_renyi_graph(N, p, seed=int(rng.integers(0, 1_000_000)))
        if nx.is_connected(G):
            return G
    raise RuntimeError("could not get a connected ER graph")

# --------------------- letter shapes ---------------------
def _line(a, b, n=30):
    t = np.linspace(0, 1, n)
    return np.stack([a[0] + t*(b[0]-a[0]), a[1] + t*(b[1]-a[1])], axis=1)

def _arc(c, r, a0, a1, n=40):
    a = np.linspace(a0, a1, n)
    return np.stack([c[0] + r*np.cos(a), c[1] + r*np.sin(a)], axis=1)

def letter_strokes(L):
    """Return a list of strokes for letter L. Each stroke is a Kx2 poly-line
       drawn inside [0,1]x[0,1]. Sampling later gives each stroke a share of
       N proportional to its arc length (with a floor of 3 per stroke), so
       short strokes like the middle bar of H still get enough points."""
    if L == 'K':
        return [_line((0,0),(0,1)),
                _line((0,0.5),(1,1)),
                _line((0,0.5),(1,0))]
    if L == 'R':
        return [_line((0,0),(0,1)),
                _arc((0.4,0.75), 0.25, np.pi/2, -np.pi/2),
                _line((0,0.5),(1,0))]
    if L == 'I':
        return [_line((0,1),(1,1)),
                _line((0.5,1),(0.5,0)),
                _line((0,0),(1,0))]
    if L == 'S':
        return [_arc((0.5,0.75), 0.25, 0, np.pi),
                _arc((0.5,0.75), 0.25, np.pi, 3*np.pi/2),
                _arc((0.5,0.25), 0.25, np.pi/2, -np.pi/2),
                _arc((0.5,0.25), 0.25, -np.pi/2, 0)]
    if L == 'H':
        return [_line((0,0),(0,1)),
                _line((1,0),(1,1)),
                _line((0,0.5),(1,0.5))]
    if L == 'N':
        return [_line((0,0),(0,1)),
                _line((0,1),(1,0)),
                _line((1,0),(1,1))]
    if L == 'A':
        return [_line((0,0),(0.5,1)),
                _line((0.5,1),(1,0)),
                _line((0.25,0.5),(0.75,0.5))]
    if L == 'T':
        return [_line((0,1),(1,1)),
                _line((0.5,1),(0.5,0))]
    if L == 'E':
        return [_line((0,0),(0,1)),
                _line((0,1),(1,1)),
                _line((0,0.5),(0.7,0.5)),
                _line((0,0),(1,0))]
    if L == 'J':
        return [_line((0,1),(1,1)),
                _line((1,1),(1,0.2)),
                _arc((0.5,0.2), 0.5, 0, -np.pi)]
    raise ValueError(L)

def _sample_stroke_uniform(poly, n):
    """Equally spaced n points along a single poly-line (arc-length param.)."""
    d = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(d)])
    total = s[-1]
    ts = np.linspace(0, total, n)
    out = np.zeros((n, 2))
    for k, t in enumerate(ts):
        idx = np.searchsorted(s, t)
        idx = min(max(idx, 1), len(poly)-1)
        alpha = (t - s[idx-1]) / max(s[idx] - s[idx-1], 1e-12)
        out[k] = poly[idx-1] + alpha * (poly[idx] - poly[idx-1])
    return out

def sample_letter(strokes, N, floor_per_stroke=3):
    """Distribute N samples over the strokes proportional to arc length,
       but give every stroke at least `floor_per_stroke` points so that
       short strokes (like the middle bar of H) are not under-represented."""
    S = len(strokes)
    if S * floor_per_stroke > N:
        floor_per_stroke = max(1, N // S)
    lengths = np.array([np.linalg.norm(np.diff(p, axis=0), axis=1).sum()
                        for p in strokes])
    remaining = N - S * floor_per_stroke
    # proportional share of the remaining points by length
    prop = lengths / lengths.sum()
    extra = np.floor(prop * remaining).astype(int)
    # hand out the leftovers (from rounding) to the longest strokes
    leftover = remaining - extra.sum()
    order = np.argsort(-lengths)
    for k in range(leftover):
        extra[order[k % S]] += 1
    counts = extra + floor_per_stroke
    assert counts.sum() == N
    parts = [_sample_stroke_uniform(strokes[i], counts[i]) for i in range(S)]
    return np.vstack(parts)

# ----------------------- main sim -----------------------
def main():
    np.random.seed(7)
    N = 20
    p_edge = 0.3
    G = connected_er_graph(N, p_edge, seed=7)
    A = nx.to_numpy_array(G)   # unweighted adjacency

    name = "KRISHNA"          # first name only, as instructed (Manoj example)

    # per-letter targets, centered in a 1.2 x 1.2 box around origin
    letters_pts = []
    for L in name:
        strokes = letter_strokes(L)
        pts = sample_letter(strokes, N, floor_per_stroke=4) * 1.2 - np.array([0.6, 0.6])
        letters_pts.append(pts)

    # random initial positions in a big box
    P = np.random.uniform(-1.6, 1.6, size=(N, 2))

    dt = 0.05
    iters_per_letter = 300
    frames = [P.copy()]
    letter_boundaries = [0]     # frame index at which each letter starts
    formation_error_hist = []

    for r_pts in letters_pts:
        for _ in range(iters_per_letter):
            u = np.zeros_like(P)
            # Algorithm 2 (Distributed Formation Control):
            # u_i = sum_{j in N_i} [(p_j - p_i) - (r_j - r_i)]
            for i in range(N):
                neigh = np.where(A[i] > 0)[0]
                for j in neigh:
                    u[i] += (P[j] - P[i]) - (r_pts[j] - r_pts[i])
            P = P + dt * u
            frames.append(P.copy())
            formation_error_hist.append(np.max(np.linalg.norm(P - r_pts, axis=1)))
        letter_boundaries.append(len(frames) - 1)

    print(f"total frames: {len(frames)}")
    print(f"final formation error (last letter): {formation_error_hist[-1]:.4f}")

    # ------------- draw the ER graph -------------
    fig_g, ax_g = plt.subplots(figsize=(5,5))
    pos = nx.spring_layout(G, seed=1)
    nx.draw(G, pos, ax=ax_g, with_labels=True, node_color='tab:blue',
            node_size=350, font_color='white', font_size=9,
            edge_color='gray')
    ax_g.set_title(f"Erdos-Renyi graph, N={N}, p={p_edge}, connected")
    fig_g.tight_layout()
    fig_g.savefig(os.path.join(OUT_DIR, "graph.png"), dpi=150)
    plt.close(fig_g)

    # ------------- animation (GIF via Pillow) -------------
    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_xlim(-1.8, 1.8); ax.set_ylim(-1.8, 1.8); ax.set_aspect('equal')
    ax.set_title("Formation control : KRISHNA")

    edges_lines = []
    for (i, j) in G.edges():
        line, = ax.plot([], [], color='lightgray', lw=0.6, zorder=1)
        edges_lines.append((i, j, line))
    scat = ax.scatter([], [], s=45, c='tab:blue', zorder=2, edgecolors='k',
                      linewidths=0.5)
    letter_label = ax.text(0.02, 0.95, "", transform=ax.transAxes,
                           fontsize=14, weight='bold')

    step = 4  # sub-sample frames so the gif is not huge
    frame_ids = list(range(0, len(frames), step))

    def which_letter(k):
        for li in range(len(letter_boundaries)-1):
            if letter_boundaries[li] <= k < letter_boundaries[li+1]:
                return name[li]
        return name[-1]

    def update(idx):
        k = frame_ids[idx]
        Pk = frames[k]
        scat.set_offsets(Pk)
        for (i, j, line) in edges_lines:
            line.set_data([Pk[i,0], Pk[j,0]], [Pk[i,1], Pk[j,1]])
        letter_label.set_text(f"letter: {which_letter(k)}")
        return [scat, letter_label] + [ln for _, _, ln in edges_lines]

    ani = animation.FuncAnimation(fig, update, frames=len(frame_ids),
                                  interval=40, blit=False)
    out_gif = os.path.join(OUT_DIR, "krishna_formation.gif")
    ani.save(out_gif, writer=animation.PillowWriter(fps=25))
    plt.close(fig)
    print(f"saved animation to {out_gif}")

    # ------------- still shots of every letter in the sequence -------------
    fig_s, axes = plt.subplots(1, len(name), figsize=(2.6*len(name), 3.0))
    if len(name) == 1:
        axes = [axes]
    for li, L in enumerate(name):
        ax = axes[li]
        k = letter_boundaries[li+1] - 1     # end of that letter's phase
        Pk = frames[k]
        for (i, j) in G.edges():
            ax.plot([Pk[i,0], Pk[j,0]], [Pk[i,1], Pk[j,1]], color='lightgray', lw=0.5)
        ax.scatter(Pk[:,0], Pk[:,1], s=32, c='tab:blue', edgecolors='k', linewidths=0.4)
        ax.set_title(f"letter {L}")
        ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2); ax.set_aspect('equal')
        ax.set_xticks([]); ax.set_yticks([])
    fig_s.tight_layout()
    fig_s.savefig(os.path.join(OUT_DIR, "letter_stills.png"), dpi=150)
    plt.close(fig_s)

    # ------------- formation error plot -------------
    fig_e, ax_e = plt.subplots(figsize=(8, 3))
    ax_e.plot(formation_error_hist, color='tab:red')
    for b in letter_boundaries[1:-1]:
        ax_e.axvline(b, color='k', lw=0.4, ls='--')
    ax_e.set_xlabel("iteration"); ax_e.set_ylabel("max_i || p_i - r_i ||")
    ax_e.set_title("formation error across the sequence")
    fig_e.tight_layout()
    fig_e.savefig(os.path.join(OUT_DIR, "formation_error.png"), dpi=150)
    plt.close(fig_e)

if __name__ == "__main__":
    main()
