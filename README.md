# AI3403 Multi-Agent Systems, Assignment 3

Krishna Teja, roll no CS23BTECH11028
Instructor: Dr. Venkatraman Renganathan
Term: Autumn 2026

This repo has my code, figures and report for Assignment 3.

## Folders

```
p1_formation/       Problem 1 (formation control, spells KRISHNA)
  problem1_formation.py
  krishna_formation.gif     <- the animation the problem asked for
  graph.png                 ER graph used
  letter_stills.png         one still per letter
  formation_error.png

p2_dse/             Problem 2 (distributed state estimation, DGD)
  problem2_dse.py
  graph.png
  estimate.png              true vs estimated 3D + error norm
  coord_trace.png

p3_admm_task/       Problem 3 (ADMM task allocation)
  problem3_admm_taskalloc.py
  graph.png
  allocations.png           3 cost matrices + allocations
  convergence.png           residual + objective

p4_lasso/           Problem 4 (LASSO by ADMM)
  problem4_lasso_admm.py
  lasso.png
  lasso_overlay.png

report/             LaTeX source and the final PDF
  cs23btech11028_Assignment3.tex
  cs23btech11028_Assignment3.pdf
```

## How to run

Needs Python 3.10+. Install the depedencies:

```
pip install -r requirements.txt
```

Then run each script:

```
python p1_formation/problem1_formation.py
python p2_dse/problem2_dse.py
python p3_admm_task/problem3_admm_taskalloc.py
python p4_lasso/problem4_lasso_admm.py
```

Each script drops its figures next to itself. Problem 1 also writes
`krishna_formation.gif` which is the animation.

## Which lecture goes with what

- P1: formation control from MAS Algorithms lecture, Algorithm 2 on page 7.
- P2: distributed gradient descent from Distributed Learning lecture page 7,
  applied to the distributed state estimation motivating problem on pages 12-14.
- P3: consensus ADMM template from ADMM lecture page 8 and the "ADMM hack"
  from page 11. Augmented Lagrangian and scaled form from Helper notes.
- P4: variable splitting + scaled form ADMM from the Helper notes, and
  soft thresholding as the prox of the ell_1 norm.

## Notes

- All the Erdos-Renyi graphs are resampled until connected before I use them.
- For P2 I use Metropolis-Hastings weights for the mixing matrix W since
  the lecture only asks that W be doubly stochastic and compatible with
  the graph.
- For P3 I use the standard LP relaxation and round the ADMM output using
  the Hungarian algorithm on an expanded cost matrix (each agent duplicated
  b_i times). This gives a valid binary allocation that satisfies both the
  task cover constraint and the capacity constraint.
