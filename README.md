# AI3403 Multi-Agent Systems, Assignment 3

Krishna Teja, CS23BTECH11028.

Code, animation and figures for the four problems of Assignment 3.

## Folders

```
p1_formation/       Problem 1 (formation control, spells KRISHNA)
  problem1_formation.py
  krishna_formation.gif       the animation
  graph.png, letter_stills.png, formation_error.png

p2_dse/             Problem 2 (distributed state estimation, DGD)
  problem2_dse.py
  graph.png, estimate.png, coord_trace.png

p3_admm_task/       Problem 3 (ADMM task allocation)
  problem3_admm_taskalloc.py
  graph.png, allocations.png, convergence.png

p4_lasso/           Problem 4 (LASSO by ADMM)
  problem4_lasso_admm.py
  lasso.png, lasso_overlay.png

report/             cs23btech11028_Assignment3.pdf and the tex source
```

## Run

```
pip install -r requirements.txt
python p1_formation/problem1_formation.py
python p2_dse/problem2_dse.py
python p3_admm_task/problem3_admm_taskalloc.py
python p4_lasso/problem4_lasso_admm.py
```

Each script drops its figures next to itself. Problem 1 also writes
`krishna_formation.gif`.
