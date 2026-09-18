# AI3403 Multi-Agent Systems, Assignment 3

**Krishna Teja** ,  Roll no.  **CS23BTECH11028**
Instructor : Dr. Venkatraman Renganathan
Term : Autumn 2026
Due : 18 September 2026

This repository contains the code, generated figures and animation for
Assignment 3 of AI3403 Multi-Agent Systems at IIT Hyderabad.

## Layout

```
p1_formation/     Problem 1 : Formation Control ( name KRISHNA )
    problem1_formation.py       formation control simulation
    graph.png                   Erdos-Renyi communication graph
    krishna_formation.gif       the animation asked in the problem
    letter_stills.png           still shots of every letter K R I S H N A
    formation_error.png         max formation error vs iteration

p2_dse/           Problem 2 : Distributed State Estimation via DGD
    problem2_dse.py             distributed estimator using DGD
    graph.png                   communication graph of the drones
    estimate.png                true vs estimated 3D trajectory + error norm
    coord_trace.png             per coordinate (x,y,z) time series

p3_admm_task/     Problem 3 : Task allocation via consensus ADMM
    problem3_admm_taskalloc.py  binary LP, LP relaxation, distributed ADMM
    graph.png                   communication graph
    allocations.png             three cost matrices and their allocations
    convergence.png             primal residual and objective vs iter

p4_lasso/         Problem 4 : LASSO by ADMM
    problem4_lasso_admm.py      closed form x update + soft thresholding z update
    lasso.png                   objective, residuals, true vs recovered
    lasso_overlay.png           overlay of true x and ADMM estimate z

report/           LaTeX source and compiled PDF of the report
```

## How to run

Requires Python 3.10 or newer. Install the dependencies with

```
pip install -r requirements.txt
```

Then

```
python p1_formation/problem1_formation.py
python p2_dse/problem2_dse.py
python p3_admm_task/problem3_admm_taskalloc.py
python p4_lasso/problem4_lasso_admm.py
```

Each script writes its own PNG figures next to itself. Problem 1 also
writes `krishna_formation.gif` which is the animation asked in
the problem statement.

## References to the lecture material

- Problem 1 uses Algorithm 2 (Distributed Formation Control) from the
  MAS Algorithms lecture, page 7.
- Problem 2 uses the DGD update from the Distributed Learning lecture,
  page 7, applied to the distributed state estimation motivating problem
  (page 12 to 14 of that same lecture).
- Problem 3 uses the consensus ADMM template from the ADMM lecture,
  page 8, and the "ADMM hack for any distributed decision making problem"
  from the same lecture, page 11. The augmented Lagrangian and the scaled
  form are taken from the Helper notes, pages 12 to 14.
- Problem 4 uses variable splitting and scaled form ADMM from the
  Helper notes, pages 11 to 14, and the soft thresholding proximal
  operator from Helper notes, page 5.

## Notes

- All Erdos-Renyi graphs are resampled until connected before being used.
- Metropolis-Hastings weights are used for the mixing matrix in Problem 2
  because the lecture does not fix a specific formula, only asks that W be
  doubly stochastic and compatible with the graph.
- Problem 3 uses the standard LP relaxation and a Hungarian based rounding
  step on an expanded cost matrix so that the final allocation is binary
  and satisfies both the capacity and the task cover constraint. The
  relaxation and rounding are declared explicitly in the report.
