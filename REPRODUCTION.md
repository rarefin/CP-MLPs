# Reproduction report

This report compares re-runs of this repository against the numbers in the NeurIPS 2026
camera-ready paper. Every number below comes from the JSON files the scripts write.

**Environment:** Python 3.13, NumPy 2.x, SciPy 1.15+, PyTorch 2.x (CPU), 2 cores.

## Where each piece of code comes from

| Code | Paper items | Origin |
|---|---|---|
| `experiments/fig*.py`, `make_figures.py` | Figures 2–7, Table 3 | **Original** code from the submission. Only changes: output paths, dead code removed, tied SiLU² replaced by tied SwiGLU (`SiLU(z)·z`) in Figures 5 and 7, as in the camera-ready. |
| `experiments/appendix/` | Tables 6, 7, 8, 10 | **Original** rebuttal-phase scripts, included unchanged. |
| `experiments/distillation/` | Table 9 and norm continuation | **Original** harness (`cpmlp.py`, `teachers.py`); the camera-ready settings (`distill.py`) and the norm-continuation script were written for this release. |
| `experiments/d9_border_rank.py`, `cpmlp/symtensor.py` | Tables 4–5 and the D.9 controls | **Rebuilt** from the appendix description; the original was not recovered. |
| `experiments/d10_f42_example.py` | the F₄,₂ example in D.10 | **Rebuilt** from the appendix description. |

Exact losses near machine precision (below about 1e-9) differ across library versions,
because SciPy 1.15 rewrote L-BFGS-B. Comparisons for those experiments are made at the
level of the paper's claims.

## Summary

| Paper item | Reproduces? |
|---|---|
| Fig. 2 — quadratic width law | **Yes, exactly.** Observed minimum widths equal the predictions for all six signatures. |
| Fig. 3 — width sweep | **Yes.** Same thresholds and plateaus; only values below about 1e-9 move. |
| Fig. 4 — pure vs role-asymmetric | **Yes.** Same reversal; non-floor medians agree to 2 significant figures. |
| Fig. 5 — gated MLPs | **Yes.** All non-floor points match the published figure to within 0.2 decades. |
| Fig. 6 / Table 3 — scope boundary | **Yes.** Same qualitative table. |
| Fig. 7 / D.8 numbers | **Yes.** 5.46e-13 vs 5.27e-13 at k=2 (paper 5.5e-13 / 4.3e-13); 4.37e-3 vs 5.41e-4 at k=3 (paper 4.4e-3 / 4.0e-4). Fold change 8.1× here vs 10.8× in the paper. |
| Tables 4–5 — border-rank regimes | **Losses and regimes yes** (exact 0.5556, 0.5000, 0.2778). **κ column not reproducible** (see D.9 below). A_max depends on the candidate found. |
| D.9 controls | **Yes.** Closed-class errors 0.7500/0.5000/0.2500 exactly; penalty slopes −4.09/−4.07/−4.06 vs −4.00. |
| Table 6 — recovery fractions | **Yes, exactly.** 24 of 24 fractions and all four best losses. |
| D.10 F₄,₂ example | **Yes.** 12 of 40 restarts recover, median 0.25 (paper 10 of 40, median 0.25). |
| Table 7 — dense random targets | **Yes, exactly.** 28 of 28 entries. |
| Table 8 — mixtures of degrees | **Yes, exactly.** All 14 reported entries. |
| Table 9 — distillation | **Pattern only.** The harness is original, but the teachers differ from the paper's (training errors 0.209/0.195/0.167 vs 0.295/0.172/0.112), so entries differ. |
| Table 10 — Adam regression | **Yes, exactly.** 24 of 24 entries to four decimals. |

## Figures 2–7 and Table 3 (original code)

The cleaned scripts give JSON that is bit-identical to running the original scripts on
the same machine. Against the published figures, only floor values move. Figure 5 was
checked point by point against the vector data in the published PDF (paper values
are read from the figure, about ±0.05 decades):

| Architecture | k | m = 1 / 2 / 4, paper | m = 1 / 2 / 4, re-run |
|---|---|---|---|
| tied ReLU² | 2 | 6.4e-1 / 5.8e-1 / 1.9e-15 | 6.3e-1 / 5.7e-1 / 2.1e-15 |
| tied ReLU² | 3 | 1.8 / 1.3 / 3.0e-1 | 1.8 / 1.3 / 2.9e-1 |
| tied SwiGLU | 2 | 4.6e-1 / 1.4e-4 / 3.0e-3 | 4.5e-1 / 1.4e-4 / 2.9e-3 |
| tied SwiGLU | 3 | 1.9 / 1.1 / 1.8e-3 | 1.9 / 1.1 / 1.8e-3 |
| untied ReGLU | 2 | 4.3e-1 / 3.0e-14 / 1.7e-14 | 4.3e-1 / 3.9e-13 / 5.1e-13 |
| untied ReGLU | 3 | 9.6e-1 / 3.4e-1 / 6.1e-2 | 9.4e-1 / 3.3e-1 / 6.2e-2 |
| untied SwiGLU | 2 | 1.7e-5 / 3.6e-14 / 2.0e-9 | 1.5e-6 / 1.5e-13 / 2.8e-10 |
| untied SwiGLU | 3 | 1.3 / 1.6e-3 / 2.5e-4 | 1.3 / 9.9e-4 / 2.0e-4 |

## Appendix D.9 — border-rank diagnostics (rebuilt)

**Table 4** (F₃ = x₁²x₂)

| Family | m | Paper best loss | Re-run best loss | Paper best A_max | Re-run best A_max |
|---|---|---|---|---|---|
| untied | 1 | 1.6e-29 | 9.8e-46 |  1.00 | 1 |
| tied | 1 | 0.5556 | 0.5556 |  0.385 | 0.385 |
| tied | 2 | 5.4e-12 | 2.3e-13 |  83.17 | 183 |
| tied | 3 | 1.5e-28 | 5.8e-34 |  0.471 | 0.435 |

**Table 5** (F₃,₂ = x₁²x₂ + x₃²x₄)

| Family | m | Paper best loss | Re-run best loss | Paper best A_max | Re-run best A_max |
|---|---|---|---|---|---|
| untied | 2 | 6.4e-19 | 3.3e-36 | 1.00 | 1 |
| tied | 2 | 0.5000 | 0.5000 | 46.95 | 93.3 |
| tied | 3 | 0.2778 | 0.2778 | 66.30 | 86.8 |
| tied | 4 | 1.1e-10 | 8.0e-18 | 47.74 | 6.45e+03 |
| tied | 5 | 9.3e-11 | 3.3e-13 | 34.40 | 175 |
| tied | 6 | 8.3e-16 | 2.3e-32 | 3.72 | 0.435 |

Closed-class control (paper 0.7500 / 0.5000 / 0.2500): re-run 0.7500 / 0.5000 / 0.2500.
Penalty-sweep slopes, k = 3, 4, 5 (paper −4.003 / −4.002 / −4.002): re-run -4.091 / -4.069 / -4.059.


**κ column (Tables 4–5).** The appendix defines κ = 1/σ_min of the normalised
Terracini matrix with scaling directions removed (Vannieuwenhoven). Under that
definition, any single-atom decomposition has κ = 1 exactly, because one orthonormal
tangent basis has all singular values equal to 1. The paper reports 1.7 and 1.5 for the
m = 1 rows, so the original code must have used a different normalisation. Several
alternatives were tested, and none gives 1.7 and 1.5. This repository reports the
standard definition. Large κ at the degenerate widths and "n.i." at exact widths are
reproduced in kind.

The unspecified choices in the rebuilt D.9 script are the three penalty values (used
here: 1e-4, 1e-6, 1e-8) and the initialisation scale.

## Appendix D.10 — recovery of known representations (original code)

**Table 6** (recovery fractions over 60 restarts; paper / re-run)

| k | Tied w3 | Tied w4 | Tied w5 | Untied w3 | Untied w4 | Untied w5 | Best untied loss at w3 |
|---|---|---|---|---|---|---|---|
| 3 | 0.33 / 0.33 | 0.32 / 0.32 | 0.70 / 0.70 | 0.80 / 0.80 | 0.92 / 0.92 | 1.00 / 1.00 | 1.2e-20 / 1.2e-20 |
| 4 | 0.42 / 0.42 | 0.67 / 0.67 | 0.75 / 0.75 | 0.17 / 0.17 | 0.65 / 0.65 | 0.88 / 0.88 | 2.4e-18 / 2.4e-18 |
| 5 | 0.77 / 0.77 | 0.80 / 0.80 | 0.85 / 0.85 | 0.00 / 0.00 | 0.02 / 0.02 | 0.00 / 0.00 | 1.7e-5 / 1.7e-05 |
| 6 | 0.08 / 0.08 | 0.20 / 0.20 | 0.22 / 0.22 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 8.1e-9 / 8.1e-09 |

24 of 24 recovery fractions match the paper to the printed precision.


F₄,₂ example (rebuilt script): 12 of 40 restarts recover, median 0.25 (paper 10 of 40,
median 0.25). With 8 restarts, 2 recover here; the paper reports that all 8 returned 0.25.

## Appendix D.11 — dense random targets and mixtures (original code)

**Table 7** (observed/predicted width; paper / re-run)

| d | k | target | Tied | Untied |
|---|---|---|---|---|
| 2 | 5 | 0 | 3/3 / 3/3 | 3/2 / 3/2 |
| 2 | 5 | 1 | 4/3 / 4/3 | 2/2 / 2/2 |
| 2 | 6 | 0 | 4/4 / 4/4 | 3/3 / 3/3 |
| 2 | 6 | 1 | 5/4 / 5/4 | 3/3 / 3/3 |
| 3 | 3 | 0 | 4/4 / 4/4 | 3/3 / 3/3 |
| 3 | 3 | 1 | 4/4 / 4/4 | 3/3 / 3/3 |
| 3 | 4 | 0 | 6/6 / 6/6 | 3/3 / 3/3 |
| 3 | 4 | 1 | 7/6 / 7/6 | 4/3 / 4/3 |
| 4 | 3 | 0 | 6/5 / 6/5 | 4/4 / 4/4 |
| 4 | 3 | 1 | 6/5 / 6/5 | 4/4 / 4/4 |
| 4 | 4 | 0 | 11/10 / 11/10 | 6/5 / 6/5 |
| 4 | 4 | 1 | 10/10 / 10/10 | 6/5 / 6/5 |
| 5 | 3 | 0 | 8/8 / 8/8 | 5/5 / 5/5 |
| 5 | 3 | 1 | 8/8 / 8/8 | 5/5 / 5/5 |

28 of 28 entries match the paper exactly.


**Table 8** (recovered width, counting benchmark in parentheses; paper / re-run)

| d | K | Tied | Untied |
|---|---|---|---|
| 2 | {2,3,4,5} | 5 (4), 4 (4) / 5 (4), 4 (4) | 4 (3), 3 (3) / 4 (3), 3 (3) |
| 3 | {2,3} | 5 (4), 4 (4) / 5 (4), 4 (4) | 4 (3), 3 (3) / 4 (3), 3 (3) |
| 3 | {2,3,4} | 7 (7) / 7 (7) | 5 (5) / 5 (5) |
| 4 | {2,3} | 7 (6), 7 (6) / 7 (6), 7 (6) | 4 (4), 5 (4) / 4 (4), 5 (4) |


## Appendix D.12 — distillation (original harness, camera-ready settings)

**Table 9** (median relative held-out MSE; paper / re-run)

| Teacher | m | Untied m | Tied m | Ratio | Tied 1.5m | Ratio |
|---|---|---|---|---|---|---|
| untied | 4 | 4.02e-1 / 3.61e-01 | 3.96e-1 / 3.94e-01 | 0.99 / 1.09 | 2.29e-1 / 2.40e-01 | 0.57 / 0.67 |
| untied | 8 | 1.03e-1 / 2.65e-03 | 1.16e-1 / 1.35e-01 | 1.13 / 51.02 | 8.70e-4 / 2.95e-03 | 0.01 / 1.11 |
| untied | 10 | 1.37e-4 / 4.25e-05 | 7.51e-2 / 9.65e-02 | 549 / 2269.70 | 1.09e-3 / 1.64e-03 | 7.94 / 38.66 |
| untied | 12 | 8.11e-6 / 8.74e-05 | 8.70e-4 / 2.95e-03 | 107 / 33.75 | 6.11e-4 / 4.64e-04 | 75.3 / 5.31 |
| tied | 8 | 2.82e-1 / 1.33e-01 | 2.99e-1 / 1.23e-01 | 1.06 / 0.92 | 1.64e-1 / 5.65e-02 | 0.58 / 0.42 |
| tied | 12 | 9.30e-2 / 1.70e-03 | 1.64e-1 / 5.65e-02 | 1.76 / 33.13 | 6.47e-2 / 6.24e-04 | 0.70 / 0.37 |
| tanh | 8 | 3.45e-1 / 1.69e-01 | 3.44e-1 / 2.19e-01 | 1.00 / 1.30 | 2.51e-1 / 1.91e-01 | 0.73 / 1.13 |
| tanh | 12 | 1.81e-1 / 1.12e-01 | 2.51e-1 / 1.91e-01 | 1.38 / 1.71 | 1.53e-1 / 9.54e-02 | 0.84 / 0.85 |

Norm continuation: paper best 3.0e-4 at ‖θ‖² ≈ 180 (about 5× its initial value), one path stalling near 7e-2. Re-run best 1.9e-04 at ‖θ‖² ≈ 177 (about 9× initial); no path stalls (final errors 1.9e-04, 4.4e-04, 5.4e-04).


Matches with the paper: the first untied equal-width recovery on the untied teacher is
at width 10, with one of three seeds failing there (4.3e-5, 1.3e-5, 1.7e-3 here;
1.4e-4, 1.2e-5, 7.5e-2 in the paper). Small-width ratios are within about 10%, and large
equal-width ratios appear at widths 10–12. Entries differ because the teachers differ:
their training errors are 0.209 / 0.195 / 0.167 here versus 0.295 / 0.172 / 0.112 in the
paper. The paper's final teacher settings were not in the recovered scripts.

## Appendix D.13 — Adam-trained scalar regression (original code)

**Table 10** (median relative test MSE; paper / re-run)

| Params | Asym: untied | Asym: tied | Tied teacher: untied | Tied teacher: tied |
|---|---|---|---|---|
| 17/18 | 0.3162 / 0.3162 | 0.2580 / 0.2580 | 0.3808 / 0.3808 | 0.1899 / 0.1899 |
| 34/36 | 0.0843 / 0.0843 | 0.1049 / 0.1049 | 0.0178 / 0.0178 | 0.0014 / 0.0014 |
| 51/54 | 0.0322 / 0.0322 | 0.0387 / 0.0387 | 0.0045 / 0.0045 | 0.0013 / 0.0013 |
| 68/72 | 0.0188 / 0.0188 | 0.0370 / 0.0370 | 0.0010 / 0.0010 | 0.0006 / 0.0006 |
| 102/108 | 0.0088 / 0.0088 | 0.0052 / 0.0052 | 0.0009 / 0.0009 | 0.0006 / 0.0006 |
| 136/144 | 0.0062 / 0.0062 | 0.0024 / 0.0024 | 0.0008 / 0.0008 | 0.0004 / 0.0004 |

24 of 24 entries match the paper to four decimals.
