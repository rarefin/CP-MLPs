# Appendix D.12 — distillation of trained gated blocks

`cpmlp.py` (bias-free SwiGLU / tied SwiGLU blocks with analytic gradients) and
`teachers.py` (the frozen tanh target and the three 192-parameter teachers) are the
original rebuttal-phase harness. `distill.py` runs the camera-ready design on top of
it: 3 seeds, L-BFGS for up to 6,000 iterations, widths {2,3,4,6,8,10,12}, tied
students at width m and round(1.5 m). `norm_continuation.py` reproduces the
norm-continuation paragraph. `table9.py` prints the Table 9 summary.

Run from inside this folder:

```bash
python teachers.py            # writes teachers.json
python distill.py             # writes results_cr.json (resumable)
python norm_continuation.py   # writes norm_continuation.json
python table9.py results_cr.json
```

The teachers produced here have training errors 0.209 / 0.195 / 0.167, whereas the
paper's teachers had 0.295 / 0.172 / 0.112. The paper's final teacher settings were not
recovered, so Table 9 reproduces in pattern rather than entry by entry. See
REPRODUCTION.md.
