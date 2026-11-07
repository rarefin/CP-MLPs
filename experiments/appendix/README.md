# Appendix D.10, D.11, D.13 — original rebuttal-phase scripts

These are the scripts that produced Tables 6, 7, 8 and 10. They were recovered from
the rebuttal sessions and are included unchanged. Each script writes a resumable
JSON checkpoint into the **current directory** and skips cells that are already done.

| Script | Paper | Output |
|---|---|---|
| `selftest.py` | harness checks (run first) | — |
| `e_q3.py` | D.10, Table 6 — recovery of known representations | `q3.json` |
| `e_q1a.py` | D.11, Table 7 — dense random targets | `q1a.json` |
| `e_q1b.py` | D.11, Table 8 — mixtures of degrees | `q1b.json` |
| `e_q4.py` | D.13, Table 10 — Adam-trained gated scalar regression | `q4.json` |

`cpcore.py` is the shared coefficient-space harness (apolar norm, tied/untied atoms,
generic-rank formulas, L-BFGS with PyTorch autograd). `gated.py` provides genuine
SwiGLU channel components and is used by `selftest.py`.

Notes:

- The "tied" student and teacher in `e_q4.py` compute `SiLU(z)·z` (tied SwiGLU), as in
  the camera-ready paper.
- `e_q1b.py` also runs a fifth case, `(n, K) = (3, {3, 4})`, and a second target for
  `(3, {2, 3, 4})`. Neither is reported in Table 8.
- In these scripts `n` is the number of variables (the paper's `d`).
