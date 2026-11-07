#!/usr/bin/env bash
# Reproduce every experiment in the paper. CPU only.
# Main text + D.7/D.8: ~5 min. Appendix D.9-D.13: roughly 2-3 hours on one core.
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$PWD"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
OUT=$(realpath -m "${1:-results}")
mkdir -p "$OUT/appendix" "$OUT/distillation"

echo "== Main text and Appendix D.7-D.8 =="
python experiments/fig2_indefinite_quadratic.py "$OUT"
python experiments/fig3_width_sweep.py          "$OUT"
python experiments/fig4_pure_vs_mixed.py        "$OUT"
python experiments/fig5_gated_mlps.py           "$OUT"
python experiments/fig6_scope_boundary.py       "$OUT"
python experiments/fig7_matched_params.py       "$OUT"
python experiments/make_figures.py              "$OUT"

echo "== Appendix D.9 (border-rank diagnostics) and the D.10 F_{4,2} example =="
python experiments/d9_border_rank.py     --out "$OUT/d9"
python experiments/d10_f42_example.py    --out "$OUT/d10"

echo "== Appendix D.10, D.11, D.13 (original scripts; outputs go to $OUT/appendix) =="
( cd "$OUT/appendix"
  python "$ROOT/experiments/appendix/selftest.py"
  python "$ROOT/experiments/appendix/e_q4.py"      # Table 10
  python "$ROOT/experiments/appendix/e_q3.py"      # Table 6
  python "$ROOT/experiments/appendix/e_q1a.py"     # Table 7
  python "$ROOT/experiments/appendix/e_q1b.py" )   # Table 8

echo "== Appendix D.12 (distillation; outputs go to $OUT/distillation) =="
( cd "$ROOT/experiments/distillation"
  python teachers.py
  python distill.py "$OUT/distillation/results_cr.json"
  python norm_continuation.py "$OUT/distillation/norm_continuation.json"
  python table9.py "$OUT/distillation/results_cr.json"
  mv -f teachers.json "$OUT/distillation/" )
